// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

// Benchmark of IvfIndex and HnswIndex on .fbin / .i8bin vector files (header: uint32 n, uint32 dims).

#include <vespa/eval/eval/value_type.h>
#include <vespa/searchlib/common/bitvector.h>
#include <vespa/searchlib/queryeval/global_filter.h>
#include <vespa/searchlib/tensor/distance_function_factory.h>
#include <vespa/searchlib/tensor/doc_vector_access.h>
#include <vespa/searchlib/tensor/empty_subspace.h>
#include <vespa/searchlib/tensor/hnsw_index.h>
#include <vespa/searchlib/tensor/inv_log_level_generator.h>
#include <vespa/searchlib/tensor/ivf_index.h>
#include <vespa/searchlib/tensor/subspace_type.h>
#include <vespa/searchlib/tensor/vector_bundle.h>
#include <vespa/vespalib/data/slime/slime.h>
#include <vespa/vespalib/datastore/compaction_strategy.h>
#include <vespa/vespalib/net/http/state_explorer.h>
#include <vespa/vespalib/util/deadline.h>
#include <vespa/vespalib/util/threadstackexecutor.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <vespa/log/log.h>
LOG_SETUP("ivf_index_bench");

using namespace search::tensor;
using search::attribute::DistanceMetric;
using search::queryeval::GlobalFilter;
using vespalib::Deadline;
using vespalib::GenerationGuard;
using vespalib::eval::CellType;
using vespalib::eval::TypedCells;
using vespalib::eval::ValueType;
using Neighbor = NearestNeighborIndex::Neighbor;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double   max_distance = std::numeric_limits<double>::max();
constexpr uint32_t gt_k = 100;
constexpr uint32_t commit_interval = 256;
constexpr uint32_t prepare_batch = 1024;
constexpr uint32_t gt_query_batch = 128;
// Vespa rank profile defaults for filtered HNSW search.
constexpr double filter_first_upper_limit = 0.2;
constexpr double filter_first_exploration = 0.01;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

class Args {
    std::map<std::string, std::string> _kv;

public:
    Args(int argc, char** argv) : _kv() {
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            auto        eq = arg.find('=');
            if (arg.rfind("--", 0) != 0 || eq == std::string::npos) {
                throw std::invalid_argument("expected --key=value, got " + arg);
            }
            _kv[arg.substr(2, eq - 2)] = arg.substr(eq + 1);
        }
    }
    const std::string* find(const std::string& key) const {
        auto it = _kv.find(key);
        return (it == _kv.end()) ? nullptr : &it->second;
    }
    std::string str(const std::string& key, const std::string& def) const {
        auto it = _kv.find(key);
        return (it == _kv.end()) ? def : it->second;
    }
    std::string required(const std::string& key) const {
        auto it = _kv.find(key);
        if (it == _kv.end()) {
            throw std::invalid_argument("missing --" + key);
        }
        return it->second;
    }
    uint64_t num(const std::string& key, uint64_t def) const { return std::stoull(str(key, std::to_string(def))); }
    double real(const std::string& key, double def) const { return std::stod(str(key, std::to_string(def))); }
    std::vector<uint32_t> list(const std::string& key, const std::string& def) const {
        std::vector<uint32_t> result;
        std::string           value = str(key, def);
        size_t                pos = 0;
        while (pos < value.size()) {
            size_t comma = value.find(',', pos);
            comma = (comma == std::string::npos) ? value.size() : comma;
            result.push_back(std::stoul(value.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        return result;
    }
};

void read_fully(int fd, char* buf, size_t len, const std::string& path) {
    size_t done = 0;
    while (done < len) {
        ssize_t got = ::read(fd, buf + done, len - done);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            throw std::runtime_error(path + ": read failed: " + (got == 0 ? "end of file" : std::strerror(errno)));
        }
        done += got;
    }
}

struct VectorFile {
    CellType          cell_type;
    uint32_t          n;
    uint32_t          dims;
    size_t            vector_bytes;
    std::vector<char> data;
    const char* vec(uint32_t i) const noexcept { return data.data() + size_t(i) * vector_bytes; }
};

VectorFile load_vectors(const std::string& path, uint32_t max_n) {
    VectorFile file{CellType::FLOAT, 0, 0, 0, {}};
    bool       is_float = path.size() > 5 && path.substr(path.size() - 5) == ".fbin";
    file.cell_type = is_float ? CellType::FLOAT : CellType::INT8;
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(path + ": open failed: " + std::strerror(errno));
    }
    uint32_t header[2];
    read_fully(fd, reinterpret_cast<char*>(header), sizeof(header), path);
    file.n = std::min(header[0], max_n);
    file.dims = header[1];
    file.vector_bytes = size_t(file.dims) * (is_float ? sizeof(float) : 1);
    file.data.resize(size_t(file.n) * file.vector_bytes);
    read_fully(fd, file.data.data(), file.data.size(), path);
    if (::close(fd) != 0) {
        throw std::runtime_error(path + ": close failed: " + std::strerror(errno));
    }
    fprintf(stderr, "loaded %s: n=%u dims=%u (file has %u)\n", path.c_str(), file.n, file.dims, header[0]);
    return file;
}

// Appends the rows of a pool file after the rows of base; returns the first pool row. base.n stays the doc count.
uint32_t append_pool(VectorFile& base, const std::string& path) {
    auto pool = load_vectors(path, std::numeric_limits<uint32_t>::max());
    if (pool.dims != base.dims || pool.cell_type != base.cell_type) {
        throw std::invalid_argument(path + ": pool vectors differ in type from base");
    }
    size_t first_row = base.data.size() / base.vector_bytes;
    if (first_row + pool.n > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(path + ": too many rows");
    }
    base.data.insert(base.data.end(), pool.data.begin(), pool.data.end());
    return static_cast<uint32_t>(first_row);
}

// Docid d (1-based) maps to vector doc_to_vec[d] of the base file.
class BenchVectors : public DocVectorAccess {
    const VectorFile&     _base;
    SubspaceType          _type;
    EmptySubspace         _empty;
    std::vector<uint32_t> _doc_to_vec;

public:
    explicit BenchVectors(const VectorFile& base)
        : _base(base),
          _type(ValueType::make_type(base.cell_type, {{"x", base.dims}})),
          _empty(_type),
          _doc_to_vec(base.n + 1) {
        for (uint32_t docid = 1; docid <= base.n; ++docid) {
            _doc_to_vec[docid] = docid - 1;
        }
    }
    ~BenchVectors() override;
    uint32_t vec_of(uint32_t docid) const noexcept { return _doc_to_vec[docid]; }
    void swap_vectors(uint32_t a, uint32_t b) noexcept { std::swap(_doc_to_vec[a], _doc_to_vec[b]); }
    void set_vector(uint32_t docid, uint32_t row) noexcept { _doc_to_vec[docid] = row; }
    TypedCells cells(uint32_t docid) const noexcept {
        return {_base.vec(_doc_to_vec[docid]), _base.cell_type, _base.dims};
    }
    TypedCells get_vector(uint32_t docid, uint32_t subspace) const noexcept override {
        return (subspace == 0 && docid >= 1 && docid <= _base.n) ? cells(docid) : _empty.cells();
    }
    VectorBundle get_vectors(uint32_t docid) const noexcept override {
        if (docid < 1 || docid > _base.n) {
            return {};
        }
        return {_base.vec(_doc_to_vec[docid]), 1, _type};
    }
};

BenchVectors::~BenchVectors() = default;

// Per base row: (bucket, topic) labels from a labels.u16bin file (header uint32 n, uint32 2).
std::vector<uint16_t> load_labels(const std::string& path, uint32_t n) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(path + ": open failed: " + std::strerror(errno));
    }
    uint32_t header[2];
    read_fully(fd, reinterpret_cast<char*>(header), sizeof(header), path);
    if (header[1] != 2 || header[0] < n) {
        ::close(fd);
        throw std::runtime_error(path + ": expected at least " + std::to_string(n) + " rows of 2 labels");
    }
    std::vector<uint16_t> labels(size_t(n) * 2);
    read_fully(fd, reinterpret_cast<char*>(labels.data()), labels.size() * sizeof(uint16_t), path);
    if (::close(fd) != 0) {
        throw std::runtime_error(path + ": close failed: " + std::strerror(errno));
    }
    return labels;
}

DistanceMetric parse_metric(const std::string& name) {
    if (name == "euclidean") {
        return DistanceMetric::Euclidean;
    } else if (name == "angular") {
        return DistanceMetric::Angular;
    } else if (name == "prenormalized-angular") {
        return DistanceMetric::PrenormalizedAngular;
    } else if (name == "hamming") {
        return DistanceMetric::Hamming;
    }
    throw std::invalid_argument("unknown metric " + name);
}

struct GroundTruth {
    uint32_t              nq;
    uint32_t              n_base;
    std::vector<uint32_t> ids;   // nq * gt_k vector indexes, by increasing distance
    std::vector<double>   dists; // nq * gt_k
    GroundTruth();
    GroundTruth(uint32_t nq_in, uint32_t n_base_in);
    GroundTruth(GroundTruth&&) noexcept;
    GroundTruth& operator=(GroundTruth&&) noexcept;
    ~GroundTruth();
};

GroundTruth::GroundTruth() : nq(0), n_base(0), ids(), dists() {
}
GroundTruth::GroundTruth(uint32_t nq_in, uint32_t n_base_in)
    : nq(nq_in), n_base(n_base_in), ids(size_t(nq_in) * gt_k), dists(size_t(nq_in) * gt_k, max_distance) {
}
GroundTruth::GroundTruth(GroundTruth&&) noexcept = default;
GroundTruth& GroundTruth::operator=(GroundTruth&&) noexcept = default;
GroundTruth::~GroundTruth() = default;

using Heap = std::vector<std::pair<double, uint32_t>>;

void push_bounded(Heap& heap, double distance, uint32_t id) {
    if (heap.size() < gt_k) {
        heap.emplace_back(distance, id);
        std::push_heap(heap.begin(), heap.end());
    } else if (distance < heap.front().first) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {distance, id};
        std::push_heap(heap.begin(), heap.end());
    }
}

GroundTruth compute_gt(const VectorFile& base, const VectorFile& queries, const DistanceFunctionFactory& dff,
                       uint32_t threads) {
    auto                           start = Clock::now();
    std::vector<std::vector<Heap>> heaps(threads, std::vector<Heap>(queries.n));
    std::vector<std::thread>       workers;
    for (uint32_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            uint32_t lo = uint64_t(base.n) * t / threads;
            uint32_t hi = uint64_t(base.n) * (t + 1) / threads;
            for (uint32_t q0 = 0; q0 < queries.n; q0 += gt_query_batch) {
                uint32_t                               q1 = std::min(queries.n, q0 + gt_query_batch);
                std::vector<BoundDistanceFunction::UP> dfs;
                for (uint32_t q = q0; q < q1; ++q) {
                    dfs.push_back(dff.for_query_vector({queries.vec(q), queries.cell_type, queries.dims}));
                }
                for (uint32_t i = lo; i < hi; ++i) {
                    TypedCells cells(base.vec(i), base.cell_type, base.dims);
                    for (uint32_t q = q0; q < q1; ++q) {
                        push_bounded(heaps[t][q], dfs[q - q0]->calc(cells), i);
                    }
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    GroundTruth gt(queries.n, base.n);
    for (uint32_t q = 0; q < queries.n; ++q) {
        Heap merged;
        for (uint32_t t = 0; t < threads; ++t) {
            for (const auto& entry : heaps[t][q]) {
                push_bounded(merged, entry.first, entry.second);
            }
        }
        std::sort_heap(merged.begin(), merged.end());
        for (size_t j = 0; j < merged.size(); ++j) {
            gt.dists[size_t(q) * gt_k + j] = merged[j].first;
            gt.ids[size_t(q) * gt_k + j] = merged[j].second;
        }
    }
    fprintf(stderr, "ground truth: %u queries x %u base in %.1fs with %u threads\n", queries.n, base.n,
            seconds_since(start), threads);
    return gt;
}

GroundTruth compute_gt_live(const BenchVectors& vectors, uint32_t n, const VectorFile& queries,
                            const DistanceFunctionFactory& dff, uint32_t threads) {
    auto                           start = Clock::now();
    std::vector<std::vector<Heap>> heaps(threads, std::vector<Heap>(queries.n));
    std::vector<std::thread>       workers;
    for (uint32_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            uint32_t lo = 1 + uint64_t(n) * t / threads;
            uint32_t hi = 1 + uint64_t(n) * (t + 1) / threads;
            for (uint32_t q0 = 0; q0 < queries.n; q0 += gt_query_batch) {
                uint32_t                               q1 = std::min(queries.n, q0 + gt_query_batch);
                std::vector<BoundDistanceFunction::UP> dfs;
                for (uint32_t q = q0; q < q1; ++q) {
                    dfs.push_back(dff.for_query_vector({queries.vec(q), queries.cell_type, queries.dims}));
                }
                for (uint32_t docid = lo; docid < hi; ++docid) {
                    TypedCells cells = vectors.cells(docid);
                    for (uint32_t q = q0; q < q1; ++q) {
                        push_bounded(heaps[t][q], dfs[q - q0]->calc(cells), docid - 1);
                    }
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    GroundTruth gt(queries.n, n);
    for (uint32_t q = 0; q < queries.n; ++q) {
        Heap merged;
        for (uint32_t t = 0; t < threads; ++t) {
            for (const auto& entry : heaps[t][q]) {
                push_bounded(merged, entry.first, entry.second);
            }
        }
        std::sort_heap(merged.begin(), merged.end());
        for (size_t j = 0; j < merged.size(); ++j) {
            gt.dists[size_t(q) * gt_k + j] = merged[j].first;
            gt.ids[size_t(q) * gt_k + j] = merged[j].second;
        }
    }
    fprintf(stderr, "live ground truth: %u queries x %u docs in %.1fs\n", queries.n, n, seconds_since(start));
    return gt;
}

void save_gt(const GroundTruth& gt, const std::string& path) {
    std::ofstream out(path + ".part", std::ios::binary);
    uint32_t      header[3] = {gt.nq, gt.n_base, gt_k};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(gt.ids.data()), gt.ids.size() * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(gt.dists.data()), gt.dists.size() * sizeof(double));
    out.close();
    if (!out) {
        throw std::runtime_error(path + ": write failed");
    }
    if (::rename((path + ".part").c_str(), path.c_str()) != 0) {
        throw std::runtime_error(path + ": rename failed: " + std::strerror(errno));
    }
}

bool load_gt(const std::string& path, uint32_t nq, uint32_t n_base, GroundTruth& gt) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    uint32_t header[3];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || header[0] < nq || header[1] != n_base || header[2] != gt_k) {
        fprintf(stderr, "%s: header (%u,%u,%u) does not match (%u,%u,%u), recomputing\n", path.c_str(), header[0],
                header[1], header[2], nq, n_base, gt_k);
        return false;
    }
    gt = GroundTruth(nq, n_base);
    in.read(reinterpret_cast<char*>(gt.ids.data()), gt.ids.size() * sizeof(uint32_t));
    in.seekg(sizeof(header) + size_t(header[0]) * gt_k * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(gt.dists.data()), gt.dists.size() * sizeof(double));
    if (!in) {
        throw std::runtime_error(path + ": short read");
    }
    return true;
}

// One query workload: no filter, or a filter spec "<bucket|topic>:<limit>" that keeps docs whose label is below
// the limit. Mirrors Vespa's plan: exact search when the hit ratio is below the approximate threshold.
struct FilterCase {
    std::string                   spec;
    GroundTruth                   gt;
    std::shared_ptr<GlobalFilter> filter;
    std::vector<uint32_t>         matching;
    double                        hit_ratio;
    bool                          exact;
    explicit FilterCase(std::string spec_in);
    FilterCase(FilterCase&&) noexcept;
    ~FilterCase();
};

FilterCase::FilterCase(std::string spec_in)
    : spec(std::move(spec_in)), gt(), filter(), matching(), hit_ratio(1.0), exact(false) {
}
FilterCase::FilterCase(FilterCase&&) noexcept = default;
FilterCase::~FilterCase() = default;

// Recomputes ground truth distances with the bench's distance function (external tools may round differently).
void recompute_gt_dists(GroundTruth& gt, const VectorFile& queries, const BenchVectors& vectors,
                        const DistanceFunctionFactory& dff) {
    for (uint32_t q = 0; q < gt.nq; ++q) {
        auto df = dff.for_query_vector({queries.vec(q), queries.cell_type, queries.dims});
        for (uint32_t j = 0; j < gt_k; ++j) {
            size_t i = size_t(q) * gt_k + j;
            if (gt.dists[i] != max_distance) {
                gt.dists[i] = df->calc(vectors.cells(gt.ids[i] + 1));
            }
        }
        std::sort(gt.dists.begin() + size_t(q) * gt_k, gt.dists.begin() + size_t(q + 1) * gt_k);
    }
}

FilterCase make_filter_case(const std::string& spec, const std::vector<uint16_t>& labels, const BenchVectors& vectors,
                            uint32_t n, uint32_t nq, const std::string& main_gt, const std::string& filter_dir,
                            double approximate_threshold, const VectorFile& queries,
                            const DistanceFunctionFactory& dff) {
    FilterCase result(spec);
    if (spec == "none") {
        if (!load_gt(main_gt, nq, n, result.gt)) {
            throw std::runtime_error(main_gt + ": cannot load ground truth");
        }
        return result;
    }
    auto colon = spec.find(':');
    if (colon == std::string::npos) {
        throw std::invalid_argument("filter must be none or <bucket|topic>:<limit>");
    }
    std::string field = spec.substr(0, colon);
    uint32_t    column = (field == "bucket") ? 0 : (field == "topic") ? 1 : 2;
    if (column > 1) {
        throw std::invalid_argument("unknown filter field " + field);
    }
    std::string limit_text = spec.substr(colon + 1);
    uint32_t    limit = std::stoul(limit_text);
    std::string gt_path = filter_dir + "/gt_" + field + "_" + limit_text + ".gt100";
    if (!load_gt(gt_path, std::min(nq, 1000u), n, result.gt)) {
        throw std::runtime_error(gt_path + ": cannot load ground truth");
    }
    recompute_gt_dists(result.gt, queries, vectors, dff);
    auto bits = search::BitVector::create(n + 1);
    for (uint32_t docid = 1; docid <= n; ++docid) {
        if (labels[size_t(vectors.vec_of(docid)) * 2 + column] < limit) {
            bits->setBit(docid);
            result.matching.push_back(docid);
        }
    }
    bits->invalidateCachedCount();
    result.filter = GlobalFilter::create(std::move(bits));
    result.hit_ratio = double(result.matching.size()) / n;
    result.exact = result.hit_ratio < approximate_threshold;
    return result;
}

std::vector<Neighbor> exact_top_k(const BenchVectors& vectors, const std::vector<uint32_t>& matching, uint32_t k,
                                  NearestNeighborIndex::Stats& stats, const BoundDistanceFunction& df) {
    std::vector<Neighbor> all;
    all.reserve(matching.size());
    for (uint32_t docid : matching) {
        all.emplace_back(docid, df.calc(vectors.cells(docid)));
        stats.count_computed_distance();
    }
    size_t keep = std::min<size_t>(k, all.size());
    std::partial_sort(all.begin(), all.begin() + keep, all.end(),
                      [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
    all.resize(keep);
    return all;
}

// Mirrors TensorAttribute::onCommit: new generation, index maintenance, another generation if it changed anything.
void commit(NearestNeighborIndex& index) {
    index.inc_generation();
    if (index.consider_compact(vespalib::datastore::CompactionStrategy())) {
        index.inc_generation();
    }
    index.reclaim_unused_memory();
}

double build(NearestNeighborIndex& index, const BenchVectors& vectors, uint32_t n, uint32_t prepare_threads) {
    auto start = Clock::now();
    if (prepare_threads == 0) {
        for (uint32_t docid = 1; docid <= n; ++docid) {
            index.add_document(docid);
            if ((docid % commit_interval) == 0) {
                commit(index);
            }
        }
    } else {
        std::vector<std::unique_ptr<PrepareResult>> prepared(prepare_batch);
        for (uint32_t first = 1; first <= n; first += prepare_batch) {
            uint32_t                 count = std::min(prepare_batch, n - first + 1);
            std::vector<std::thread> workers;
            for (uint32_t t = 0; t < prepare_threads; ++t) {
                workers.emplace_back([&, t]() {
                    for (uint32_t j = t; j < count; j += prepare_threads) {
                        prepared[j] =
                            index.prepare_add_document(first + j, vectors.get_vectors(first + j), GenerationGuard());
                    }
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
            for (uint32_t j = 0; j < count; ++j) {
                index.complete_add_document(first + j, std::move(prepared[j]));
                if (((first + j) % commit_interval) == 0) {
                    commit(index);
                }
            }
            if ((first / prepare_batch) % 100 == 0) {
                fprintf(stderr, "  built %u/%u in %.0fs\n", first + count - 1, n, seconds_since(start));
            }
        }
    }
    commit(index);
    return seconds_since(start);
}

double churn(NearestNeighborIndex& index, BenchVectors& vectors, uint32_t n, double fraction, uint32_t& updated) {
    std::vector<uint32_t> docids(n);
    for (uint32_t i = 0; i < n; ++i) {
        docids[i] = i + 1;
    }
    std::mt19937 rng(4711);
    std::shuffle(docids.begin(), docids.end(), rng);
    uint32_t pairs = uint32_t(fraction * n) / 2;
    auto     start = Clock::now();
    for (uint32_t p = 0; p < pairs; ++p) {
        uint32_t a = docids[2 * p];
        uint32_t b = docids[2 * p + 1];
        index.remove_document(a);
        index.remove_document(b);
        vectors.swap_vectors(a, b);
        index.add_document(a);
        index.add_document(b);
        if ((p % (commit_interval / 2)) == 0) {
            commit(index);
        }
    }
    commit(index);
    updated = 2 * pairs;
    return seconds_since(start);
}

// Replaces round(fraction * n) random docs with the next pool rows (remove, then add the new vector).
double replace_round(NearestNeighborIndex& index, BenchVectors& vectors, uint32_t n, double fraction,
                     uint32_t& next_row, uint32_t row_limit, std::mt19937& rng, uint32_t& replaced) {
    uint32_t count = static_cast<uint32_t>(fraction * n);
    if (uint64_t(next_row) + count > row_limit) {
        throw std::invalid_argument("churn pool exhausted");
    }
    std::vector<uint32_t> docids(n);
    for (uint32_t i = 0; i < n; ++i) {
        docids[i] = i + 1;
    }
    std::shuffle(docids.begin(), docids.end(), rng);
    auto start = Clock::now();
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t docid = docids[i];
        index.remove_document(docid);
        vectors.set_vector(docid, next_row++);
        index.add_document(docid);
        if ((i % commit_interval) == 0) {
            commit(index);
        }
    }
    commit(index);
    replaced = count;
    return seconds_since(start);
}

uint64_t proc_status_kb(const char* key) {
    std::ifstream in("/proc/self/status");
    std::string   line;
    while (std::getline(in, line)) {
        if (line.rfind(key, 0) == 0) {
            return std::stoull(line.substr(std::strlen(key)));
        }
    }
    return 0;
}

struct QueryStats {
    double   recall_10 = 0;
    double   recall_100 = 0;
    double   postings = 0;
    double   entries_scanned = 0;
    double   distances = 0;
    double   centroid_scan_us = 0;
    uint64_t mismatches = 0;
    double   p50_ms = 0;
    double   p99_ms = 0;
    uint32_t queries = 0;
};

double tie_aware_recall(std::vector<Neighbor> result, const GroundTruth& gt, uint32_t q, uint32_t k) {
    std::sort(result.begin(), result.end(),
              [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
    double   kth = gt.dists[size_t(q) * gt_k + k - 1];
    double   eps = 1e-9 * std::max(1.0, std::abs(kth));
    uint32_t hits = 0;
    for (uint32_t i = 0; i < std::min<size_t>(k, result.size()); ++i) {
        if (result[i].distance <= kth + eps) {
            ++hits;
        }
    }
    return double(hits) / k;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t idx = std::min(values.size() - 1, size_t(p * (values.size() - 1) + 0.5));
    return values[idx];
}

template <typename QueryFn>
QueryStats run_queries(const VectorFile& queries, uint32_t nq, const GroundTruth& gt, const BenchVectors& vectors,
                       const DistanceFunctionFactory& dff, QueryFn&& query_fn, const IvfIndex* drop_cache_of) {
    QueryStats          qs;
    std::vector<double> latencies;
    latencies.reserve(nq);
    for (uint32_t q = 0; q < nq; ++q) {
        TypedCells query(queries.vec(q), queries.cell_type, queries.dims);
        auto       df = dff.for_query_vector(query);
        if (drop_cache_of != nullptr) {
            drop_cache_of->drop_file_cache();
        }
        NearestNeighborIndex::Stats stats;
        IvfProbeStats               probe_stats;
        auto                        start = Clock::now();
        auto                        result = query_fn(stats, *df, query, probe_stats);
        latencies.push_back(seconds_since(start) * 1000.0);
        qs.recall_10 += tie_aware_recall(result, gt, q, 10);
        qs.recall_100 += tie_aware_recall(result, gt, q, gt_k);
        qs.postings += probe_stats.postings_probed;
        qs.entries_scanned += probe_stats.entries_scanned;
        qs.distances += stats.distances_computed();
        qs.centroid_scan_us += probe_stats.centroid_scan_ns / 1000.0;
        for (const auto& hit : result) {
            if (df->calc(vectors.cells(hit.docid)) != hit.distance) {
                ++qs.mismatches;
            }
        }
    }
    qs.queries = nq;
    qs.recall_10 /= nq;
    qs.recall_100 /= nq;
    qs.postings /= nq;
    qs.entries_scanned /= nq;
    qs.distances /= nq;
    qs.centroid_scan_us /= nq;
    qs.p50_ms = percentile(latencies, 0.50);
    qs.p99_ms = percentile(latencies, 0.99);
    return qs;
}

void print_ivf_stats(const char* label, const IvfIndex& index) {
    auto s = index.get_stats();
    auto mu = index.memory_usage();
    printf("%s: postings=%lu entries=%lu live=%lu splits=%lu compactions=%lu reassign_checks=%lu moves=%lu "
           "dropped_steps=%lu dropped_entries=%lu prepared_retired=%lu file_bytes=%lu bytes_written=%lu "
           "ram_allocated=%zu ram_used=%zu\n",
           label, s.postings, s.entries, s.live_entries, s.splits, s.compactions, s.reassign_checks, s.moves,
           s.dropped_steps, s.dropped_entries, s.prepared_retired, s.file_bytes, s.bytes_written, mu.allocatedBytes(),
           mu.usedBytes());
}

struct SweepSetup {
    const VectorFile&              queries;
    uint32_t                       nq;
    const BenchVectors&            vectors;
    const DistanceFunctionFactory& dff;
    uint32_t                       k;
};

// Runs every case: exact search once for cases below the approximate threshold, else the index at every param.
// make_query(param, case) returns the query function for the index path.
template <typename MakeQuery>
void sweep_cases(const char* label, const char* param_name, const std::vector<FilterCase>& cases,
                 const std::vector<uint32_t>& params, const SweepSetup& setup, double live, MakeQuery&& make_query) {
    if (!cases.empty() && !cases.front().exact) {
        run_queries(setup.queries, std::min(setup.nq, 1000u), cases.front().gt, setup.vectors, setup.dff,
                    make_query(params.back(), cases.front()), nullptr);
    }
    printf("%s:\n| filter | hit_ratio | path | %s | recall@10 | recall@100 | postings | entries | scanned_frac | "
           "distances | centroid_us | p50_ms | p99_ms | mismatches |\n",
           label, param_name);
    std::vector<std::string> interpolations;
    for (const auto& c : cases) {
        uint32_t nq = std::min(setup.nq, c.gt.nq);
        auto     row = [&](const char* path, const std::string& param, const QueryStats& qs) {
            printf("| %s | %.4f | %s | %s | %.4f | %.4f | %.1f | %.0f | %.4f | %.0f | %.1f | %.3f | %.3f | %lu |\n",
                   c.spec.c_str(), c.hit_ratio, path, param.c_str(), qs.recall_10, qs.recall_100, qs.postings,
                   qs.entries_scanned, qs.entries_scanned / live, qs.distances, qs.centroid_scan_us, qs.p50_ms,
                   qs.p99_ms, qs.mismatches);
            fflush(stdout);
        };
        if (c.exact) {
            auto exact = [&](NearestNeighborIndex::Stats& stats, const BoundDistanceFunction& df, TypedCells,
                             IvfProbeStats&) { return exact_top_k(setup.vectors, c.matching, setup.k, stats, df); };
            row("exact", "-", run_queries(setup.queries, nq, c.gt, setup.vectors, setup.dff, exact, nullptr));
            continue;
        }
        std::vector<std::array<double, 3>> curve; // param, recall@10, scanned fraction
        for (uint32_t param : params) {
            auto qs = run_queries(setup.queries, nq, c.gt, setup.vectors, setup.dff, make_query(param, c), nullptr);
            row("index", std::to_string(param), qs);
            curve.push_back({double(param), qs.recall_10, qs.entries_scanned / live});
        }
        for (double target : {0.80, 0.85, 0.90}) {
            std::string text = "| " + c.spec + " | " + std::to_string(target).substr(0, 4) + " | ";
            bool        found = false;
            for (size_t i = 1; i < curve.size() && !found; ++i) {
                if (curve[i - 1][1] < target && curve[i][1] >= target) {
                    double t = (target - curve[i - 1][1]) / (curve[i][1] - curve[i - 1][1]);
                    char   buf[96];
                    snprintf(buf, sizeof(buf), "%.1f | %.4f |", curve[i - 1][0] + t * (curve[i][0] - curve[i - 1][0]),
                             curve[i - 1][2] + t * (curve[i][2] - curve[i - 1][2]));
                    text += buf;
                    found = true;
                }
            }
            if (!found) {
                text += (!curve.empty() && curve.front()[1] >= target) ? "<= first | - |" : "not reached | - |";
            }
            interpolations.push_back(text);
        }
    }
    printf("%s at recall@10 targets:\n| filter | target | %s | scanned_frac |\n", label, param_name);
    for (const auto& line : interpolations) {
        printf("%s\n", line.c_str());
    }
}

int run(const Args& args) {
    auto     metric = parse_metric(args.required("metric"));
    uint32_t max_n = args.num("n", 1000000);
    uint32_t nq = args.num("nq", 10000);
    auto     base = load_vectors(args.required("base"), max_n);
    auto     queries = load_vectors(args.required("query"), nq);
    nq = queries.n;
    if (queries.dims != base.dims || queries.cell_type != base.cell_type) {
        throw std::invalid_argument("query and base vectors differ in type");
    }
    auto        dff = make_distance_function_factory(metric, base.cell_type);
    GroundTruth gt;
    std::string gt_path = args.required("gt");
    if (!load_gt(gt_path, nq, base.n, gt)) {
        gt = compute_gt(base, queries, *dff, args.num("gt-threads", 6));
        save_gt(gt, gt_path);
    }
    std::string index_kind = args.str("index", "ivf");
    if (index_kind == "none") {
        return 0;
    }
    std::string pool_path = args.str("pool", "");
    uint32_t    pool_first = 0;
    uint32_t    pool_limit = 0;
    if (!pool_path.empty()) {
        pool_first = append_pool(base, pool_path);
        pool_limit = static_cast<uint32_t>(base.data.size() / base.vector_bytes);
    }
    BenchVectors             vectors(base);
    uint32_t                 prepare_threads = args.num("prepare-threads", 6);
    double                   churn_fraction = args.real("churn", 0.1);
    uint32_t                 k = args.num("target-hits", gt_k);
    double                   approximate_threshold = args.real("approximate-threshold", 0.05);
    std::vector<std::string> filter_specs;
    {
        std::string list = args.str("filters", "none");
        size_t      pos = 0;
        while (pos < list.size()) {
            size_t comma = list.find(',', pos);
            comma = (comma == std::string::npos) ? list.size() : comma;
            filter_specs.push_back(list.substr(pos, comma - pos));
            pos = comma + 1;
        }
    }
    std::vector<uint16_t> labels;
    if (filter_specs.size() > 1 || filter_specs.front() != "none") {
        labels = load_labels(args.required("labels"), base.n);
    }
    auto make_cases = [&]() {
        std::vector<FilterCase> cases;
        for (const auto& spec : filter_specs) {
            cases.push_back(make_filter_case(spec, labels, vectors, base.n, nq, gt_path, args.str("filter-dir", ""),
                                             approximate_threshold, queries, *dff));
        }
        return cases;
    };
    SweepSetup setup{queries, nq, vectors, *dff, k};
    printf("dataset base=%s n=%u dims=%u nq=%u metric=%s index=%s prepare_threads=%u churn=%.2f\n",
           args.required("base").c_str(), base.n, base.dims, nq, args.required("metric").c_str(), index_kind.c_str(),
           prepare_threads, churn_fraction);
    if (index_kind == "ivf") {
        // Index knobs come from flags named like the env knobs: VESPA_IVF_REPLICA_EPS is --replica-eps.
        auto cfg = IvfIndexConfig::from_lookup(args.required("ivf-dir"), [&args](const char* knob) -> const char* {
            std::string flag(knob);
            std::transform(flag.begin(), flag.end(), flag.begin(),
                           [](char c) { return c == '_' ? '-' : static_cast<char>(std::tolower(c)); });
            const std::string* value = args.find(flag);
            return (value != nullptr) ? value->c_str() : nullptr;
        });
        std::string build_kind = args.str("build", "incremental");
        IvfIndex    index(vectors, make_distance_function_factory(metric, base.cell_type), base.cell_type, base.dims,
                          metric, cfg);
        double      build_s = 0.0;
        if (build_kind == "bulk") {
            std::vector<uint32_t> lids(base.n);
            for (uint32_t i = 0; i < base.n; ++i) {
                lids[i] = i + 1;
            }
            vespalib::ThreadStackExecutor executor(std::max(1u, prepare_threads));
            auto                          start = Clock::now();
            if (!index.bulk_build(lids, &executor)) {
                throw std::runtime_error("bulk build refused");
            }
            commit(index);
            build_s = seconds_since(start);
        } else if (build_kind == "incremental") {
            build_s = build(index, vectors, base.n, prepare_threads);
        } else {
            throw std::invalid_argument("unknown --build " + build_kind);
        }
        auto s = index.get_stats();
        printf("build: kind=%s capacity=%u seconds=%.1f inserts_per_s=%.0f moves_per_insert=%.3f "
               "bytes_written_per_insert=%.0f write_amplification=%.2f split_fallbacks=%lu "
               "file_bytes_per_live_entry=%.0f rss_kb=%lu hwm_kb=%lu\n",
               build_kind.c_str(), cfg.posting_capacity, build_s, base.n / build_s, double(s.moves) / base.n,
               double(s.bytes_written) / base.n, double(s.bytes_written) / (double(base.n) * index.entry_bytes()),
               s.split_fallbacks, double(s.file_bytes) / s.live_entries, proc_status_kb("VmRSS:"),
               proc_status_kb("VmHWM:"));
        print_ivf_stats("after_build", index);
        {
            vespalib::Slime                slime;
            vespalib::slime::SlimeInserter inserter(slime);
            index.make_state_explorer()->get_state(inserter, true);
            printf("cfg: %s\n", slime.get()["cfg"].toString().c_str());
        }
        auto ivf_query = [&index, k](uint32_t nprobe, const FilterCase& c) {
            const GlobalFilter* filter = c.filter.get();
            return [&index, k, nprobe, filter](NearestNeighborIndex::Stats& stats, const BoundDistanceFunction& df,
                                               TypedCells query, IvfProbeStats& probe_stats) {
                return index.find_top_k_with_query(stats, k, df, query, filter, nprobe, Deadline::never(),
                                                   max_distance, probe_stats);
            };
        };
        auto nprobes = args.list("nprobes", "8,16,32,64,128");
        auto sweep = [&](const char* phase) {
            sweep_cases((build_kind + " " + phase).c_str(), "nprobe", make_cases(), nprobes, setup,
                        index.get_stats().live_entries, ivf_query);
        };
        if (args.num("sweep-before-churn", 1) != 0) {
            sweep("before_churn");
        }
        if (churn_fraction > 0) {
            uint32_t updated = 0;
            auto     before = index.get_stats();
            double   churn_s = churn(index, vectors, base.n, churn_fraction, updated);
            auto     after = index.get_stats();
            printf("churn: updated=%u seconds=%.1f updates_per_s=%.0f moves_per_update=%.3f "
                   "bytes_written_per_update=%.0f\n",
                   updated, churn_s, updated / churn_s, double(after.moves - before.moves) / updated,
                   double(after.bytes_written - before.bytes_written) / updated);
            print_ivf_stats("after_churn", index);
            sweep("after_churn");
        }
        uint32_t rounds = args.num("churn-rounds", 0);
        if (rounds > 0) {
            if (pool_path.empty()) {
                throw std::invalid_argument("--churn-rounds needs --pool");
            }
            double       round_fraction = args.real("round-fraction", 0.1);
            uint32_t     next_row = pool_first;
            std::mt19937 rng(8191);
            auto         round_nprobes = args.list("round-nprobes", "16,32,64,128");
            for (uint32_t r = 1; r <= rounds; ++r) {
                uint32_t replaced = 0;
                auto     before = index.get_stats();
                double   round_s =
                    replace_round(index, vectors, base.n, round_fraction, next_row, pool_limit, rng, replaced);
                auto after = index.get_stats();
                printf("round %u: replaced=%u turnover=%.2f seconds=%.1f updates_per_s=%.0f moves_per_update=%.3f "
                       "bytes_written_per_update=%.0f postings=%lu entries=%lu live=%lu stale_frac=%.3f "
                       "file_bytes=%lu split_fallbacks=%lu gc_compactions=%lu merges=%lu\n",
                       r, replaced, double(next_row - pool_first) / base.n, round_s, replaced / round_s,
                       double(after.moves - before.moves) / replaced,
                       double(after.bytes_written - before.bytes_written) / replaced, after.postings, after.entries,
                       after.live_entries,
                       after.entries > 0 ? double(after.entries - after.live_entries) / after.entries : 0.0,
                       after.file_bytes, after.split_fallbacks, after.gc_compactions, after.merges);
                FilterCase none_case("none");
                none_case.gt = compute_gt_live(vectors, base.n, queries, *dff, args.num("gt-threads", 6));
                std::vector<FilterCase> cases;
                cases.push_back(std::move(none_case));
                char label[64];
                snprintf(label, sizeof(label), "round %u", r);
                sweep_cases(label, "nprobe", cases, round_nprobes, setup, after.live_entries, ivf_query);
            }
        }
    } else if (index_kind == "hnsw") {
        uint32_t                         m = args.num("hnsw-m", 16);
        uint32_t                         ef = args.num("hnsw-ef-construction", 200);
        HnswIndex<HnswIndexType::SINGLE> index(vectors, make_distance_function_factory(metric, base.cell_type),
                                               std::make_unique<InvLogLevelGenerator>(m),
                                               HnswIndexConfig(2 * m, m, ef, 10000, true));
        double                           build_s = build(index, vectors, base.n, prepare_threads);
        auto                             mu = index.memory_usage();
        printf("build: m=%u ef_construction=%u seconds=%.1f inserts_per_s=%.0f graph_allocated=%zu graph_used=%zu "
               "vectors_bytes=%zu rss_kb=%lu hwm_kb=%lu\n",
               m, ef, build_s, base.n / build_s, mu.allocatedBytes(), mu.usedBytes(), base.data.size(),
               proc_status_kb("VmRSS:"), proc_status_kb("VmHWM:"));
        if (churn_fraction > 0) {
            uint32_t updated = 0;
            double   churn_s = churn(index, vectors, base.n, churn_fraction, updated);
            printf("churn: updated=%u seconds=%.1f updates_per_s=%.0f\n", updated, churn_s, updated / churn_s);
        }
        auto hnsw_query = [&index, k](uint32_t explore_additional, const FilterCase& c) {
            const GlobalFilter* filter = c.filter.get();
            bool                low_hit_ratio = c.hit_ratio < filter_first_upper_limit;
            return [&index, k, explore_additional, filter, low_hit_ratio](NearestNeighborIndex::Stats& stats,
                                                                          const BoundDistanceFunction& df, TypedCells,
                                                                          IvfProbeStats&) {
                if (filter != nullptr) {
                    return index.find_top_k_with_filter(stats, k, df, *filter, low_hit_ratio,
                                                        filter_first_exploration, k + explore_additional, 0.0, false,
                                                        Deadline::never(), max_distance);
                }
                return index.find_top_k(stats, k, df, k + explore_additional, 0.0, false, Deadline::never(),
                                        max_distance);
            };
        };
        sweep_cases("hnsw", "explore_additional", make_cases(), args.list("explore", "0,100,400"), setup,
                    double(base.n), hnsw_query);
    } else {
        throw std::invalid_argument("unknown --index " + index_kind);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(Args(argc, argv));
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
