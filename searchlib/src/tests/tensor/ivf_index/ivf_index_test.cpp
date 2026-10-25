// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/eval/eval/value_type.h>
#include <vespa/fastos/file.h>
#include <vespa/searchcommon/attribute/config.h>
#include <vespa/searchlib/queryeval/global_filter.h>
#include <vespa/searchlib/tensor/default_nearest_neighbor_index_factory.h>
#include <vespa/searchlib/tensor/dense_tensor_attribute.h>
#include <vespa/searchlib/tensor/distance_function_factory.h>
#include <vespa/searchlib/tensor/doc_vector_access.h>
#include <vespa/searchlib/tensor/empty_subspace.h>
#include <vespa/searchlib/tensor/ivf_index.h>
#include <vespa/searchlib/tensor/ivf_kmeans.h>
#include <vespa/searchlib/tensor/nearest_neighbor_index_loader.h>
#include <vespa/searchlib/tensor/nearest_neighbor_index_saver.h>
#include <vespa/searchlib/tensor/subspace_type.h>
#include <vespa/searchlib/tensor/vector_bundle.h>
#include <vespa/searchlib/test/directory_handler.h>
#include <vespa/searchlib/test/vector_buffer_writer.h>
#include <vespa/vespalib/data/fileheader.h>
#include <vespa/vespalib/data/slime/slime.h>
#include <vespa/vespalib/datastore/compaction_strategy.h>
#include <vespa/vespalib/gtest/gtest.h>
#include <vespa/vespalib/net/http/state_explorer.h>
#include <vespa/vespalib/util/deadline.h>
#include <vespa/vespalib/util/threadstackexecutor.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <thread>
#include <vector>

#include <vespa/log/log.h>
LOG_SETUP("ivf_index_test");

using namespace search::tensor;
using search::attribute::DistanceMetric;
using search::attribute::HnswIndexParams;
using search::queryeval::GlobalFilter;
using vespalib::Deadline;
using vespalib::GenerationGuard;
using vespalib::Slime;
using vespalib::eval::CellType;
using vespalib::eval::FastValueBuilderFactory;
using vespalib::eval::TensorSpec;
using vespalib::eval::TypedCells;
using vespalib::eval::ValueType;

using Neighbor = NearestNeighborIndex::Neighbor;
using Bytes = std::vector<char>;

namespace {

constexpr double max_distance = std::numeric_limits<double>::max();

std::string ivf_test_dir() {
    const char* env_dir = std::getenv("VESPA_IVF_TEST_DIR");
    std::string dir = (env_dir != nullptr && *env_dir != '\0') ? env_dir : "ivf_index_test_dir";
    std::filesystem::create_directories(dir);
    return dir;
}

class TestVectors : public DocVectorAccess {
    CellType           _cell_type;
    uint32_t           _dims;
    SubspaceType       _subspace_type;
    EmptySubspace      _empty;
    std::vector<Bytes> _docs;

public:
    TestVectors(CellType cell_type, uint32_t dims)
        : _cell_type(cell_type),
          _dims(dims),
          _subspace_type(ValueType::make_type(cell_type, {{"x", dims}})),
          _empty(_subspace_type),
          _docs() {}
    ~TestVectors() override;
    size_t vector_bytes() const noexcept { return _subspace_type.mem_size(); }
    CellType cell_type() const noexcept { return _cell_type; }
    uint32_t dims() const noexcept { return _dims; }
    void set(uint32_t docid, const Bytes& bytes) {
        if (docid >= _docs.size()) {
            _docs.resize(docid + 1);
        }
        _docs[docid] = bytes;
    }
    void clear(uint32_t docid) {
        if (docid < _docs.size()) {
            _docs[docid].clear();
        }
    }
    uint32_t docid_limit() const noexcept { return _docs.size(); }
    const Bytes& raw(uint32_t docid) const { return _docs[docid]; }
    bool has(uint32_t docid) const noexcept { return docid < _docs.size() && !_docs[docid].empty(); }
    VectorBundle bundle_of(const Bytes& bytes) const noexcept {
        return {bytes.data(), static_cast<uint32_t>(bytes.size() / vector_bytes()), _subspace_type};
    }
    TypedCells cells_of(const char* data) const noexcept { return {data, _cell_type, _dims}; }
    TypedCells get_vector(uint32_t docid, uint32_t subspace) const noexcept override {
        auto bundle = get_vectors(docid);
        return (subspace < bundle.subspaces()) ? bundle.cells(subspace) : _empty.cells();
    }
    VectorBundle get_vectors(uint32_t docid) const noexcept override {
        if (docid >= _docs.size()) {
            return {};
        }
        return bundle_of(_docs[docid]);
    }
};

TestVectors::~TestVectors() = default;

// Synthetic clustered data: float vectors around gaussian centers, or int8 bit vectors with random bit flips.
class Generator {
    CellType           _cell_type;
    uint32_t           _dims;
    double             _noise;
    std::vector<Bytes> _centers;
    std::mt19937       _rng;

public:
    Generator(CellType cell_type, uint32_t dims, uint32_t clusters, double noise, uint32_t seed)
        : _cell_type(cell_type), _dims(dims), _noise(noise), _centers(), _rng(seed) {
        for (uint32_t c = 0; c < clusters; ++c) {
            _centers.push_back(random_center());
        }
    }
    Bytes random_center() {
        if (_cell_type == CellType::FLOAT) {
            std::normal_distribution<float> dist(0.0, 1.0);
            std::vector<float>              v(_dims);
            for (auto& x : v) {
                x = dist(_rng);
            }
            return Bytes(reinterpret_cast<const char*>(v.data()), reinterpret_cast<const char*>(v.data() + _dims));
        }
        std::uniform_int_distribution<int> dist(0, 255);
        Bytes                              v(_dims);
        for (auto& x : v) {
            x = static_cast<char>(dist(_rng));
        }
        return v;
    }
    Bytes sample() {
        const Bytes& center = _centers[std::uniform_int_distribution<size_t>(0, _centers.size() - 1)(_rng)];
        Bytes        v = center;
        if (_cell_type == CellType::FLOAT) {
            std::normal_distribution<float> dist(0.0, _noise);
            for (uint32_t i = 0; i < _dims; ++i) {
                float x;
                std::memcpy(&x, v.data() + i * sizeof(float), sizeof(float));
                x += dist(_rng);
                std::memcpy(v.data() + i * sizeof(float), &x, sizeof(float));
            }
        } else {
            std::bernoulli_distribution flip(_noise);
            for (uint32_t i = 0; i < _dims; ++i) {
                for (uint32_t bit = 0; bit < 8; ++bit) {
                    if (flip(_rng)) {
                        v[i] = static_cast<char>(v[i] ^ (1 << bit));
                    }
                }
            }
        }
        return v;
    }
    Bytes multi_sample(uint32_t subspaces) {
        Bytes result;
        for (uint32_t s = 0; s < subspaces; ++s) {
            Bytes v = sample();
            result.insert(result.end(), v.begin(), v.end());
        }
        return result;
    }
    uint32_t uniform(uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(_rng); }
};

struct RecallRow {
    uint32_t nprobe;
    double   recall;
    double   distances;
    double   postings;
};

class IvfIndexTest : public ::testing::Test {
protected:
    std::unique_ptr<TestVectors> vectors;
    std::unique_ptr<IvfIndex>    index;

    IvfIndexTest() : vectors(), index() {}
    ~IvfIndexTest() override;

    void make_index(CellType cell_type, uint32_t dims, DistanceMetric metric, const IvfIndexConfig& cfg) {
        make_named_index(cell_type, dims, metric, cfg, "");
    }
    void make_named_index(CellType cell_type, uint32_t dims, DistanceMetric metric, const IvfIndexConfig& cfg,
                          const std::string& posting_path) {
        index.reset();
        vectors = std::make_unique<TestVectors>(cell_type, dims);
        reopen_index(metric, cfg, posting_path);
    }
    // A new index over the current vectors, as after a restart.
    void reopen_index(DistanceMetric metric, const IvfIndexConfig& cfg, const std::string& posting_path) {
        index.reset();
        index = std::make_unique<IvfIndex>(*vectors, make_distance_function_factory(metric, vectors->cell_type()),
                                           vectors->cell_type(), vectors->dims(), metric, cfg, posting_path);
    }
    IvfIndexConfig config(uint32_t posting_capacity) const {
        IvfIndexConfig cfg;
        cfg.posting_capacity = posting_capacity;
        return cfg;
    }
    void commit() {
        index->inc_generation();
        index->reclaim_unused_memory();
    }
    void add(uint32_t docid, const Bytes& bytes) {
        vectors->set(docid, bytes);
        index->add_document(docid);
        if ((docid % 256) == 0) {
            commit();
        }
    }
    void remove(uint32_t docid) {
        index->remove_document(docid);
        vectors->clear(docid);
    }
    std::vector<Neighbor> query(const char* q, uint32_t k, uint32_t nprobe, const GlobalFilter* filter,
                                NearestNeighborIndex::Stats& stats, IvfProbeStats& probe_stats) const {
        auto df = index->distance_function_factory().for_query_vector(vectors->cells_of(q));
        return index->find_top_k_with_nprobe(stats, k, *df, filter, nprobe, Deadline::never(), max_distance,
                                             probe_stats);
    }
    std::vector<Neighbor> query(const Bytes& q, uint32_t k) const {
        NearestNeighborIndex::Stats stats;
        IvfProbeStats               probe_stats;
        return query(q.data(), k, index->config().nprobe, nullptr, stats, probe_stats);
    }
    // Exact top k over live docs, sorted by distance; multi-vector docs use their minimum distance.
    std::vector<Neighbor> exact(const char* q, uint32_t k, const GlobalFilter* filter) const {
        auto                  df = index->distance_function_factory().for_query_vector(vectors->cells_of(q));
        std::vector<Neighbor> all;
        for (uint32_t docid = 0; docid < vectors->docid_limit(); ++docid) {
            if (!vectors->has(docid) || (filter != nullptr && !filter->check(docid))) {
                continue;
            }
            auto   bundle = vectors->get_vectors(docid);
            double best = max_distance;
            for (uint32_t s = 0; s < bundle.subspaces(); ++s) {
                best = std::min(best, df->calc(bundle.cells(s)));
            }
            all.emplace_back(docid, best);
        }
        auto   by_distance = [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; };
        size_t keep = std::min<size_t>(k, all.size());
        std::partial_sort(all.begin(), all.begin() + keep, all.end(), by_distance);
        all.resize(keep);
        return all;
    }
    // Tie-aware recall: a hit counts when it is no farther than the k-th exact neighbor.
    static double recall(const std::vector<Neighbor>& got, const std::vector<Neighbor>& want) {
        if (want.empty()) {
            return 1.0;
        }
        double   kth = want.back().distance;
        uint32_t hits = 0;
        for (const auto& n : got) {
            if (n.distance <= kth + 1e-9 * std::max(1.0, std::abs(kth))) {
                ++hits;
            }
        }
        return std::min(1.0, double(hits) / want.size());
    }
    std::vector<RecallRow> recall_table(const std::vector<Bytes>& queries, uint32_t k,
                                        const std::vector<uint32_t>& nprobes, const GlobalFilter* filter) const {
        std::vector<std::vector<Neighbor>> truth;
        for (const auto& q : queries) {
            truth.push_back(exact(q.data(), k, filter));
        }
        std::vector<RecallRow> rows;
        for (uint32_t nprobe : nprobes) {
            RecallRow row{nprobe, 0.0, 0.0, 0.0};
            for (size_t i = 0; i < queries.size(); ++i) {
                NearestNeighborIndex::Stats stats;
                IvfProbeStats               probe_stats;
                auto                        got = query(queries[i].data(), k, nprobe, filter, stats, probe_stats);
                row.recall += recall(got, truth[i]);
                row.distances += stats.distances_computed();
                row.postings += probe_stats.postings_probed;
            }
            row.recall /= queries.size();
            row.distances /= queries.size();
            row.postings /= queries.size();
            rows.push_back(row);
        }
        return rows;
    }
    static void print_table(const char* label, const std::vector<RecallRow>& rows) {
        fprintf(stderr, "%s\n%8s %10s %14s %16s\n", label, "nprobe", "recall@10", "mean_distances", "mean_postings");
        for (const auto& row : rows) {
            fprintf(stderr, "%8u %10.4f %14.1f %16.1f\n", row.nprobe, row.recall, row.distances, row.postings);
        }
    }
    Slime state() const {
        Slime                          slime;
        vespalib::slime::SlimeInserter inserter(slime);
        index->make_state_explorer()->get_state(inserter, true);
        return slime;
    }
    static bool contains(const std::vector<Neighbor>& result, uint32_t docid) {
        return std::any_of(result.begin(), result.end(), [docid](const Neighbor& n) { return n.docid == docid; });
    }
    static void expect_sorted_unique(const std::vector<Neighbor>& result) {
        for (size_t i = 1; i < result.size(); ++i) {
            EXPECT_LT(result[i - 1].docid, result[i].docid);
        }
    }

    void churn_recall_test(const char* label, CellType cell_type, uint32_t dims, DistanceMetric metric, double noise);
};

IvfIndexTest::~IvfIndexTest() = default;

void IvfIndexTest::churn_recall_test(const char* label, CellType cell_type, uint32_t dims, DistanceMetric metric,
                                     double noise) {
    constexpr uint32_t num_docs = 20000;
    constexpr uint32_t k = 10;
    make_index(cell_type, dims, metric, IvfIndexConfig());
    Generator gen(cell_type, dims, 200, noise, 42);
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    std::vector<std::pair<uint32_t, Bytes>> updated;
    std::vector<std::pair<uint32_t, Bytes>> removed;
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        if ((docid % 5) == 0) {
            Bytes old_vector = vectors->raw(docid);
            remove(docid);
            add(docid, gen.sample());
            updated.emplace_back(docid, std::move(old_vector));
        }
    }
    commit();
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        if ((docid % 10) == 1) {
            removed.emplace_back(docid, vectors->raw(docid));
            remove(docid);
        }
    }
    commit();

    for (size_t i = 0; i < removed.size(); i += 10) {
        auto result = query(removed[i].second, k);
        EXPECT_FALSE(contains(result, removed[i].first)) << "removed docid " << removed[i].first;
    }
    uint32_t found_by_new = 0;
    uint32_t checked = 0;
    for (size_t i = 0; i < updated.size(); i += 10) {
        uint32_t docid = updated[i].first;
        auto     df_old =
            index->distance_function_factory().for_query_vector(vectors->cells_of(updated[i].second.data()));
        double old_to_new = df_old->calc(vectors->cells_of(vectors->raw(docid).data()));
        for (const auto& n : query(updated[i].second, k)) {
            if (n.docid == docid) {
                EXPECT_DOUBLE_EQ(old_to_new, n.distance) << "updated docid " << docid << " found by old vector";
            }
        }
        auto by_new = query(vectors->raw(docid), k);
        for (const auto& n : by_new) {
            if (n.docid == docid) {
                EXPECT_EQ(0.0, n.distance);
                ++found_by_new;
            }
        }
        ++checked;
    }
    EXPECT_EQ(checked, found_by_new);

    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 200; ++i) {
        queries.push_back(gen.sample());
    }
    auto rows = recall_table(queries, k, {1, 2, 4, 8, 16, 32, 64}, nullptr);
    print_table(label, rows);
    auto default_row = std::find_if(rows.begin(), rows.end(),
                                    [this](const RecallRow& row) { return row.nprobe == index->config().nprobe; });
    ASSERT_NE(default_row, rows.end());
    EXPECT_GE(default_row->recall, 0.95);

    auto  slime = state();
    auto& root = slime.get();
    fprintf(stderr,
            "%s state: postings=%ld entries=%ld live=%ld stale=%ld splits=%ld compactions=%ld "
            "reassign_checks=%ld moves=%ld dropped_steps=%ld dropped_entries=%ld file_bytes=%ld\n",
            label, root["postings"].asLong(), root["entries"].asLong(), root["live_entries"].asLong(),
            root["stale_entries"].asLong(), root["splits"].asLong(), root["compactions"].asLong(),
            root["reassign_checks"].asLong(), root["moves"].asLong(), root["dropped_steps"].asLong(),
            root["dropped_entries"].asLong(), root["file_bytes"].asLong());
    EXPECT_GT(root["splits"].asLong(), 0);
    EXPECT_GT(root["reassign_checks"].asLong(), 0);
    EXPECT_GT(root["moves"].asLong(), 0);
    EXPECT_EQ(0, root["dropped_entries"].asLong());
    EXPECT_EQ(int64_t(num_docs - removed.size()), root["live_entries"].asLong());
}

} // namespace

TEST_F(IvfIndexTest, recall_float_euclidean_after_insert_update_remove) {
    churn_recall_test("float x[64] euclidean", CellType::FLOAT, 64, DistanceMetric::Euclidean, 0.6);
}

TEST_F(IvfIndexTest, recall_int8_hamming_after_insert_update_remove) {
    churn_recall_test("int8 x[32] hamming", CellType::INT8, 32, DistanceMetric::Hamming, 0.12);
}

TEST_F(IvfIndexTest, sweep_compacts_stale_postings_and_merges_small_ones) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t num_docs = 4000;
    auto               cfg = config(64);
    cfg.gc_steps = 65536;
    cfg.merge_fraction = 0.25;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg);
    Generator gen(CellType::FLOAT, dims, 40, 0.5, 11);
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    uint32_t kept = 0;
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        if ((docid % 10) < 7) {
            remove(docid);
        } else {
            ++kept;
        }
    }
    commit();
    ASSERT_GT(state().get()["stale_entries"].asLong(), 0);
    for (uint32_t round = 0; round < 3; ++round) {
        index->inc_generation();
        index->consider_compact(vespalib::datastore::CompactionStrategy());
        commit();
    }
    auto  slime = state();
    auto& root = slime.get();
    EXPECT_EQ(0, root["stale_entries"].asLong());
    EXPECT_EQ(int64_t(kept), root["live_entries"].asLong());
    EXPECT_GT(root["gc_compactions"].asLong(), 0);
    EXPECT_GT(root["merges"].asLong(), 0);
    EXPECT_EQ(0, root["dropped_entries"].asLong());
    uint32_t postings = root["postings"].asLong();
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        if (!vectors->has(docid)) {
            continue;
        }
        NearestNeighborIndex::Stats stats;
        IvfProbeStats               probe_stats;
        auto result = query(vectors->raw(docid).data(), 1, postings, nullptr, stats, probe_stats);
        ASSERT_EQ(1u, result.size()) << "docid " << docid;
        EXPECT_EQ(0.0, result[0].distance) << "docid " << docid;
    }
}

TEST_F(IvfIndexTest, multi_vector_doc_is_returned_once_with_min_distance) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t k = 10;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(64));
    Generator gen(CellType::FLOAT, dims, 50, 0.5, 7);
    for (uint32_t docid = 1; docid <= 3000; ++docid) {
        add(docid, gen.multi_sample(gen.uniform(2, 4)));
    }
    commit();
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 100; ++i) {
        queries.push_back(gen.sample());
    }
    for (const auto& q : queries) {
        auto result = query(q, k);
        EXPECT_EQ(k, result.size());
        expect_sorted_unique(result);
        auto df = index->distance_function_factory().for_query_vector(vectors->cells_of(q.data()));
        for (const auto& n : result) {
            auto   bundle = vectors->get_vectors(n.docid);
            double best = max_distance;
            for (uint32_t s = 0; s < bundle.subspaces(); ++s) {
                best = std::min(best, df->calc(bundle.cells(s)));
            }
            EXPECT_DOUBLE_EQ(best, n.distance);
        }
    }
    auto rows = recall_table(queries, k, {index->config().nprobe}, nullptr);
    print_table("multi-vector float x[16], 2-4 subspaces", rows);
    EXPECT_GE(rows[0].recall, 0.95);
    EXPECT_GT(state().get()["splits"].asLong(), 0);
}

TEST_F(IvfIndexTest, adaptive_filter_probing_skips_postings_without_matches) {
    constexpr uint32_t dims = 32;
    constexpr uint32_t num_docs = 8000;
    constexpr uint32_t k = 10;
    auto               cfg = config(64);
    cfg.filter_probe_mode = IvfFilterProbeMode::ADAPTIVE;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg);
    Generator gen(CellType::FLOAT, dims, 100, 0.6, 13);
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    // A filter correlated with the vectors: the docs nearest one query point.
    Bytes                 center = gen.sample();
    auto                  near = exact(center.data(), num_docs / 10, nullptr);
    std::vector<uint32_t> allowed;
    for (const auto& n : near) {
        allowed.push_back(n.docid);
    }
    std::sort(allowed.begin(), allowed.end());
    auto               filter = GlobalFilter::create(allowed, num_docs + 1);
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 50; ++i) {
        queries.push_back(gen.sample());
    }
    uint32_t nprobe = index->config().nprobe;
    double   examined = 0;
    double   probed = 0;
    for (const auto& q : queries) {
        NearestNeighborIndex::Stats stats;
        IvfProbeStats               probe_stats;
        auto                        result = query(q.data(), k, nprobe, filter.get(), stats, probe_stats);
        EXPECT_EQ(k, result.size());
        for (const auto& n : result) {
            EXPECT_TRUE(filter->check(n.docid)) << "docid " << n.docid;
        }
        examined += probe_stats.postings_examined;
        probed += probe_stats.postings_probed;
    }
    EXPECT_GT(examined, probed);
    auto rows = recall_table(queries, k, {nprobe}, filter.get());
    print_table("adaptive filtered (10% of docs, correlated) float x[32]", rows);
    EXPECT_GE(rows[0].recall, 0.95);
}

TEST_F(IvfIndexTest, selective_filter_below_exact_ratio_is_searched_exactly) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t num_docs = 4000;
    constexpr uint32_t k = 10;
    auto               cfg = config(64);
    cfg.filter_exact_ratio = 0.05;
    cfg.nprobe = 1;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg);
    Generator gen(CellType::FLOAT, dims, 40, 0.5, 17);
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    std::vector<uint32_t> allowed;
    for (uint32_t docid = 3; docid <= num_docs; docid += 40) {
        allowed.push_back(docid);
    }
    auto filter = GlobalFilter::create(allowed, num_docs + 1);
    for (uint32_t i = 0; i < 20; ++i) {
        Bytes                       q = gen.sample();
        NearestNeighborIndex::Stats stats;
        IvfProbeStats               probe_stats;
        auto                        got = query(q.data(), k, 1, filter.get(), stats, probe_stats);
        EXPECT_EQ(0u, probe_stats.postings_probed);
        EXPECT_EQ(1.0, recall(got, exact(q.data(), k, filter.get())));
    }
}

TEST_F(IvfIndexTest, large_k_reads_enough_postings_for_k_hits) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t num_docs = 6000;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(64));
    Generator gen(CellType::FLOAT, dims, 60, 0.5, 19);
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    Bytes q = gen.sample();
    for (uint32_t k : {10u, 1000u, 3000u}) {
        auto df = index->distance_function_factory().for_query_vector(vectors->cells_of(q.data()));
        NearestNeighborIndex::Stats stats;
        auto result = index->find_top_k(stats, k, *df, k, 0.0, false, Deadline::never(), max_distance);
        EXPECT_EQ(k, result.size()) << "k " << k;
    }
}

TEST_F(IvfIndexTest, filter_is_honored) {
    constexpr uint32_t dims = 32;
    constexpr uint32_t num_docs = 5000;
    constexpr uint32_t k = 10;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, IvfIndexConfig());
    Generator gen(CellType::FLOAT, dims, 100, 0.6, 11);
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    std::vector<uint32_t> allowed;
    for (uint32_t docid = 1; docid <= num_docs; docid += 50) {
        allowed.push_back(docid);
    }
    auto               filter = GlobalFilter::create(allowed, num_docs + 1);
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 50; ++i) {
        queries.push_back(gen.sample());
    }
    for (const auto& q : queries) {
        auto df = index->distance_function_factory().for_query_vector(vectors->cells_of(q.data()));
        NearestNeighborIndex::Stats stats;
        auto result = index->find_top_k_with_filter(stats, k, *df, *filter, true, 0.01, k, 0.0, false,
                                                    Deadline::never(), max_distance);
        EXPECT_EQ(k, result.size());
        for (const auto& n : result) {
            EXPECT_TRUE(filter->check(n.docid)) << "docid " << n.docid;
        }
    }
    auto rows = recall_table(queries, k, {index->config().nprobe}, filter.get());
    print_table("filtered (2% of docs) float x[32]", rows);
    EXPECT_GE(rows[0].recall, 0.95);
}

TEST_F(IvfIndexTest, explore_k_scales_postings_probed) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t k = 10;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(32));
    Generator gen(CellType::FLOAT, dims, 50, 0.5, 17);
    for (uint32_t docid = 1; docid <= 10000; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    uint32_t nprobe = index->config().nprobe;
    EXPECT_EQ(nprobe, index->nprobe_for_explore(k, k));
    EXPECT_EQ(nprobe, index->nprobe_for_explore(k, 1));
    EXPECT_EQ((3 * nprobe + 1) / 2, index->nprobe_for_explore(k, 15));
    EXPECT_EQ(2 * nprobe, index->nprobe_for_explore(k, 2 * k));
    EXPECT_EQ(5 * nprobe, index->nprobe_for_explore(0, 5));
    EXPECT_EQ(index->config().max_postings, index->nprobe_for_explore(1, 1u << 30));
    ASSERT_GT(state().get()["postings"].asLong(), int64_t(8 * nprobe));

    Bytes                 q = gen.sample();
    auto                  df = index->distance_function_factory().for_query_vector(vectors->cells_of(q.data()));
    std::vector<uint32_t> half;
    for (uint32_t docid = 2; docid <= 10000; docid += 2) {
        half.push_back(docid);
    }
    auto     filter = GlobalFilter::create(half, 10001);
    uint64_t prev_plain = 0;
    uint64_t prev_filtered = 0;
    for (uint32_t explore_k : {k, 2 * k, 4 * k}) {
        SCOPED_TRACE(explore_k);
        NearestNeighborIndex::Stats plain;
        index->find_top_k(plain, k, *df, explore_k, 0.0, false, Deadline::never(), max_distance);
        NearestNeighborIndex::Stats filtered;
        index->find_top_k_with_filter(filtered, k, *df, *filter, false, 0.0, explore_k, 0.0, false, Deadline::never(),
                                      max_distance);
        EXPECT_GT(plain.nodes_visited(), prev_plain);
        EXPECT_GT(filtered.nodes_visited(), prev_filtered);
        EXPECT_GE(filtered.nodes_visited(), plain.nodes_visited());
        prev_plain = plain.nodes_visited();
        prev_filtered = filtered.nodes_visited();
    }
}

TEST_F(IvfIndexTest, bulk_build_recall_and_versions) {
    constexpr uint32_t dims = 32;
    constexpr uint32_t num_docs = 10000;
    constexpr uint32_t k = 10;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(64));
    Generator             gen(CellType::FLOAT, dims, 100, 0.6, 21);
    std::vector<uint32_t> lids;
    for (uint32_t docid = 1; docid <= num_docs; ++docid) {
        vectors->set(docid, gen.sample());
        lids.push_back(docid);
    }
    vespalib::ThreadStackExecutor executor(2);
    ASSERT_TRUE(index->bulk_build(lids, &executor));
    commit();
    auto stats = index->get_stats();
    EXPECT_EQ(1u, stats.bulk_builds);
    EXPECT_EQ(0u, stats.moves);
    EXPECT_EQ(0u, stats.dropped_entries);
    EXPECT_EQ(num_docs, stats.live_entries);
    EXPECT_EQ(num_docs, stats.entries);
    EXPECT_GE(stats.postings, num_docs / 64);
    EXPECT_FALSE(index->bulk_build(lids, &executor));

    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 100; ++i) {
        queries.push_back(gen.sample());
    }
    auto rows = recall_table(queries, k, {index->config().nprobe}, nullptr);
    print_table("bulk built float x[32]", rows);
    EXPECT_GE(rows[0].recall, 0.95);

    std::vector<std::pair<uint32_t, Bytes>> removed;
    std::vector<std::pair<uint32_t, Bytes>> updated;
    for (uint32_t docid = 1; docid <= num_docs; docid += 50) {
        removed.emplace_back(docid, vectors->raw(docid));
        remove(docid);
        uint32_t update = docid + 1;
        updated.emplace_back(update, vectors->raw(update));
        remove(update);
        add(update, gen.sample());
        add(num_docs + docid, gen.sample());
    }
    commit();
    for (const auto& r : removed) {
        EXPECT_FALSE(contains(query(r.second, k), r.first)) << "removed docid " << r.first;
    }
    for (const auto& u : updated) {
        auto   df_old = index->distance_function_factory().for_query_vector(vectors->cells_of(u.second.data()));
        double old_to_new = df_old->calc(vectors->cells_of(vectors->raw(u.first).data()));
        for (const auto& n : query(u.second, k)) {
            if (n.docid == u.first) {
                EXPECT_DOUBLE_EQ(old_to_new, n.distance);
            }
        }
        EXPECT_TRUE(contains(query(vectors->raw(u.first), 1), u.first)) << "updated docid " << u.first;
    }
    for (uint32_t docid = 1; docid <= num_docs; docid += 50) {
        EXPECT_TRUE(contains(query(vectors->raw(num_docs + docid), 1), num_docs + docid));
    }
}

TEST_F(IvfIndexTest, bulk_build_multi_vector_without_executor) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t k = 10;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(32));
    Generator             gen(CellType::FLOAT, dims, 50, 0.5, 23);
    std::vector<uint32_t> lids;
    uint64_t              subspaces = 0;
    for (uint32_t docid = 1; docid <= 3000; ++docid) {
        uint32_t count = gen.uniform(2, 4);
        vectors->set(docid, gen.multi_sample(count));
        subspaces += count;
        lids.push_back(docid);
    }
    ASSERT_TRUE(index->bulk_build(lids, nullptr));
    commit();
    EXPECT_EQ(subspaces, index->get_stats().live_entries);
    EXPECT_EQ(subspaces, index->get_stats().entries);
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 100; ++i) {
        queries.push_back(gen.sample());
    }
    for (const auto& q : queries) {
        auto result = query(q, k);
        EXPECT_EQ(k, result.size());
        expect_sorted_unique(result);
        auto df = index->distance_function_factory().for_query_vector(vectors->cells_of(q.data()));
        for (const auto& n : result) {
            auto   bundle = vectors->get_vectors(n.docid);
            double best = max_distance;
            for (uint32_t s = 0; s < bundle.subspaces(); ++s) {
                best = std::min(best, df->calc(bundle.cells(s)));
            }
            EXPECT_DOUBLE_EQ(best, n.distance);
        }
    }
    auto rows = recall_table(queries, k, {index->config().nprobe}, nullptr);
    print_table("bulk built multi-vector float x[16]", rows);
    EXPECT_GE(rows[0].recall, 0.95);
}

TEST(IvfKmeansTest, centroid_rules_follow_the_metric) {
    auto                     int8_dff = make_distance_function_factory(DistanceMetric::Euclidean, CellType::INT8);
    ivf::VectorSpace         int8_space(*int8_dff, CellType::INT8, 3, DistanceMetric::Euclidean);
    std::vector<int8_t>      a{1, -1, 127};
    std::vector<int8_t>      b{2, -2, 126};
    ivf::CentroidAccumulator int8_acc(int8_space);
    int8_acc.add(TypedCells(a.data(), CellType::INT8, 3));
    int8_acc.add(TypedCells(b.data(), CellType::INT8, 3));
    std::vector<int8_t> mean(3);
    int8_acc.write(reinterpret_cast<char*>(mean.data()));
    EXPECT_EQ((std::vector<int8_t>{2, -2, 127}), mean);

    auto                     bits_dff = make_distance_function_factory(DistanceMetric::Hamming, CellType::INT8);
    ivf::VectorSpace         bits_space(*bits_dff, CellType::INT8, 1, DistanceMetric::Hamming);
    ivf::CentroidAccumulator bits_acc(bits_space);
    for (uint8_t value : {0xc1, 0x81, 0x40}) {
        int8_t cell = static_cast<int8_t>(value);
        bits_acc.add(TypedCells(&cell, CellType::INT8, 1));
    }
    uint8_t majority = 0;
    bits_acc.write(reinterpret_cast<char*>(&majority));
    EXPECT_EQ(0xc1, majority);

    auto float_dff = make_distance_function_factory(DistanceMetric::PrenormalizedAngular, CellType::FLOAT);
    ivf::VectorSpace         unit_space(*float_dff, CellType::FLOAT, 2, DistanceMetric::PrenormalizedAngular);
    ivf::CentroidAccumulator unit_acc(unit_space);
    std::vector<float>       x{1.0f, 0.0f};
    std::vector<float>       y{0.0f, 1.0f};
    unit_acc.add(TypedCells(std::span<const float>(x)));
    unit_acc.add(TypedCells(std::span<const float>(y)));
    std::vector<float> unit(2);
    unit_acc.write(reinterpret_cast<char*>(unit.data()));
    EXPECT_NEAR(std::sqrt(0.5), unit[0], 1e-6);
    EXPECT_NEAR(std::sqrt(0.5), unit[1], 1e-6);
}

TEST(IvfKmeansTest, two_means_separates_clusters_and_falls_back_for_identical_points) {
    auto               dff = make_distance_function_factory(DistanceMetric::Euclidean, CellType::FLOAT);
    ivf::VectorSpace   space(*dff, CellType::FLOAT, 2, DistanceMetric::Euclidean);
    std::mt19937       rng(1);
    std::vector<float> data;
    for (uint32_t i = 0; i < 30; ++i) {
        data.push_back((i < 10) ? 100.0f + i * 0.01f : -100.0f - i * 0.01f);
        data.push_back(0.0f);
    }
    std::vector<TypedCells> points;
    for (uint32_t i = 0; i < 30; ++i) {
        points.emplace_back(std::span<const float>(data.data() + 2 * i, 2));
    }
    auto split = ivf::two_means(space, points, rng, 5, 0.1);
    EXPECT_FALSE(split.balanced_fallback);
    for (uint32_t i = 1; i < 30; ++i) {
        EXPECT_EQ(i < 10, split.side[i] == split.side[0]) << i;
    }
    std::vector<float>      same(2 * 8, 1.0f);
    std::vector<TypedCells> same_points;
    for (uint32_t i = 0; i < 8; ++i) {
        same_points.emplace_back(std::span<const float>(same.data() + 2 * i, 2));
    }
    auto fallback = ivf::two_means(space, same_points, rng, 5, 0.1);
    EXPECT_TRUE(fallback.balanced_fallback);
    EXPECT_EQ(4, std::count(fallback.side.begin(), fallback.side.end(), 1));
}

TEST_F(IvfIndexTest, replicas_store_copies_and_versions_hide_all_of_them) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t num_docs = 6000;
    constexpr uint32_t k = 10;
    for (bool bulk : {false, true}) {
        for (uint32_t replicas : {1u, 4u}) {
            SCOPED_TRACE(std::string(bulk ? "bulk" : "incremental") + " replicas=" + std::to_string(replicas));
            auto cfg = config(32);
            cfg.replicas = replicas;
            cfg.replica_eps = 0.1;
            cfg.rng_factor = 1.0;
            make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg);
            Generator gen(CellType::FLOAT, dims, 40, 0.5, 29);
            if (bulk) {
                std::vector<uint32_t> lids;
                for (uint32_t docid = 1; docid <= num_docs; ++docid) {
                    vectors->set(docid, gen.sample());
                    lids.push_back(docid);
                }
                ASSERT_TRUE(index->bulk_build(lids, nullptr));
            } else {
                for (uint32_t docid = 1; docid <= num_docs; ++docid) {
                    add(docid, gen.sample());
                }
            }
            commit();
            auto stats = index->get_stats();
            fprintf(stderr, "replicas=%u: %.3f entries per vector\n", replicas, double(stats.entries) / num_docs);
            EXPECT_EQ(num_docs, stats.live_entries);
            if (replicas > 1) {
                EXPECT_GT(stats.entries, uint64_t(num_docs));
            }
            std::vector<std::pair<uint32_t, Bytes>> removed;
            for (uint32_t docid = 1; docid <= num_docs; docid += 30) {
                removed.emplace_back(docid, vectors->raw(docid));
                remove(docid);
            }
            commit();
            for (const auto& r : removed) {
                EXPECT_FALSE(contains(query(r.second, k), r.first)) << r.first;
            }
            std::vector<Bytes> queries;
            for (uint32_t i = 0; i < 100; ++i) {
                queries.push_back(gen.sample());
            }
            for (const auto& q : queries) {
                expect_sorted_unique(query(q, k));
            }
            auto rows = recall_table(queries, k, {4, index->config().nprobe}, nullptr);
            print_table(bulk ? "bulk float x[16]" : "incremental float x[16]", rows);
            EXPECT_GE(rows[1].recall, 0.95);
        }
    }
}

TEST_F(IvfIndexTest, medoid_balanced_split_mode_still_works) {
    constexpr uint32_t dims = 16;
    auto               cfg = config(32);
    cfg.split_mode = IvfSplitMode::MEDOID_BALANCED;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg);
    Generator gen(CellType::FLOAT, dims, 30, 0.5, 31);
    for (uint32_t docid = 1; docid <= 3000; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    EXPECT_GT(index->get_stats().splits, 0u);
    EXPECT_EQ(index->get_stats().splits, index->get_stats().split_fallbacks);
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 50; ++i) {
        queries.push_back(gen.sample());
    }
    EXPECT_GE(recall_table(queries, 10, {index->config().nprobe}, nullptr)[0].recall, 0.95);
    EXPECT_EQ("medoid_balanced", state().get()["cfg"]["split_mode"].asString().make_string());
}

TEST(IvfConfigTest, knobs_are_parsed_and_bad_values_fall_back_to_defaults) {
    std::map<std::string, std::string> env = {{"POSTING_CAPACITY", "128"},
                                              {"REPLICAS", "4"},
                                              {"REPLICA_EPS", "0.25"},
                                              {"SPLIT_MODE", "medoid_balanced"},
                                              {"BULK_BUILD", "0"},
                                              {"PRUNE_EPS", "0.5"},
                                              {"NPROBE", "0"},
                                              {"RNG_FACTOR", "abc"},
                                              {"BULK_FILL", "2.0"},
                                              {"FILTER_MIN_RATIO", "0.2"},
                                              {"REPLICA_CANDIDATES", "2"},
                                              {"REBUILD_ON_LOAD", "true"},
                                              {"PREFETCH", "0"}};
    auto                               lookup = [&env](const char* knob) -> const char* {
        auto it = env.find(knob);
        return (it == env.end()) ? nullptr : it->second.c_str();
    };
    auto                 cfg = IvfIndexConfig::from_lookup(lookup);
    const IvfIndexConfig def;
    EXPECT_EQ(128u, cfg.posting_capacity);
    EXPECT_EQ(4u, cfg.replicas);
    EXPECT_DOUBLE_EQ(0.25, cfg.replica_eps);
    EXPECT_EQ(IvfSplitMode::MEDOID_BALANCED, cfg.split_mode);
    EXPECT_FALSE(cfg.bulk_build);
    EXPECT_DOUBLE_EQ(0.5, cfg.prune_eps);
    EXPECT_DOUBLE_EQ(0.2, cfg.filter_min_ratio);
    EXPECT_EQ(def.nprobe, cfg.nprobe);
    EXPECT_DOUBLE_EQ(def.rng_factor, cfg.rng_factor);
    EXPECT_DOUBLE_EQ(def.bulk_fill, cfg.bulk_fill);
    EXPECT_EQ(def.replica_candidates, cfg.replica_candidates);
    EXPECT_FALSE(def.rebuild_on_load);
    EXPECT_TRUE(cfg.rebuild_on_load);
    EXPECT_FALSE(cfg.prefetch);
    EXPECT_EQ(37u, IvfIndexConfig::knob_names().size());
}

TEST_F(IvfIndexTest, two_phase_add_with_prepare_on_other_thread) {
    constexpr uint32_t dims = 16;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(16));
    Generator gen(CellType::FLOAT, dims, 20, 0.5, 3);
    for (uint32_t docid = 1; docid <= 500; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    auto prepare = [this](uint32_t docid, const Bytes& v) {
        std::unique_ptr<PrepareResult> result;
        std::thread                    thread(
            [&]() { result = index->prepare_add_document(docid, vectors->bundle_of(v), GenerationGuard()); });
        thread.join();
        return result;
    };

    Bytes y = gen.sample();
    auto  prepared_y = prepare(600, y);
    vectors->set(600, y);
    index->complete_add_document(600, std::move(prepared_y));
    commit();
    EXPECT_EQ(0, state().get()["prepared_retired"].asLong());
    EXPECT_TRUE(contains(query(y, 1), 600));

    Bytes x = gen.sample();
    auto  prepared_x = prepare(700, x);
    auto* prepared = dynamic_cast<ivf::PreparedAdd*>(prepared_x.get());
    ASSERT_NE(nullptr, prepared);
    ASSERT_EQ(1u, prepared->slots.size());
    ASSERT_EQ(1u, prepared->slots[0].size());
    uint32_t slot = prepared->slots[0][0];
    ASSERT_TRUE(index->slot_is_live(slot));
    uint32_t copies = 0;
    for (; copies < 200 && index->slot_is_live(slot); ++copies) {
        vectors->set(2000 + copies, x);
        index->add_document(2000 + copies);
        commit();
    }
    ASSERT_FALSE(index->slot_is_live(slot)) << "prepared posting was not split";
    vectors->set(700, x);
    index->complete_add_document(700, std::move(prepared_x));
    commit();
    EXPECT_EQ(1, state().get()["prepared_retired"].asLong());
    auto result = query(x, copies + 10);
    EXPECT_TRUE(contains(result, 700));
    for (const auto& n : result) {
        if (n.docid == 700) {
            EXPECT_EQ(0.0, n.distance);
        }
    }
}

TEST_F(IvfIndexTest, concurrent_readers_during_feed) {
    constexpr uint32_t dims = 16;
    make_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(32));
    Generator          gen(CellType::FLOAT, dims, 30, 0.5, 5);
    std::vector<Bytes> reader_queries;
    for (uint32_t i = 0; i < 16; ++i) {
        reader_queries.push_back(gen.sample());
    }
    std::atomic<bool>     stop(false);
    std::atomic<uint64_t> searches(0);
    std::atomic<uint64_t> bad_results(0);
    std::thread           searcher([&]() {
        for (uint64_t i = 0; !stop.load(std::memory_order_relaxed); ++i) {
            const auto& q = reader_queries[i % reader_queries.size()];
            auto        result = query(q, 10);
            for (size_t j = 1; j < result.size(); ++j) {
                if (result[j - 1].docid >= result[j].docid) {
                    bad_results.fetch_add(1);
                }
            }
            searches.fetch_add(1);
        }
    });
    std::thread           preparer([&]() {
        for (uint64_t i = 0; !stop.load(std::memory_order_relaxed); ++i) {
            const auto& q = reader_queries[i % reader_queries.size()];
            auto        prepared = index->prepare_add_document(1, vectors->bundle_of(q), GenerationGuard());
            EXPECT_TRUE(prepared);
        }
    });
    for (uint32_t docid = 1; docid <= 5000; ++docid) {
        add(docid, gen.sample());
        if ((docid % 7) == 0) {
            remove(docid - 3);
        }
        if ((docid % 16) == 0) {
            commit();
        }
    }
    stop.store(true);
    searcher.join();
    preparer.join();
    commit();
    EXPECT_GT(searches.load(), 0u);
    EXPECT_EQ(0u, bad_results.load());
    EXPECT_GT(state().get()["splits"].asLong(), 0);
}

namespace {

struct Op {
    enum class Kind { ADD, REMOVE, COMMIT, SWEEP };
    Kind     kind;
    uint32_t docid;
    Bytes    vector;
};

// A deterministic mix of inserts that split postings, removes, updates, GC sweeps and merges.
std::vector<Op> heavy_mutations(Generator& gen, uint32_t first_new_docid, uint32_t old_docs, uint32_t count) {
    std::vector<Op> ops;
    uint32_t        next = first_new_docid;
    for (uint32_t i = 1; i <= count; ++i) {
        uint32_t pick = gen.uniform(0, 9);
        if (pick < 4) {
            ops.push_back({Op::Kind::ADD, next++, gen.sample()});
        } else if (pick < 7) {
            ops.push_back({Op::Kind::REMOVE, gen.uniform(1, old_docs), {}});
        } else {
            uint32_t docid = gen.uniform(1, old_docs);
            ops.push_back({Op::Kind::REMOVE, docid, {}});
            ops.push_back({Op::Kind::ADD, docid, gen.sample()});
        }
        if ((i % 50) == 0) {
            ops.push_back({Op::Kind::COMMIT, 0, {}});
        }
        if ((i % 400) == 0) {
            ops.push_back({Op::Kind::SWEEP, 0, {}});
        }
    }
    ops.push_back({Op::Kind::COMMIT, 0, {}});
    return ops;
}

Bytes read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_file(const std::string& path, const Bytes& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), data.size());
    ASSERT_TRUE(out.good()) << path;
}

Bytes read_ranges(const std::string& path, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
    std::ifstream in(path, std::ios::binary);
    Bytes         result;
    for (const auto& [offset, bytes] : ranges) {
        Bytes part(bytes);
        in.seekg(offset);
        in.read(part.data(), bytes);
        EXPECT_TRUE(in.good()) << "offset " << offset;
        result.insert(result.end(), part.begin(), part.end());
    }
    return result;
}

void expect_same_counters(const IvfIndexStats& a, const IvfIndexStats& b, bool same_extents) {
    EXPECT_EQ(a.postings, b.postings);
    EXPECT_EQ(a.slot_limit, b.slot_limit);
    EXPECT_EQ(a.entries, b.entries);
    EXPECT_EQ(a.live_entries, b.live_entries);
    EXPECT_EQ(a.splits, b.splits);
    EXPECT_EQ(a.split_fallbacks, b.split_fallbacks);
    EXPECT_EQ(a.compactions, b.compactions);
    EXPECT_EQ(a.reassign_checks, b.reassign_checks);
    EXPECT_EQ(a.moves, b.moves);
    EXPECT_EQ(a.dropped_steps, b.dropped_steps);
    EXPECT_EQ(a.dropped_entries, b.dropped_entries);
    EXPECT_EQ(a.prepared_retired, b.prepared_retired);
    EXPECT_EQ(a.bytes_written, b.bytes_written);
    EXPECT_EQ(a.bulk_builds, b.bulk_builds);
    EXPECT_EQ(a.gc_checks, b.gc_checks);
    EXPECT_EQ(a.gc_compactions, b.gc_compactions);
    EXPECT_EQ(a.merges, b.merges);
    if (same_extents) {
        EXPECT_EQ(a.extents, b.extents);
        EXPECT_EQ(a.file_bytes, b.file_bytes);
    }
}

} // namespace

class IvfPersistenceTest : public IvfIndexTest {
protected:
    std::string dir;

    IvfPersistenceTest() : IvfIndexTest(), dir(ivf_test_dir() + "/persistence") {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
    }
    ~IvfPersistenceTest() override;
    std::string path(const char* name) const { return dir + "/" + name; }
    void apply(const Op& op) {
        switch (op.kind) {
        case Op::Kind::ADD:
            vectors->set(op.docid, op.vector);
            index->add_document(op.docid);
            break;
        case Op::Kind::REMOVE:
            remove(op.docid);
            break;
        case Op::Kind::COMMIT:
            commit();
            break;
        case Op::Kind::SWEEP:
            index->inc_generation();
            index->consider_compact(vespalib::datastore::CompactionStrategy());
            commit();
            break;
        }
    }
    std::unique_ptr<NearestNeighborIndexSaver> init_save(uint64_t serial_num) {
        vespalib::GenericHeader header;
        auto                    saver = index->make_saver(header, serial_num);
        EXPECT_TRUE(header.hasTag(IvfIndex::save_file_tag()));
        return saver;
    }
    void save_to(const std::string& save_path, uint64_t serial_num) {
        auto saver = init_save(serial_num);
        ASSERT_TRUE(saver);
        search::test::VectorBufferWriter writer;
        saver->save(writer);
        write_file(save_path, writer.output);
    }
    void load_from(const std::string& save_path) {
        FastOS_File file(save_path.c_str());
        ASSERT_TRUE(file.OpenReadOnly());
        vespalib::GenericHeader header;
        auto                    loader = index->make_loader(file, header);
        while (loader->load_next()) {
        }
    }
    std::vector<std::vector<Neighbor>> run_queries(const std::vector<Bytes>& queries, uint32_t nprobe) const {
        std::vector<std::vector<Neighbor>> results;
        for (const auto& q : queries) {
            NearestNeighborIndex::Stats stats;
            IvfProbeStats               probe_stats;
            results.push_back(query(q.data(), 10, nprobe, nullptr, stats, probe_stats));
        }
        return results;
    }
    std::string rejected_reason() const { return state().get()["load"]["rejected_reason"].asString().make_string(); }
};

IvfPersistenceTest::~IvfPersistenceTest() = default;

TEST_F(IvfPersistenceTest, save_and_load_restore_search_results_and_counters) {
    constexpr uint32_t dims = 16;
    auto               cfg = config(32);
    make_named_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg, path("p.ivfpostings"));
    ASSERT_TRUE(index->persistent());
    Generator gen(CellType::FLOAT, dims, 40, 0.5, 37);
    for (uint32_t docid = 1; docid <= 3000; ++docid) {
        add(docid, gen.sample());
    }
    for (uint32_t docid = 1; docid <= 3000; docid += 7) {
        remove(docid);
    }
    commit();
    save_to(path("s.nnidx"), 10);
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 50; ++i) {
        queries.push_back(gen.sample());
    }
    auto expected = run_queries(queries, 8);
    auto expected_stats = index->get_stats();
    EXPECT_GT(expected_stats.splits, 0u);

    reopen_index(DistanceMetric::Euclidean, cfg, path("p.ivfpostings"));
    EXPECT_EQ(0u, index->get_stats().postings);
    load_from(path("s.nnidx"));
    EXPECT_EQ(IvfLoadSource::SAVE_FILE, index->load_source());
    EXPECT_FALSE(index->needs_build_after_load());
    EXPECT_EQ("save_file", state().get()["load"]["source"].asString().make_string());
    EXPECT_EQ(expected, run_queries(queries, 8));
    expect_same_counters(expected_stats, index->get_stats(), true);

    for (uint32_t docid = 3001; docid <= 3500; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    for (uint32_t docid = 3001; docid <= 3500; docid += 50) {
        EXPECT_TRUE(contains(query(vectors->raw(docid), 1), docid)) << docid;
    }
}

TEST_F(IvfPersistenceTest, old_save_plus_replay_gives_the_live_index) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t initial = 3000;
    auto               cfg = config(32);
    cfg.gc_steps = 64;
    cfg.merge_fraction = 0.25;
    const std::string posting = path("p.ivfpostings");
    make_named_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg, posting);
    Generator             gen(CellType::FLOAT, dims, 40, 0.5, 41);
    std::vector<uint32_t> lids;
    for (uint32_t docid = 1; docid <= initial; ++docid) {
        vectors->set(docid, gen.sample());
        lids.push_back(docid);
    }
    ASSERT_TRUE(index->bulk_build(lids, nullptr));
    commit();
    for (uint32_t docid = initial + 1; docid <= initial + 500; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    save_to(path("s.nnidx"), 10);
    std::vector<Bytes> saved_docs;
    for (uint32_t docid = 0; docid < vectors->docid_limit(); ++docid) {
        saved_docs.push_back(vectors->has(docid) ? vectors->raw(docid) : Bytes());
    }
    auto saved_ranges = index->live_posting_ranges();
    auto saved_bytes = read_ranges(posting, saved_ranges);
    auto saved_stats = index->get_stats();

    auto ops = heavy_mutations(gen, initial + 501, initial + 500, 6000);
    for (const auto& op : ops) {
        apply(op);
    }
    auto live_stats = index->get_stats();
    EXPECT_GT(live_stats.splits, saved_stats.splits);
    EXPECT_GT(live_stats.compactions + live_stats.gc_compactions,
              saved_stats.compactions + saved_stats.gc_compactions);
    EXPECT_GT(live_stats.merges, saved_stats.merges);
    EXPECT_GT(live_stats.pinned_extents, 0u);
    EXPECT_EQ(saved_bytes, read_ranges(posting, saved_ranges)) << "a posting of the old save was overwritten";
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 50; ++i) {
        queries.push_back(gen.sample());
    }
    auto live_results = run_queries(queries, 16);

    // A crash: the old save is the newest durable one, and the transaction log replays the same operations.
    index.reset();
    vectors = std::make_unique<TestVectors>(CellType::FLOAT, dims);
    for (uint32_t docid = 0; docid < saved_docs.size(); ++docid) {
        if (!saved_docs[docid].empty()) {
            vectors->set(docid, saved_docs[docid]);
        }
    }
    reopen_index(DistanceMetric::Euclidean, cfg, posting);
    load_from(path("s.nnidx"));
    ASSERT_EQ(IvfLoadSource::SAVE_FILE, index->load_source()) << rejected_reason();
    for (const auto& op : ops) {
        apply(op);
    }
    EXPECT_EQ(live_results, run_queries(queries, 16));
    expect_same_counters(live_stats, index->get_stats(), false);
}

TEST_F(IvfPersistenceTest, freed_extents_are_reused_only_after_a_newer_snapshot_is_valid) {
    constexpr uint32_t dims = 16;
    make_named_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, config(16), path("p.ivfpostings"));
    Generator gen(CellType::FLOAT, dims, 30, 0.5, 43);
    uint32_t  docid = 0;
    auto      feed = [&](uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            ++docid;
            add(docid, gen.sample());
        }
        commit();
    };
    feed(1000);
    auto     stats = index->get_stats();
    uint64_t pinned_at_10 = stats.pinned_extents;
    EXPECT_GT(pinned_at_10, 0u);
    EXPECT_EQ(0u, stats.free_extents);

    save_to(path("s10.nnidx"), 10);
    index->on_snapshot_valid(10);
    feed(1000);
    EXPECT_EQ(0u, index->get_stats().free_extents) << "snapshot 10 being valid does not free what 10 pinned";

    save_to(path("s20.nnidx"), 20);
    uint64_t pinned_after_s20 = index->get_stats().pinned_extents;
    feed(1000);
    EXPECT_EQ(0u, index->get_stats().free_extents) << "snapshot 20 is saved but not valid yet";
    uint64_t pinned_before_valid = index->get_stats().pinned_extents;

    index->on_snapshot_valid(20);
    EXPECT_EQ(0u, index->get_stats().free_extents) << "the writer frees pinned extents at its next commit";
    commit();
    stats = index->get_stats();
    EXPECT_EQ(pinned_at_10, stats.free_extents);
    EXPECT_EQ(pinned_before_valid - pinned_at_10, stats.pinned_extents);

    save_to(path("s30.nnidx"), 30);
    index->on_snapshot_valid(30);
    commit();
    EXPECT_EQ(pinned_before_valid - pinned_after_s20, index->get_stats().pinned_extents)
        << "what snapshot 30 pinned waits for a snapshot newer than 30";
}

TEST_F(IvfPersistenceTest, saves_that_never_become_valid_keep_the_old_snapshot_loadable) {
    constexpr uint32_t dims = 16;
    constexpr uint32_t initial = 3000;
    auto               cfg = config(32);
    cfg.gc_steps = 64;
    cfg.merge_fraction = 0.25;
    const std::string posting = path("p.ivfpostings");
    make_named_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg, posting);
    Generator gen(CellType::FLOAT, dims, 40, 0.5, 53);
    for (uint32_t docid = 1; docid <= initial; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    save_to(path("s10.nnidx"), 10);
    index->on_snapshot_valid(10);
    std::vector<Bytes> saved_docs;
    for (uint32_t docid = 0; docid < vectors->docid_limit(); ++docid) {
        saved_docs.push_back(vectors->has(docid) ? vectors->raw(docid) : Bytes());
    }
    auto saved_ranges = index->live_posting_ranges();
    auto saved_bytes = read_ranges(posting, saved_ranges);

    // Flushes 20 and 30 write their save files but fail before their snapshots are marked valid.
    std::vector<Op> ops;
    uint32_t        next_docid = initial + 1;
    for (uint64_t serial : {20u, 30u, 40u}) {
        auto round = heavy_mutations(gen, next_docid, initial, 2000);
        for (const auto& op : round) {
            if (op.kind == Op::Kind::ADD) {
                next_docid = std::max(next_docid, op.docid + 1);
            }
            apply(op);
        }
        ops.insert(ops.end(), round.begin(), round.end());
        if (serial < 40) {
            save_to(path("failed.nnidx"), serial);
        }
    }
    auto live_stats = index->get_stats();
    EXPECT_GT(live_stats.splits, 0u);
    EXPECT_GT(live_stats.merges, 0u);
    EXPECT_EQ(saved_bytes, read_ranges(posting, saved_ranges)) << "a posting of the valid snapshot was overwritten";
    std::vector<Bytes> queries;
    for (uint32_t i = 0; i < 50; ++i) {
        queries.push_back(gen.sample());
    }
    auto live_results = run_queries(queries, 16);

    index.reset();
    vectors = std::make_unique<TestVectors>(CellType::FLOAT, dims);
    for (uint32_t docid = 0; docid < saved_docs.size(); ++docid) {
        if (!saved_docs[docid].empty()) {
            vectors->set(docid, saved_docs[docid]);
        }
    }
    reopen_index(DistanceMetric::Euclidean, cfg, posting);
    load_from(path("s10.nnidx"));
    ASSERT_EQ(IvfLoadSource::SAVE_FILE, index->load_source()) << rejected_reason();
    for (const auto& op : ops) {
        apply(op);
    }
    EXPECT_EQ(live_results, run_queries(queries, 16));
    expect_same_counters(live_stats, index->get_stats(), false);
}

TEST_F(IvfPersistenceTest, unusable_save_files_are_rejected_and_leave_the_index_empty) {
    constexpr uint32_t dims = 16;
    auto               cfg = config(32);
    const std::string  posting = path("p.ivfpostings");
    make_named_index(CellType::FLOAT, dims, DistanceMetric::Euclidean, cfg, posting);
    Generator gen(CellType::FLOAT, dims, 40, 0.5, 47);
    for (uint32_t docid = 1; docid <= 1500; ++docid) {
        add(docid, gen.sample());
    }
    commit();
    save_to(path("good.nnidx"), 10);
    uint64_t postings = index->get_stats().postings;
    Bytes    good = read_file(path("good.nnidx"));

    auto expect_rejected = [&](const IvfIndexConfig& load_cfg, const Bytes& data, const char* label,
                               const char* reason_part) {
        SCOPED_TRACE(label);
        write_file(path("case.nnidx"), data);
        reopen_index(DistanceMetric::Euclidean, load_cfg, posting);
        load_from(path("case.nnidx"));
        EXPECT_TRUE(index->needs_build_after_load());
        EXPECT_EQ(IvfLoadSource::NONE, index->load_source());
        EXPECT_EQ(0u, index->get_stats().postings);
        EXPECT_NE(std::string::npos, rejected_reason().find(reason_part)) << rejected_reason();
    };
    Bytes truncated(good.begin(), good.begin() + good.size() / 2);
    expect_rejected(cfg, truncated, "truncated", "");
    Bytes flipped = good;
    flipped[flipped.size() / 2] ^= 0x10;
    expect_rejected(cfg, flipped, "bad checksum", "checksum");
    expect_rejected(config(64), good, "other posting-capacity", "posting-capacity");
    auto rebuild = cfg;
    rebuild.rebuild_on_load = true;
    expect_rejected(rebuild, good, "rebuild-on-load", "rebuild-on-load");

    reopen_index(DistanceMetric::Euclidean, cfg, posting);
    load_from(path("good.nnidx"));
    EXPECT_EQ(IvfLoadSource::SAVE_FILE, index->load_source()) << rejected_reason();
    EXPECT_EQ(postings, index->get_stats().postings);

    index.reset();
    std::filesystem::resize_file(posting, 4096 + 1);
    expect_rejected(cfg, good, "short posting file", "shorter");
    std::vector<uint32_t> lids;
    for (uint32_t docid = 1; docid <= 1500; ++docid) {
        lids.push_back(docid);
    }
    ASSERT_TRUE(index->bulk_build(lids, nullptr));
    commit();
    EXPECT_GT(index->get_stats().postings, 0u);
    expect_rejected(cfg, good, "posting file rewritten", "rewritten");
}

namespace {

const std::string attr_spec("tensor<float>(x[8])");
const std::string attr_dir("ivf_attr_test");
const std::string attr_name(attr_dir + "/my_attr");

std::unique_ptr<vespalib::eval::Value> make_tensor(const Bytes& bytes) {
    TensorSpec spec(attr_spec);
    for (uint32_t i = 0; i < 8; ++i) {
        float x;
        std::memcpy(&x, bytes.data() + i * sizeof(float), sizeof(float));
        spec.add({{"x", i}}, x);
    }
    return vespalib::eval::value_from_spec(spec, FastValueBuilderFactory::get());
}

// An attribute with an index; with ivf params (as the schema's 'ivf' block produces) the index is IVF.
std::shared_ptr<DenseTensorAttribute> make_attr(const NearestNeighborIndexFactory&                      factory,
                                                const std::optional<search::attribute::IvfIndexParams>& ivf) {
    search::attribute::Config cfg(search::attribute::BasicType::TENSOR, search::attribute::CollectionType::SINGLE);
    cfg.setTensorType(ValueType::from_spec(attr_spec));
    cfg.set_distance_metric(DistanceMetric::Euclidean);
    cfg.set_hnsw_index_params(HnswIndexParams(16, 100, DistanceMetric::Euclidean));
    if (ivf.has_value()) {
        cfg.set_ivf_index_params(*ivf);
    }
    return std::make_shared<DenseTensorAttribute>(attr_name, cfg, factory);
}

const IvfIndex* ivf_of(const DenseTensorAttribute& attr) {
    return dynamic_cast<const IvfIndex*>(attr.nearest_neighbor_index());
}

std::vector<Neighbor> attr_query(const DenseTensorAttribute& attr, const Bytes& q, uint32_t k) {
    auto*                       index = attr.nearest_neighbor_index();
    TypedCells                  cells(q.data(), CellType::FLOAT, 8);
    auto                        df = index->distance_function_factory().for_query_vector(cells);
    NearestNeighborIndex::Stats stats;
    return index->find_top_k(stats, k, *df, k, 0.0, false, Deadline::never(), max_distance);
}

void set_tensor(DenseTensorAttribute& attr, uint32_t docid, const Bytes& v) {
    while (attr.getNumDocs() <= docid) {
        uint32_t new_docid = 0;
        attr.addDoc(new_docid);
    }
    attr.setTensor(docid, *make_tensor(v));
    attr.commit();
}

} // namespace

TEST(IvfAttributeTest, ivf_params_in_attribute_config_select_the_ivf_index) {
    search::test::DirectoryHandler     dir_handler(attr_dir);
    DefaultNearestNeighborIndexFactory factory;
    auto                               hnsw_attr = make_attr(factory, std::nullopt);
    EXPECT_EQ(nullptr, ivf_of(*hnsw_attr));
    EXPECT_NE(nullptr, hnsw_attr->nearest_neighbor_index());
    hnsw_attr.reset();
    search::attribute::IvfIndexParams params(
        {{"posting-capacity", "32"}, {"filter-probe-mode", "adaptive"}, {"no-such-knob", "1"}});
    auto ivf_attr = make_attr(factory, params);
    ASSERT_NE(nullptr, ivf_of(*ivf_attr));
    EXPECT_EQ(32u, ivf_of(*ivf_attr)->config().posting_capacity);
    EXPECT_EQ(IvfFilterProbeMode::ADAPTIVE, ivf_of(*ivf_attr)->config().filter_probe_mode);
    EXPECT_TRUE(ivf_of(*ivf_attr)->persistent());
    EXPECT_EQ(attr_name + ".ivfpostings", ivf_of(*ivf_attr)->posting_path());
}

TEST(IvfAttributeTest, posting_file_lives_in_the_attribute_directory) {
    EXPECT_EQ("/var/db/attribute/emb/emb.ivfpostings",
              IvfIndex::posting_path_for("/var/db/attribute/emb/snapshot-1234/emb"));
    EXPECT_EQ("dir/emb.ivfpostings", IvfIndex::posting_path_for("dir/emb"));
    EXPECT_EQ("", IvfIndex::posting_path_for("emb"));
}

namespace {

struct AttrFixture {
    search::attribute::IvfIndexParams  ivf;
    search::test::DirectoryHandler     dir_handler;
    DefaultNearestNeighborIndexFactory factory;
    Generator                          gen;
    std::vector<Bytes>                 current;
    std::vector<Bytes>                 old_vectors;
    std::vector<Bytes>                 queries;
    std::vector<std::vector<Neighbor>> expected;
    IvfIndexStats                      saved_stats;

    explicit AttrFixture(search::attribute::IvfIndexParams ivf_in)
        : ivf(std::move(ivf_in)),
          dir_handler(attr_dir),
          factory(),
          gen(CellType::FLOAT, 8, 20, 0.5, 13),
          current(401),
          old_vectors(401),
          queries(),
          expected(),
          saved_stats() {}
    ~AttrFixture();

    // Feeds 400 docs, updates 40 and removes 40, records query results, saves and drops the attribute.
    void feed_and_save() {
        auto attr = make_attr(factory, ivf);
        attr->addReservedDoc();
        ASSERT_NE(nullptr, ivf_of(*attr));
        EXPECT_EQ(16u, ivf_of(*attr)->config().posting_capacity);
        for (uint32_t docid = 1; docid <= 400; ++docid) {
            current[docid] = gen.sample();
            set_tensor(*attr, docid, current[docid]);
        }
        for (uint32_t docid = 1; docid <= 40; ++docid) {
            old_vectors[docid] = current[docid];
            current[docid] = gen.sample();
            auto tensor = make_tensor(current[docid]);
            auto prepared = attr->prepare_set_tensor(docid, *tensor);
            EXPECT_TRUE(prepared);
            attr->complete_set_tensor(docid, *tensor, std::move(prepared));
            attr->commit();
        }
        for (uint32_t docid = 41; docid <= 80; ++docid) {
            old_vectors[docid] = current[docid];
            current[docid].clear();
            attr->clearDoc(docid);
            attr->commit();
        }
        check_updates_and_removes(*attr);
        EXPECT_GT(ivf_of(*attr)->get_stats().splits, 0u);
        for (uint32_t i = 0; i < 20; ++i) {
            queries.push_back(gen.sample());
            expected.push_back(attr_query(*attr, queries.back(), 10));
            EXPECT_EQ(10u, expected.back().size());
        }
        ASSERT_TRUE(attr->save());
        saved_stats = ivf_of(*attr)->get_stats();
        EXPECT_TRUE(std::filesystem::exists(attr_name + ".nnidx"));
    }
    void check_updates_and_removes(const DenseTensorAttribute& attr) const {
        for (uint32_t docid = 1; docid <= 40; ++docid) {
            auto by_new = attr_query(attr, current[docid], 1);
            ASSERT_EQ(1u, by_new.size());
            EXPECT_EQ(docid, by_new[0].docid);
            EXPECT_EQ(0.0, by_new[0].distance);
            for (const auto& n : attr_query(attr, old_vectors[docid], 5)) {
                EXPECT_TRUE(n.docid != docid || n.distance > 0.0);
            }
        }
        for (uint32_t docid = 41; docid <= 80; ++docid) {
            for (const auto& n : attr_query(attr, old_vectors[docid], 5)) {
                EXPECT_NE(docid, n.docid);
            }
        }
    }
    void check_loaded(const DenseTensorAttribute& loaded) const {
        const IvfIndex* index = ivf_of(loaded);
        ASSERT_NE(nullptr, index);
        EXPECT_EQ(360u, index->get_stats().live_entries);
        EXPECT_EQ(0u, index->get_stats().dropped_entries);
        for (size_t i = 0; i < queries.size(); ++i) {
            EXPECT_EQ(expected[i], attr_query(loaded, queries[i], 10));
        }
        check_updates_and_removes(loaded);
    }
};

AttrFixture::~AttrFixture() = default;

} // namespace

TEST(IvfAttributeTest, dense_tensor_attribute_with_ivf_params_reloads_from_the_save_file) {
    AttrFixture f(search::attribute::IvfIndexParams({{"posting-capacity", "16"}, {"nprobe", "100000"}}));
    ASSERT_NO_FATAL_FAILURE(f.feed_and_save());
    for (bool with_executor : {false, true}) {
        SCOPED_TRACE(with_executor ? "load with executor" : "load in foreground");
        vespalib::ThreadStackExecutor executor(2);
        auto                          loaded = make_attr(f.factory, f.ivf);
        ASSERT_TRUE(with_executor ? loaded->load(&executor) : loaded->load());
        ASSERT_NO_FATAL_FAILURE(f.check_loaded(*loaded));
        const IvfIndex* index = ivf_of(*loaded);
        EXPECT_EQ(IvfLoadSource::SAVE_FILE, index->load_source());
        EXPECT_EQ(0u, index->get_stats().bulk_builds);
        EXPECT_EQ(f.saved_stats.postings, index->get_stats().postings);
        // The commit that ends the load may run a GC sweep step, which drops stale entries.
        EXPECT_LE(index->get_stats().entries, f.saved_stats.entries);
        EXPECT_EQ(f.saved_stats.live_entries, index->get_stats().live_entries);
        EXPECT_EQ(f.saved_stats.splits, index->get_stats().splits);
    }
}

TEST(IvfAttributeTest, dense_tensor_attribute_rebuilds_without_a_usable_save_file) {
    AttrFixture f(search::attribute::IvfIndexParams({{"posting-capacity", "16"}, {"nprobe", "100000"}}));
    ASSERT_NO_FATAL_FAILURE(f.feed_and_save());
    auto check_bulk_built = [&f](const search::attribute::IvfIndexParams& params, const char* label) {
        SCOPED_TRACE(label);
        auto loaded = make_attr(f.factory, params);
        ASSERT_TRUE(loaded->load());
        ASSERT_NO_FATAL_FAILURE(f.check_loaded(*loaded));
        EXPECT_EQ(IvfLoadSource::BULK_BUILD, ivf_of(*loaded)->load_source());
        EXPECT_EQ(1u, ivf_of(*loaded)->get_stats().bulk_builds);
    };
    auto rebuild = f.ivf.params();
    rebuild.emplace_back("rebuild-on-load", "1");
    check_bulk_built(search::attribute::IvfIndexParams(rebuild), "rebuild-on-load");
    // The bulk build above started a new posting file, so the save no longer matches it.
    check_bulk_built(f.ivf, "posting file rewritten");
    {
        auto holder = make_attr(f.factory, f.ivf);
        EXPECT_TRUE(ivf_of(*holder)->persistent());
        auto second = make_attr(f.factory, f.ivf);
        EXPECT_FALSE(ivf_of(*second)->persistent());
        ASSERT_TRUE(second->load());
        EXPECT_EQ(IvfLoadSource::BULK_BUILD, ivf_of(*second)->load_source());
    }
    std::filesystem::remove(attr_name + ".nnidx");
    check_bulk_built(f.ivf, "missing save file");
}

TEST(IvfAttributeTest, saving_without_ivf_removes_the_old_posting_file) {
    AttrFixture f(search::attribute::IvfIndexParams({{"posting-capacity", "16"}}));
    ASSERT_NO_FATAL_FAILURE(f.feed_and_save());
    ASSERT_TRUE(std::filesystem::exists(attr_name + ".ivfpostings"));
    auto hnsw_attr = make_attr(f.factory, std::nullopt);
    ASSERT_TRUE(hnsw_attr->load());
    EXPECT_EQ(nullptr, ivf_of(*hnsw_attr));
    EXPECT_TRUE(std::filesystem::exists(attr_name + ".ivfpostings"));
    ASSERT_TRUE(hnsw_attr->save());
    EXPECT_FALSE(std::filesystem::exists(attr_name + ".ivfpostings"));
}

GTEST_MAIN_RUN_ALL_TESTS()
