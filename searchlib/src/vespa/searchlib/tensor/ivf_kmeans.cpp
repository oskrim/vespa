// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "ivf_kmeans.h"

#include "bound_distance_function.h"

#include <vespa/eval/eval/int8float.h>
#include <vespa/vespalib/util/bfloat16.h>
#include <vespa/vespalib/util/count_down_latch.h>
#include <vespa/vespalib/util/executor.h>
#include <vespa/vespalib/util/lambdatask.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

using search::attribute::DistanceMetric;
using vespalib::BFloat16;
using vespalib::eval::CellType;
using vespalib::eval::CellTypeUtils;
using vespalib::eval::TypedCells;

namespace search::tensor::ivf {

namespace {

constexpr uint32_t max_assign_tasks = 64;
constexpr uint32_t min_points_per_task = 1024;

std::vector<char> copy_cells(const VectorSpace& space, TypedCells cells) {
    std::vector<char> result(space.vector_bytes);
    write_as_centroid(space, cells, result.data());
    return result;
}

uint32_t assign_tasks(size_t points) {
    return std::clamp<size_t>(points / min_points_per_task, 1, max_assign_tasks);
}

// Assigns each point to the nearer of two centroids; returns the number of points on side 1.
uint32_t assign_two(const VectorSpace& space, const std::vector<TypedCells>& points, const std::vector<char>& c0,
                    const std::vector<char>& c1, std::vector<uint8_t>& side) {
    auto     df0 = space.bind(space.cells(c0.data()));
    auto     df1 = space.bind(space.cells(c1.data()));
    uint32_t ones = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        side[i] = (df1->calc(points[i]) < df0->calc(points[i])) ? 1 : 0;
        ones += side[i];
    }
    return ones;
}

void write_side_means(const VectorSpace& space, const std::vector<TypedCells>& points,
                      const std::vector<uint8_t>& side, std::vector<char>& c0, std::vector<char>& c1) {
    CentroidAccumulator acc0(space);
    CentroidAccumulator acc1(space);
    for (size_t i = 0; i < points.size(); ++i) {
        (side[i] == 0 ? acc0 : acc1).add(points[i]);
    }
    acc0.write(c0.data());
    acc1.write(c1.data());
}

} // namespace

VectorSpace::VectorSpace(const DistanceFunctionFactory& dff_in, CellType cell_type_in, size_t dims_in,
                         DistanceMetric metric_in)
    : dff(dff_in),
      cell_type(cell_type_in),
      dims(dims_in),
      vector_bytes(CellTypeUtils::mem_size(cell_type_in, dims_in)),
      metric(metric_in) {
}

BoundDistanceFunction::UP VectorSpace::bind(TypedCells vector) const {
    return (vector.type == cell_type) ? dff.for_insertion_vector(vector) : dff.for_query_vector(vector);
}

CentroidAccumulator::CentroidAccumulator(const VectorSpace& space)
    : _space(space),
      _bits(space.metric == DistanceMetric::Hamming && space.cell_type == CellType::INT8),
      _sum(_bits ? space.vector_bytes * 8 : space.dims, 0.0),
      _norm_sum(0.0),
      _count(0) {
}

CentroidAccumulator::~CentroidAccumulator() = default;

void CentroidAccumulator::clear() {
    std::fill(_sum.begin(), _sum.end(), 0.0);
    _norm_sum = 0.0;
    _count = 0;
}

void CentroidAccumulator::add(TypedCells cells) {
    assert(cells.size == _space.dims);
    ++_count;
    if (_bits) {
        const auto* bytes = static_cast<const uint8_t*>(cells.data);
        for (size_t b = 0; b < _space.vector_bytes; ++b) {
            for (uint32_t bit = 0; bit < 8; ++bit) {
                if ((bytes[b] & (0x80u >> bit)) != 0) {
                    _sum[b * 8 + bit] += 1.0;
                }
            }
        }
        return;
    }
    double norm_sq = 0.0;
    auto   accumulate = [&](auto values) {
        for (size_t i = 0; i < _space.dims; ++i) {
            double value = static_cast<double>(static_cast<float>(values[i]));
            _sum[i] += value;
            norm_sq += value * value;
        }
    };
    switch (cells.type) {
    case CellType::DOUBLE: {
        const auto* values = static_cast<const double*>(cells.data);
        for (size_t i = 0; i < _space.dims; ++i) {
            _sum[i] += values[i];
            norm_sq += values[i] * values[i];
        }
        break;
    }
    case CellType::FLOAT:
        accumulate(static_cast<const float*>(cells.data));
        break;
    case CellType::BFLOAT16:
        accumulate(static_cast<const BFloat16*>(cells.data));
        break;
    case CellType::INT8:
        accumulate(static_cast<const int8_t*>(cells.data));
        break;
    }
    _norm_sum += std::sqrt(norm_sq);
}

void CentroidAccumulator::write(char* dst) const {
    assert(_count > 0);
    if (_bits) {
        auto* bytes = reinterpret_cast<uint8_t*>(dst);
        for (size_t b = 0; b < _space.vector_bytes; ++b) {
            uint8_t value = 0;
            for (uint32_t bit = 0; bit < 8; ++bit) {
                if (2.0 * _sum[b * 8 + bit] > _count) {
                    value |= (0x80u >> bit);
                }
            }
            bytes[b] = value;
        }
        return;
    }
    std::vector<double> mean(_space.dims);
    double              norm_sq = 0.0;
    for (size_t i = 0; i < _space.dims; ++i) {
        mean[i] = _sum[i] / _count;
        norm_sq += mean[i] * mean[i];
    }
    if (_space.metric == DistanceMetric::PrenormalizedAngular && norm_sq > 0.0) {
        double scale = (_norm_sum / _count) / std::sqrt(norm_sq);
        for (auto& value : mean) {
            value *= scale;
        }
    }
    switch (_space.cell_type) {
    case CellType::DOUBLE:
        std::memcpy(dst, mean.data(), _space.vector_bytes);
        break;
    case CellType::FLOAT:
        for (size_t i = 0; i < _space.dims; ++i) {
            float value = static_cast<float>(mean[i]);
            std::memcpy(dst + i * sizeof(float), &value, sizeof(float));
        }
        break;
    case CellType::BFLOAT16:
        for (size_t i = 0; i < _space.dims; ++i) {
            BFloat16 value(static_cast<float>(mean[i]));
            std::memcpy(dst + i * sizeof(BFloat16), &value, sizeof(BFloat16));
        }
        break;
    case CellType::INT8:
        for (size_t i = 0; i < _space.dims; ++i) {
            dst[i] = static_cast<char>(static_cast<int8_t>(std::clamp(std::round(mean[i]), -128.0, 127.0)));
        }
        break;
    }
}

void write_as_centroid(const VectorSpace& space, TypedCells cells, char* dst) {
    if (cells.type == space.cell_type) {
        std::memcpy(dst, cells.data, space.vector_bytes);
        return;
    }
    CentroidAccumulator acc(space);
    acc.add(cells);
    acc.write(dst);
}

TwoMeansResult::TwoMeansResult() : side(), centroid_0(), centroid_1(), balanced_fallback(false) {
}

TwoMeansResult::~TwoMeansResult() = default;

TwoMeansResult two_means(const VectorSpace& space, const std::vector<TypedCells>& points, std::mt19937& rng,
                         uint32_t iterations, double min_side_fraction) {
    size_t n = points.size();
    assert(n >= 2);
    TwoMeansResult result;
    result.side.assign(n, 0);
    uint32_t seed = std::uniform_int_distribution<uint32_t>(0, n - 1)(rng);
    uint32_t farthest = (seed + 1) % n;
    {
        auto   df = space.bind(points[seed]);
        double farthest_distance = -1.0;
        for (size_t i = 0; i < n; ++i) {
            double distance = df->calc(points[i]);
            if (distance > farthest_distance) {
                farthest_distance = distance;
                farthest = i;
            }
        }
    }
    result.centroid_0 = copy_cells(space, points[seed]);
    result.centroid_1 = copy_cells(space, points[farthest]);
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        uint32_t ones = assign_two(space, points, result.centroid_0, result.centroid_1, result.side);
        if (ones == 0 || ones == n) {
            break;
        }
        write_side_means(space, points, result.side, result.centroid_0, result.centroid_1);
    }
    uint32_t ones = assign_two(space, points, result.centroid_0, result.centroid_1, result.side);
    size_t   min_side = std::max<size_t>(1, size_t(min_side_fraction * n));
    if (ones >= min_side && n - ones >= min_side) {
        return result;
    }
    auto                                     df0 = space.bind(space.cells(result.centroid_0.data()));
    auto                                     df1 = space.bind(space.cells(result.centroid_1.data()));
    std::vector<std::pair<double, uint32_t>> diffs(n);
    for (size_t i = 0; i < n; ++i) {
        diffs[i] = {df0->calc(points[i]) - df1->calc(points[i]), i};
    }
    std::sort(diffs.begin(), diffs.end());
    for (size_t i = 0; i < n; ++i) {
        result.side[diffs[i].second] = (i < (n + 1) / 2) ? 0 : 1;
    }
    write_side_means(space, points, result.side, result.centroid_0, result.centroid_1);
    result.balanced_fallback = true;
    return result;
}

namespace {

std::vector<uint32_t> stride_sample(const std::vector<uint32_t>& members, uint32_t max_samples) {
    std::vector<uint32_t> sample;
    size_t                stride = std::max<size_t>(1, (members.size() + max_samples - 1) / max_samples);
    for (size_t i = 0; i < members.size() && sample.size() < max_samples; i += stride) {
        sample.push_back(members[i]);
    }
    return sample;
}

uint32_t pick_medoid(const VectorSpace& space, const std::vector<TypedCells>& points,
                     const std::vector<uint32_t>& members, uint32_t sample_size) {
    assert(!members.empty());
    auto     sample = stride_sample(members, sample_size);
    uint32_t best = sample[0];
    double   best_sum = std::numeric_limits<double>::max();
    for (uint32_t candidate : sample) {
        auto   df = space.bind(points[candidate]);
        double sum = 0.0;
        for (uint32_t other : sample) {
            sum += df->calc(points[other]);
        }
        if (sum < best_sum) {
            best_sum = sum;
            best = candidate;
        }
    }
    return best;
}

} // namespace

TwoMeansResult medoid_balanced_split(const VectorSpace& space, const std::vector<TypedCells>& points,
                                     uint32_t sample_size, uint32_t rounds) {
    size_t n = points.size();
    assert(n >= 2 && sample_size >= 2);
    std::vector<uint32_t> all(n);
    std::iota(all.begin(), all.end(), 0u);
    auto     sample = stride_sample(all, sample_size);
    uint32_t medoid_0 = sample[0];
    uint32_t medoid_1 = sample[sample.size() > 1 ? 1 : 0];
    double   farthest = -1.0;
    for (size_t a = 0; a < sample.size(); ++a) {
        auto df = space.bind(points[sample[a]]);
        for (size_t b = a + 1; b < sample.size(); ++b) {
            double distance = df->calc(points[sample[b]]);
            if (distance > farthest) {
                farthest = distance;
                medoid_0 = sample[a];
                medoid_1 = sample[b];
            }
        }
    }
    TwoMeansResult result;
    result.side.assign(n, 0);
    std::vector<std::pair<double, uint32_t>> diffs(n);
    for (uint32_t round = 0; round <= rounds; ++round) {
        auto df_0 = space.bind(points[medoid_0]);
        auto df_1 = space.bind(points[medoid_1]);
        for (size_t i = 0; i < n; ++i) {
            diffs[i] = {df_0->calc(points[i]) - df_1->calc(points[i]), i};
        }
        std::sort(diffs.begin(), diffs.end());
        for (size_t i = 0; i < n; ++i) {
            result.side[diffs[i].second] = (i < (n + 1) / 2) ? 0 : 1;
        }
        if (round == rounds) {
            break;
        }
        std::vector<uint32_t> members_0;
        std::vector<uint32_t> members_1;
        for (size_t i = 0; i < n; ++i) {
            (result.side[i] == 0 ? members_0 : members_1).push_back(i);
        }
        medoid_0 = pick_medoid(space, points, members_0, sample_size);
        medoid_1 = pick_medoid(space, points, members_1, sample_size);
    }
    result.centroid_0 = copy_cells(space, points[medoid_0]);
    result.centroid_1 = copy_cells(space, points[medoid_1]);
    result.balanced_fallback = true;
    return result;
}

std::vector<uint32_t> select_replicas(const VectorSpace&                              space,
                                      const std::vector<std::pair<double, uint32_t>>& candidates,
                                      const ReplicaRule&                              rule,
                                      const std::function<TypedCells(uint32_t)>&      centroid_of) {
    std::vector<uint32_t> selected;
    if (candidates.empty()) {
        return selected;
    }
    double nearest = candidates.front().first;
    double limit = nearest + rule.eps * std::abs(nearest);
    for (const auto& candidate : candidates) {
        if (selected.size() >= rule.replicas || (rule.eps > 0.0 && candidate.first > limit)) {
            break;
        }
        bool accepted = true;
        if (rule.rng_factor > 0.0 && !selected.empty()) {
            auto df = space.bind(centroid_of(candidate.second));
            for (uint32_t other : selected) {
                if (rule.rng_factor * df->calc(centroid_of(other)) <= candidate.first) {
                    accepted = false;
                    break;
                }
            }
        }
        if (accepted) {
            selected.push_back(candidate.second);
        }
    }
    return selected;
}

uint32_t nearest_centroid(const BoundDistanceFunction& df, const VectorSpace& space, const char* centroids,
                          uint32_t count) {
    uint32_t best = 0;
    double   best_distance = std::numeric_limits<double>::max();
    for (uint32_t c = 0; c < count; ++c) {
        double distance = df.calc(space.cells(centroids + size_t(c) * space.vector_bytes));
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }
    return best;
}

std::vector<char> kmeans(const VectorSpace& space, const std::vector<TypedCells>& points, uint32_t k,
                         uint32_t iterations, std::mt19937& rng, vespalib::Executor* executor) {
    size_t m = points.size();
    assert(m > 0 && k > 0);
    k = std::min<size_t>(k, m);
    std::vector<char> centroids(size_t(k) * space.vector_bytes);
    std::vector<uint32_t> all(m);
    std::iota(all.begin(), all.end(), 0u);
    std::vector<uint32_t> seeds;
    seeds.reserve(k);
    std::sample(all.begin(), all.end(), std::back_inserter(seeds), k, rng);
    for (uint32_t c = 0; c < k; ++c) {
        write_as_centroid(space, points[seeds[c]], centroids.data() + size_t(c) * space.vector_bytes);
    }
    std::vector<uint32_t>            assignment(m, 0);
    std::vector<CentroidAccumulator> accs;
    accs.reserve(k);
    for (uint32_t c = 0; c < k; ++c) {
        accs.emplace_back(space);
    }
    uint32_t tasks = assign_tasks(m);
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        run_tasks(executor, tasks, [&](uint32_t task) {
            size_t end = m * (task + 1) / tasks;
            for (size_t i = m * task / tasks; i < end; ++i) {
                auto df = space.bind(points[i]);
                assignment[i] = nearest_centroid(*df, space, centroids.data(), k);
            }
        });
        for (auto& acc : accs) {
            acc.clear();
        }
        for (size_t i = 0; i < m; ++i) {
            accs[assignment[i]].add(points[i]);
        }
        for (uint32_t c = 0; c < k; ++c) {
            char* dst = centroids.data() + size_t(c) * space.vector_bytes;
            if (accs[c].count() > 0) {
                accs[c].write(dst);
            } else {
                uint32_t pick = std::uniform_int_distribution<size_t>(0, m - 1)(rng);
                write_as_centroid(space, points[pick], dst);
            }
        }
    }
    return centroids;
}

std::vector<uint32_t> two_level_partition(const VectorSpace& space, const std::vector<TypedCells>& points, uint32_t k,
                                          uint32_t max_sample, uint32_t iterations, uint32_t coarse_k_wanted,
                                          uint32_t coarse_probes, const ReplicaRule& rule, std::mt19937& rng,
                                          vespalib::Executor* executor, uint32_t& clusters) {
    size_t n = points.size();
    assert(n > 0 && k > 0 && max_sample > 0 && coarse_probes > 0);
    std::vector<uint32_t> all(n);
    std::iota(all.begin(), all.end(), 0u);
    std::vector<uint32_t> sample_ids;
    sample_ids.reserve(std::min<size_t>(n, max_sample));
    std::sample(all.begin(), all.end(), std::back_inserter(sample_ids), max_sample, rng);
    std::vector<TypedCells> sample;
    sample.reserve(sample_ids.size());
    for (uint32_t id : sample_ids) {
        sample.push_back(points[id]);
    }
    size_t   m = sample.size();
    size_t   coarse_default = std::lround(std::sqrt(double(k)));
    uint32_t coarse_k = std::clamp<size_t>((coarse_k_wanted > 0) ? coarse_k_wanted : coarse_default, 1, m);
    auto     coarse = kmeans(space, sample, coarse_k, iterations, rng, executor);
    coarse_k = coarse.size() / space.vector_bytes;

    std::vector<std::vector<TypedCells>> members(coarse_k);
    for (const auto& cells : sample) {
        auto df = space.bind(cells);
        members[nearest_centroid(*df, space, coarse.data(), coarse_k)].push_back(cells);
    }
    std::vector<uint32_t> fine_offset(coarse_k + 1, 0);
    for (uint32_t c = 0; c < coarse_k; ++c) {
        size_t share = members[c].empty() ? 0 : std::lround(double(k) * members[c].size() / m);
        fine_offset[c + 1] =
            fine_offset[c] + std::clamp<size_t>(share, members[c].empty() ? 0 : 1, members[c].size());
    }
    clusters = fine_offset[coarse_k];
    std::vector<char> fine(size_t(clusters) * space.vector_bytes);
    uint64_t seed = rng();
    run_tasks(executor, coarse_k, [&](uint32_t c) {
        uint32_t count = fine_offset[c + 1] - fine_offset[c];
        if (count == 0) {
            return;
        }
        std::mt19937 local_rng(seed + c);
        auto         centroids = kmeans(space, members[c], count, iterations, local_rng, nullptr);
        std::memcpy(fine.data() + size_t(fine_offset[c]) * space.vector_bytes, centroids.data(), centroids.size());
    });

    uint32_t              probes = std::min(coarse_probes, coarse_k);
    uint32_t              replicas = std::max(1u, rule.replicas);
    uint32_t              candidates = std::max(replicas, rule.candidates);
    std::vector<uint32_t> assignment(n * replicas, no_cluster);
    uint32_t              tasks = assign_tasks(n);
    auto fine_of = [&](uint32_t f) noexcept { return space.cells(fine.data() + size_t(f) * space.vector_bytes); };
    run_tasks(executor, tasks, [&](uint32_t task) {
        std::vector<std::pair<double, uint32_t>> coarse_distances(coarse_k);
        std::vector<std::pair<double, uint32_t>> fine_distances;
        size_t                                   end = n * (task + 1) / tasks;
        for (size_t i = n * task / tasks; i < end; ++i) {
            auto df = space.bind(points[i]);
            for (uint32_t c = 0; c < coarse_k; ++c) {
                coarse_distances[c] = {df->calc(space.cells(coarse.data() + size_t(c) * space.vector_bytes)), c};
            }
            std::partial_sort(coarse_distances.begin(), coarse_distances.begin() + probes, coarse_distances.end());
            fine_distances.clear();
            for (uint32_t p = 0; p < probes; ++p) {
                uint32_t c = coarse_distances[p].second;
                for (uint32_t f = fine_offset[c]; f < fine_offset[c + 1]; ++f) {
                    fine_distances.emplace_back(df->calc(fine_of(f)), f);
                }
            }
            if (fine_distances.empty()) {
                for (uint32_t f = 0; f < clusters; ++f) {
                    fine_distances.emplace_back(df->calc(fine_of(f)), f);
                }
            }
            size_t keep = std::min<size_t>(replicas == 1 ? 1 : candidates, fine_distances.size());
            std::partial_sort(fine_distances.begin(), fine_distances.begin() + keep, fine_distances.end());
            fine_distances.resize(keep);
            auto selected = (replicas == 1) ? std::vector<uint32_t>{fine_distances.front().second}
                                            : select_replicas(space, fine_distances, rule, fine_of);
            for (size_t r = 0; r < selected.size(); ++r) {
                assignment[i * replicas + r] = selected[r];
            }
        }
    });
    return assignment;
}

void run_tasks(vespalib::Executor* executor, uint32_t tasks, const std::function<void(uint32_t)>& fn) {
    if (executor == nullptr || tasks <= 1) {
        for (uint32_t task = 0; task < tasks; ++task) {
            fn(task);
        }
        return;
    }
    vespalib::CountDownLatch latch(tasks);
    for (uint32_t task = 0; task < tasks; ++task) {
        auto rejected = executor->execute(vespalib::makeLambdaTask([&fn, &latch, task]() {
            fn(task);
            latch.countDown();
        }));
        if (rejected) {
            rejected->run();
        }
    }
    latch.await();
}

} // namespace search::tensor::ivf
