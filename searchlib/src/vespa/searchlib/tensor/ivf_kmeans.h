// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "distance_function_factory.h"

#include <vespa/eval/eval/cell_type.h>
#include <vespa/eval/eval/typed_cells.h>
#include <vespa/searchcommon/attribute/distance_metric.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace vespalib {
class Executor;
}

namespace search::tensor::ivf {

/**
 * Cell type, size and metric of the vectors being clustered.
 */
struct VectorSpace {
    const DistanceFunctionFactory&    dff;
    vespalib::eval::CellType          cell_type;
    size_t                            dims;
    size_t                            vector_bytes;
    search::attribute::DistanceMetric metric;

    VectorSpace(const DistanceFunctionFactory& dff_in, vespalib::eval::CellType cell_type_in, size_t dims_in,
                search::attribute::DistanceMetric metric_in);
    vespalib::eval::TypedCells cells(const char* data) const noexcept { return {data, cell_type, dims}; }
    // Binds a vector of any cell type. Insertion functions may reference the cells without converting them, so
    // vectors of another cell type than the centroids are bound as query vectors, which converts them.
    BoundDistanceFunction::UP bind(vespalib::eval::TypedCells vector) const;
};

/**
 * Accumulates vectors and writes their centroid in the vectors' cell type.
 *
 * Hamming uses the per-bit majority. Other metrics use the arithmetic mean; int8 cells are rounded and clamped.
 * For prenormalized angular the mean is scaled to the average norm of the added vectors.
 */
class CentroidAccumulator {
    const VectorSpace&  _space;
    bool                _bits;
    std::vector<double> _sum;
    double              _norm_sum;
    uint32_t            _count;

public:
    explicit CentroidAccumulator(const VectorSpace& space);
    ~CentroidAccumulator();
    void clear();
    void add(vespalib::eval::TypedCells cells);
    uint32_t count() const noexcept { return _count; }
    // Writes space.vector_bytes bytes. Requires count() > 0.
    void write(char* dst) const;
};

// Writes cells as a centroid of the space, converting the cell type if needed.
void write_as_centroid(const VectorSpace& space, vespalib::eval::TypedCells cells, char* dst);

struct TwoMeansResult {
    std::vector<uint8_t> side; // 0 or 1 per point
    std::vector<char>    centroid_0;
    std::vector<char>    centroid_1;
    bool                 balanced_fallback;
    TwoMeansResult();
    ~TwoMeansResult();
};

/**
 * Splits points (at least 2) in two with 2-means: seeds are a random point and the point farthest from it, then
 * at most iterations Lloyd iterations. Each point ends on the side whose centroid is nearest. If one side gets
 * fewer than min_side of the points, the points are instead split in two halves by d(v,c0) - d(v,c1).
 */
TwoMeansResult two_means(const VectorSpace& space, const std::vector<vespalib::eval::TypedCells>& points,
                         std::mt19937& rng, uint32_t iterations, double min_side);

/**
 * Splits points (at least 2) in two equal halves around two medoids: seeds are the farthest pair of a sample, then
 * three rounds of a median split by d(v,c0) - d(v,c1) with medoids picked from a sample of each half.
 */
TwoMeansResult medoid_balanced_split(const VectorSpace& space, const std::vector<vespalib::eval::TypedCells>& points,
                                     uint32_t sample_size, uint32_t rounds);

/**
 * How many postings a vector is stored in: at most replicas of the candidates nearest postings, skipping a
 * candidate c when rng_factor * d(c, s) <= d(v, c) for an already selected s (SPANN RNG rule, 0 = off), and
 * candidates with d(v, c) > d0 + eps * |d0| where d0 is the nearest distance (eps 0 = off).
 */
struct ReplicaRule {
    uint32_t replicas;
    uint32_t candidates;
    double   eps;
    double   rng_factor;
};

// Picks postings from candidates sorted by increasing distance; centroid_of(id) returns a candidate's centroid.
std::vector<uint32_t> select_replicas(const VectorSpace&                                         space,
                                      const std::vector<std::pair<double, uint32_t>>&            candidates,
                                      const ReplicaRule&                                         rule,
                                      const std::function<vespalib::eval::TypedCells(uint32_t)>& centroid_of);

constexpr uint32_t no_cluster = std::numeric_limits<uint32_t>::max();

/**
 * Lloyd k-means with random initial centroids. Returns k centroids, k * space.vector_bytes bytes.
 * The assignment step runs as tasks on the executor when it is not nullptr.
 */
std::vector<char> kmeans(const VectorSpace& space, const std::vector<vespalib::eval::TypedCells>& points, uint32_t k,
                         uint32_t iterations, std::mt19937& rng, vespalib::Executor* executor);

/**
 * Partitions points into about k clusters: k-means on a sample of at most max_sample points, first into about
 * sqrt(k) coarse clusters, then each coarse cluster into its share of fine clusters. Every point is then assigned to
 * fine clusters by the replica rule, among the fine centroids of its coarse_probes nearest coarse clusters.
 * coarse_k is the number of coarse clusters; 0 means round(sqrt(k)).
 * Returns rule.replicas fine cluster ids per point (unused ones are no_cluster) and sets clusters to the number
 * of fine clusters.
 */
std::vector<uint32_t> two_level_partition(const VectorSpace&                             space,
                                          const std::vector<vespalib::eval::TypedCells>& points, uint32_t k,
                                          uint32_t max_sample, uint32_t iterations, uint32_t coarse_k,
                                          uint32_t coarse_probes, const ReplicaRule& rule, std::mt19937& rng,
                                          vespalib::Executor* executor, uint32_t& clusters);

// Index of the nearest of the given centroids (count * space.vector_bytes bytes).
uint32_t nearest_centroid(const BoundDistanceFunction& df, const VectorSpace& space, const char* centroids,
                          uint32_t count);

// Runs fn(task) for task in [0, tasks), on the executor if given, and waits for all of them.
void run_tasks(vespalib::Executor* executor, uint32_t tasks, const std::function<void(uint32_t)>& fn);

} // namespace search::tensor::ivf
