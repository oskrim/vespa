// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "distance_function_factory.h"
#include "doc_vector_access.h"
#include "ivf_kmeans.h"
#include "nearest_neighbor_index.h"
#include "prepare_result.h"
#include "vector_bundle.h"

#include <vespa/eval/eval/cell_type.h>
#include <vespa/searchcommon/attribute/ivf_index_params.h>
#include <vespa/eval/eval/typed_cells.h>
#include <vespa/vespalib/datastore/atomic_value_wrapper.h>
#include <vespa/vespalib/util/generation_hold_list.h>
#include <vespa/vespalib/util/generationhandler.h>
#include <vespa/vespalib/util/type_stable_vector.h>

#include <atomic>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vespalib::slime {
struct Cursor;
}

namespace search::tensor {

enum class IvfSplitMode { KMEANS2, MEDOID_BALANCED };
enum class IvfFilterProbeMode { SCALE, ADAPTIVE };
enum class IvfAdaptiveStop { POSTINGS, ENTRIES };

/**
 * Configuration of the experimental IVF disk index, from the schema's 'ivf' index block; see knob_names().
 */
struct IvfIndexConfig {
    using Lookup = std::function<const char*(const char* knob)>;

    std::string  dir;
    uint32_t     posting_capacity;
    uint32_t     nprobe;
    uint32_t     reassign_range;
    uint32_t     max_postings;
    uint32_t     max_rebalance_steps;
    IvfSplitMode split_mode;
    uint32_t     split_iters;
    double       split_min_side;
    uint32_t     replicas;
    uint32_t     replica_candidates;
    double       replica_eps;
    double       rng_factor;
    bool         bulk_build;
    double       bulk_fill;
    uint32_t     bulk_sample;
    uint32_t     bulk_iters;
    double       prune_eps;
    double       filter_min_ratio;
    double       filter_max_scale;
    uint32_t     bulk_coarse_probes;
    uint32_t     gc_steps;
    double       gc_stale_ratio;
    double       merge_fraction;
    double       split_live_fraction;
    uint32_t     bulk_coarse_k;
    bool         prefetch;
    uint32_t     file_grow_extents;
    double       file_grow_fraction;
    uint32_t     medoid_sample;
    uint32_t     medoid_rounds;
    IvfFilterProbeMode filter_probe_mode;
    uint32_t     filter_max_examined;
    double       filter_entry_read_density;
    IvfAdaptiveStop filter_adaptive_stop;
    double       filter_exact_ratio;
    double       k_entry_factor;

    explicit IvfIndexConfig(std::string dir_in);

    // Knob names without the VESPA_IVF_ prefix, e.g. "POSTING_CAPACITY".
    static const std::vector<std::string>& knob_names();

    // Reads every knob through lookup (nullptr = default). Bad or out-of-range values log a warning and keep the
    // default.
    static IvfIndexConfig from_lookup(std::string dir_in, const Lookup& lookup);

    // Config from the schema's 'ivf' index block: knob names in kebab case (posting-capacity), plus an optional
    // "dir" for the posting file (default $VESPA_HOME/var/tmp). Unknown names log a warning.
    static IvfIndexConfig from_params(const search::attribute::IvfIndexParams& params);

    // Resets out-of-range values to their defaults, with one warning each.
    void sanitize();
};

struct IvfIndexStats {
    uint64_t postings;
    uint64_t slot_limit;
    uint64_t entries;
    uint64_t live_entries;
    uint64_t splits;
    uint64_t split_fallbacks;
    uint64_t compactions;
    uint64_t reassign_checks;
    uint64_t moves;
    uint64_t dropped_steps;
    uint64_t dropped_entries;
    uint64_t prepared_retired;
    uint64_t extents;
    uint64_t free_extents;
    uint64_t file_bytes;
    uint64_t bytes_written;
    uint64_t bulk_builds;
    uint64_t gc_checks;
    uint64_t gc_compactions;
    uint64_t merges;
};

struct IvfProbeStats {
    uint32_t postings_probed;
    uint32_t postings_examined;
    uint64_t entries_scanned;
    uint64_t centroid_scan_ns;
    IvfProbeStats() noexcept : postings_probed(0), postings_examined(0), entries_scanned(0), centroid_scan_ns(0) {}
};

namespace ivf {

struct PreparedAdd final : public PrepareResult {
    using ReadGuard = vespalib::GenerationGuard;
    uint32_t                           docid;
    ReadGuard                          read_guard;
    ReadGuard                          index_read_guard;
    std::vector<std::vector<uint32_t>> slots; // selected posting slots per subspace
    PreparedAdd(uint32_t docid_in, ReadGuard read_guard_in, ReadGuard index_read_guard_in) noexcept;
    ~PreparedAdd() override;
};

} // namespace ivf

/**
 * Experimental inverted file (IVF) index with posting lists stored on disk.
 *
 * RAM holds centroids, one atomic posting reference per centroid and a version per docid. Each posting is a
 * fixed-size extent in an unlinked file. Entries are {docid, version, cells}, and an entry is live only while its
 * version equals the docid's current version. A vector can be stored in several postings (SPANN replicas). Full
 * postings are compacted or split in two, and vectors near a split are reassigned (SPFresh LIRE).
 *
 * A lid table in RAM mirrors the docid of every entry slot of every extent, so filtered searches can count matches
 * per posting before reading it.
 *
 * Supports 1 writer thread and multiple reader threads without mutexes. Retired extents and centroid slots are
 * reused only after a generation hold cycle. The index is not persisted; it is rebuilt from the attribute on load.
 */
class IvfIndex : public NearestNeighborIndex {
public:
    static constexpr uint32_t no_slot = std::numeric_limits<uint32_t>::max();

private:
    using TypedCells = vespalib::eval::TypedCells;
    using AtomicVersion = vespalib::datastore::AtomicValueWrapper<uint32_t>;
    using VersionVector = vespalib::TypeStableVector<AtomicVersion>;
    using IdHoldList = vespalib::GenerationHoldList<uint32_t, false, true>;
    using SlotDistances = std::vector<std::pair<double, uint32_t>>;

    struct Posting {
        uint32_t extent;
        uint32_t length;
    };

    struct ReassignTask {
        std::vector<char> centroids; // old, first new and second new centroid
        uint32_t          slot_a1;
        uint32_t          slot_a2;
    };

    struct ReassignContext {
        const BoundDistanceFunction& df_old;
        const BoundDistanceFunction& df_1;
        const BoundDistanceFunction& df_2;
        uint32_t                     slot_a1;
        uint32_t                     slot_a2;
    };

    struct Counters {
        std::atomic<uint64_t> postings{0};
        std::atomic<uint64_t> entries{0};
        std::atomic<uint64_t> live_entries{0};
        std::atomic<uint64_t> splits{0};
        std::atomic<uint64_t> split_fallbacks{0};
        std::atomic<uint64_t> compactions{0};
        std::atomic<uint64_t> reassign_checks{0};
        std::atomic<uint64_t> moves{0};
        std::atomic<uint64_t> dropped_steps{0};
        std::atomic<uint64_t> dropped_entries{0};
        std::atomic<uint64_t> prepared_retired{0};
        std::atomic<uint64_t> extents{0};
        std::atomic<uint64_t> free_extents{0};
        std::atomic<uint64_t> file_bytes{0};
        std::atomic<uint64_t> bytes_written{0};
        std::atomic<uint64_t> bulk_builds{0};
        std::atomic<uint64_t> gc_checks{0};
        std::atomic<uint64_t> gc_compactions{0};
        std::atomic<uint64_t> merges{0};
    };

    const DocVectorAccess&                   _vectors;
    std::unique_ptr<DistanceFunctionFactory> _distance_ff;
    IvfIndexConfig                           _cfg;
    vespalib::eval::CellType                 _cell_type;
    size_t                                   _vector_size;
    size_t                                   _vector_bytes;
    size_t                                   _entry_bytes;
    size_t                                   _extent_bytes;
    size_t                                   _centroid_bytes;
    ivf::VectorSpace                         _space;
    int                                      _fd;

    void*                 _slot_table;
    size_t                _slot_table_bytes;
    uint64_t*             _slot_refs;
    char*                 _centroids;
    std::atomic<uint32_t> _slot_limit;
    uint32_t*             _extent_lids;
    size_t                _lid_table_bytes;
    uint32_t              _max_extents;

    VersionVector         _versions;
    std::atomic<uint32_t> _version_limit;

    vespalib::GenerationHandler _generation_handler;
    IdHoldList                  _slot_hold;
    IdHoldList                  _extent_hold;
    std::vector<uint32_t>       _free_slots;
    std::vector<uint32_t>       _free_extents;
    uint32_t                    _extent_limit;
    uint32_t                    _file_extents;
    uint32_t                    _gc_cursor;

    std::deque<ReassignTask>  _reassign_queue;
    std::vector<char>         _entry_buf;
    std::mt19937              _rng;
    Counters                  _counters;
    mutable std::atomic<bool> _fadvise_warned;

    // Reader safe accessors.
    static uint64_t make_ref(uint32_t extent, uint32_t length) noexcept {
        return (static_cast<uint64_t>(extent + 1) << 32) | length;
    }
    static Posting decode_ref(uint64_t ref) noexcept {
        return {static_cast<uint32_t>(ref >> 32) - 1, static_cast<uint32_t>(ref)};
    }
    uint64_t acquire_ref(uint32_t slot) const noexcept;
    void release_ref(uint32_t slot, uint64_t ref) noexcept;
    TypedCells centroid(uint32_t slot) const noexcept;
    TypedCells entry_cells(const char* entry) const noexcept;
    static uint32_t entry_docid(const char* entry) noexcept;
    static uint32_t entry_version(const char* entry) noexcept;
    uint32_t acquire_version(uint32_t docid) const noexcept;
    bool entry_is_live(const char* entry) const noexcept;
    SlotDistances top_slots(const BoundDistanceFunction& df, uint32_t count, uint32_t exclude_a, uint32_t exclude_b,
                            Stats* stats) const;
    void read_posting(Posting posting, std::vector<char>& buf) const;
    uint32_t lid_at(uint32_t extent, uint32_t index) const noexcept;
    void set_lids(uint32_t extent, uint32_t first, const char* entries, uint32_t count) noexcept;
    std::vector<Neighbor> search_exact(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                       const GlobalFilter& filter, const vespalib::Deadline& doom,
                                       double distance_threshold) const;
    std::vector<Neighbor> search_adaptive(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                          const BoundDistanceFunction& centroid_df, const GlobalFilter& filter,
                                          uint32_t nprobe, const vespalib::Deadline& doom, double distance_threshold,
                                          IvfProbeStats* probe_stats) const;
    std::vector<Neighbor> search(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                 const BoundDistanceFunction& centroid_df, const GlobalFilter* filter,
                                 uint32_t nprobe, const vespalib::Deadline& doom, double distance_threshold,
                                 IvfProbeStats* probe_stats) const;

    // Writer only.
    void check_cells(TypedCells cells) const;
    void encode_entry(char* dst, uint32_t docid, uint32_t version, TypedCells cells) const noexcept;
    uint32_t bump_version(uint32_t docid);
    uint32_t allocate_extent();
    void retire_extent(uint32_t extent);
    uint32_t slots_available() const noexcept;
    uint32_t create_posting(const char* entries, uint32_t count, TypedCells centroid_cells);
    void retire_slot(uint32_t slot);
    void drop_entries(uint32_t docid, uint32_t count);
    uint32_t write_new_extent(const char* entries, uint32_t count);
    using PreparedSlots = std::vector<std::vector<uint32_t>>;
    ivf::ReplicaRule replica_rule() const noexcept;
    std::vector<uint32_t> select_postings(TypedCells cells) const;
    void add_entries(uint32_t docid, VectorBundle vectors, const PreparedSlots* prepared);
    uint32_t append_copies(uint32_t docid, uint32_t version, VectorBundle vectors, const PreparedSlots* prepared);
    ivf::TwoMeansResult split_points(const std::vector<TypedCells>& points);
    void append_entry(uint32_t slot, const char* entry);
    void rebuild_full_posting(uint32_t slot, const char* pending_entry);
    void split_posting(uint32_t slot, const char* entries, uint32_t count);
    void write_bulk_cluster(std::vector<uint32_t> members, const std::vector<uint32_t>& item_lids,
                            const std::vector<TypedCells>& points);
    void write_bulk_posting(const std::vector<uint32_t>& members, const std::vector<uint32_t>& item_lids,
                            const std::vector<TypedCells>& points);
    void run_rebalance();
    bool sweep_postings(uint32_t steps);
    bool maintain_posting(uint32_t slot);
    void rewrite_posting(uint32_t slot, Posting old, const std::vector<char>& live, uint32_t count);
    void merge_posting(uint32_t slot, const std::vector<char>& live, uint32_t count);
    void process_reassign(const ReassignTask& task, uint32_t& budget);
    void reassign_from(uint32_t slot, const ReassignContext& ctx, std::unordered_set<uint32_t>& reassigned,
                       uint32_t& budget);

public:
    IvfIndex(const DocVectorAccess& vectors, DistanceFunctionFactory::UP distance_ff,
             vespalib::eval::CellType cell_type, size_t vector_size, search::attribute::DistanceMetric metric,
             const IvfIndexConfig& cfg);
    IvfIndex(const IvfIndex&) = delete;
    IvfIndex& operator=(const IvfIndex&) = delete;
    ~IvfIndex() override;

    const IvfIndexConfig& config() const noexcept { return _cfg; }

    // Implements NearestNeighborIndex
    void add_document(uint32_t docid) override;
    bool bulk_build(const std::vector<uint32_t>& lids, vespalib::Executor* executor) override;
    std::unique_ptr<PrepareResult> prepare_add_document(uint32_t docid, VectorBundle vectors,
                                                        vespalib::GenerationGuard read_guard) const override;
    void complete_add_document(uint32_t docid, std::unique_ptr<PrepareResult> prepare_result) override;
    void remove_document(uint32_t docid) override;
    void assign_generation(vespalib::Generation current_gen) override;
    void reclaim_memory(vespalib::Generation oldest_used_gen) override;
    vespalib::GenerationGuard make_generation_read_guard() const override;
    void inc_generation() override;
    void reclaim_unused_memory() override;
    bool consider_compact(const CompactionStrategy& compaction_strategy) override;
    vespalib::MemoryUsage update_stat(const CompactionStrategy& compaction_strategy) override;
    vespalib::MemoryUsage memory_usage() const override;
    void populate_address_space_usage(search::AddressSpaceUsage& usage) const override;
    std::unique_ptr<vespalib::StateExplorer> make_state_explorer() const override;
    void shrink_lid_space(uint32_t doc_id_limit) override;
    std::unique_ptr<NearestNeighborIndexSaver> make_saver(vespalib::GenericHeader& header) const override;
    std::unique_ptr<NearestNeighborIndexLoader> make_loader(FastOS_FileInterface&          file,
                                                            const vespalib::GenericHeader& header) override;
    std::vector<Neighbor> find_top_k(Stats& stats, uint32_t k, const BoundDistanceFunction& df, uint32_t explore_k,
                                     double exploration_slack, bool prefetch_tensors, const vespalib::Deadline& doom,
                                     double distance_threshold) const override;
    std::vector<Neighbor> find_top_k_with_filter(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                                 const GlobalFilter& filter, bool low_hit_ratio, double exploration,
                                                 uint32_t explore_k, double exploration_slack, bool prefetch_tensors,
                                                 const vespalib::Deadline& doom,
                                                 double                    distance_threshold) const override;
    DistanceFunctionFactory& distance_function_factory() const override { return *_distance_ff; }
    uint32_t check_consistency(uint32_t docid_limit) const noexcept override;

    // Search with an explicit nprobe (before filter scaling). Filter may be nullptr.
    std::vector<Neighbor> find_top_k_with_nprobe(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                                 const GlobalFilter* filter, uint32_t nprobe,
                                                 const vespalib::Deadline& doom, double distance_threshold,
                                                 IvfProbeStats& probe_stats) const;
    // As above, but centroid distances are computed from the query vector in the centroid cell type.
    std::vector<Neighbor> find_top_k_with_query(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                                TypedCells query, const GlobalFilter* filter, uint32_t nprobe,
                                                const vespalib::Deadline& doom, double distance_threshold,
                                                IvfProbeStats& probe_stats) const;
    // nprobe for a query asking for k hits with explore_k = k + explore additional hits.
    uint32_t nprobe_for_explore(uint32_t k, uint32_t explore_k) const noexcept;
    bool slot_is_live(uint32_t slot) const noexcept;
    size_t entry_bytes() const noexcept { return _entry_bytes; }
    // Writes back and drops the cached pages of the index file. Used to measure cold queries.
    void drop_file_cache() const;
    IvfIndexStats get_stats() const noexcept;
    void populate_state(vespalib::slime::Cursor& object) const;
};

} // namespace search::tensor
