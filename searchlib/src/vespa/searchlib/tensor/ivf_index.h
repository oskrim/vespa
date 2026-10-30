// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "distance_function_factory.h"
#include "doc_vector_access.h"
#include "ivf_kmeans.h"
#include "ivf_page_allocator.h"
#include "ivf_save_file.h"
#include "nearest_neighbor_index.h"
#include "prepare_result.h"
#include "vector_bundle.h"

#include <vespa/eval/eval/cell_type.h>
#include <vespa/eval/eval/typed_cells.h>
#include <vespa/searchcommon/attribute/ivf_index_params.h>
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

    uint32_t           posting_capacity;
    uint32_t           nprobe;
    uint32_t           reassign_range;
    uint32_t           max_postings;
    uint32_t           max_rebalance_steps;
    IvfSplitMode       split_mode;
    uint32_t           split_iters;
    double             split_min_side;
    uint32_t           replicas;
    uint32_t           replica_candidates;
    double             replica_eps;
    double             rng_factor;
    bool               bulk_build;
    double             bulk_fill;
    uint32_t           bulk_sample;
    uint32_t           bulk_iters;
    double             prune_eps;
    double             filter_min_ratio;
    double             filter_max_scale;
    uint32_t           bulk_coarse_probes;
    uint32_t           gc_steps;
    double             gc_stale_ratio;
    double             merge_fraction;
    double             split_live_fraction;
    uint32_t           bulk_coarse_k;
    bool               prefetch;
    uint32_t           file_grow_extents;
    double             file_grow_fraction;
    uint32_t           medoid_sample;
    uint32_t           medoid_rounds;
    IvfFilterProbeMode filter_probe_mode;
    uint32_t           filter_max_examined;
    double             filter_entry_read_density;
    IvfAdaptiveStop    filter_adaptive_stop;
    double             filter_exact_ratio;
    double             k_entry_factor;
    bool               rebuild_on_load;
    double             flush_pinned_ratio;
    uint32_t           size_classes_per_doubling;
    uint32_t           min_posting_size;
    double             posting_headroom;
    double             bulk_headroom;

    IvfIndexConfig();

    // Knob names without the VESPA_IVF_ prefix, e.g. "POSTING_CAPACITY".
    static const std::vector<std::string>& knob_names();

    // Reads every knob through lookup (nullptr = default). Bad or out-of-range values log a warning and keep the
    // default.
    static IvfIndexConfig from_lookup(const Lookup& lookup);

    // Config from the schema's 'ivf' index block: knob names in kebab case (posting-capacity). Unknown names log a
    // warning.
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
    uint64_t pages;      // page limit: pages below it are allocated, free, pinned or on hold
    uint64_t free_pages; // in free ranges below the page limit
    uint64_t free_ranges;
    uint64_t file_bytes;
    uint64_t bytes_written;
    uint64_t bulk_builds;
    uint64_t gc_checks;
    uint64_t gc_compactions;
    uint64_t merges;
    uint64_t pinned_pages; // freed after the last init-save, or pinned by a save that is not valid yet
    uint64_t growth_copies; // full postings copied to a larger size class
};

enum class IvfLoadSource { NONE, SAVE_FILE, BULK_BUILD };

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
 * RAM holds centroids, one atomic posting reference per centroid and a version per docid. Each posting is a range of
 * whole pages in the posting file, sized by a size class with some free room, so appends write in place until the
 * posting is full and it is copied to a larger class. Entries are {docid, version, cells}, and an entry is live only
 * while its version equals the docid's current version. A vector can be stored in several postings (SPANN replicas).
 * Full postings are compacted or split in two, and vectors near a split are reassigned (SPFresh LIRE).
 *
 * A lid table in RAM mirrors the docid of every entry slot of every posting, so filtered searches can count matches
 * per posting before reading it.
 *
 * Supports 1 writer thread and multiple reader threads without mutexes. Retired page ranges and centroid slots are
 * reused only after a generation hold cycle.
 *
 * The posting file is named after the attribute and lives in the attribute's directory. An attribute flush saves the
 * in-memory state (centroids, posting references, lid tables, versions) to the snapshot's index save file, which
 * references the posting file by its file id; loading it, then replaying the transaction log, restores the index
 * without a rebuild. When the file location is unknown or the file is in use, the index uses an unlinked file, is not
 * saved, and is rebuilt from the attribute on load.
 */
class IvfIndex : public NearestNeighborIndex {
public:
    static constexpr uint32_t no_slot = std::numeric_limits<uint32_t>::max();

private:
    using TypedCells = vespalib::eval::TypedCells;
    using AtomicVersion = vespalib::datastore::AtomicValueWrapper<uint32_t>;
    using VersionVector = vespalib::TypeStableVector<AtomicVersion>;
    using IdHoldList = vespalib::GenerationHoldList<uint32_t, false, true>;
    using RangeHoldList = vespalib::GenerationHoldList<uint64_t, false, true>;
    using SlotDistances = std::vector<std::pair<double, uint32_t>>;

    struct Posting {
        uint32_t page;
        uint32_t size_class;
        uint32_t length;
    };

    // Page ranges (range_key) freed before the snapshot with this serial number was saved; reusable once that
    // snapshot is valid.
    struct PinnedRanges {
        uint64_t              serial_num;
        std::vector<uint64_t> ranges;
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
        std::atomic<uint64_t> pages{0};
        std::atomic<uint64_t> free_pages{0};
        std::atomic<uint64_t> free_ranges{0};
        std::atomic<uint64_t> file_bytes{0};
        std::atomic<uint64_t> bytes_written{0};
        std::atomic<uint64_t> bulk_builds{0};
        std::atomic<uint64_t> gc_checks{0};
        std::atomic<uint64_t> gc_compactions{0};
        std::atomic<uint64_t> merges{0};
        std::atomic<uint64_t> pinned_pages{0};
        std::atomic<uint64_t> growth_copies{0};
    };

    const DocVectorAccess&                   _vectors;
    std::unique_ptr<DistanceFunctionFactory> _distance_ff;
    IvfIndexConfig                           _cfg;
    vespalib::eval::CellType                 _cell_type;
    size_t                                   _vector_size;
    size_t                                   _vector_bytes;
    size_t                                   _entry_bytes;
    size_t                                   _centroid_bytes;
    ivf::SizeClasses                         _classes;
    ivf::VectorSpace                         _space;
    std::string                              _posting_path; // empty for an unlinked file
    bool                                     _persistent;
    int                                      _fd;
    bool                                     _file_ready;
    uint64_t                                 _file_id;
    uint64_t                                 _disk_file_id; // from the header found when the file was opened

    void*                 _slot_table;
    size_t                _slot_table_bytes;
    uint64_t*             _slot_refs;
    char*                 _centroids;
    std::atomic<uint32_t> _slot_limit;
    uint32_t*             _lids; // indexed by _classes.first_lid_slot(page) + entry index
    size_t                _lid_table_bytes;
    uint32_t              _max_pages;

    VersionVector         _versions;
    std::atomic<uint32_t> _version_limit;

    vespalib::GenerationHandler _generation_handler;
    IdHoldList                  _slot_hold;
    RangeHoldList               _range_hold;
    uint64_t                    _held_pages;
    std::vector<uint32_t>       _free_slots;
    ivf::PageAllocator          _alloc;
    std::vector<uint64_t>       _pending_ranges;
    std::vector<PinnedRanges>   _pinned_ranges;
    std::atomic<uint64_t>       _valid_serial; // newest snapshot marked valid, set by the flush thread
    uint32_t                    _file_pages;
    uint32_t                    _gc_cursor;

    std::deque<ReassignTask>   _reassign_queue;
    std::vector<char>          _entry_buf;
    std::mt19937               _rng;
    Counters                   _counters;
    mutable std::atomic<bool>  _fadvise_warned;
    std::atomic<IvfLoadSource> _load_source;
    std::atomic<double>        _load_seconds;
    std::string                _load_rejected; // reason the last save file was not used; written before readers exist

    // Reader safe accessors.
    static uint64_t make_ref(Posting posting) noexcept {
        return ivf::make_posting_ref(posting.page, posting.size_class, posting.length);
    }
    static Posting decode_ref(uint64_t ref) noexcept {
        return {ivf::posting_ref_page(ref), ivf::posting_ref_class(ref), ivf::posting_ref_length(ref)};
    }
    static uint64_t range_key(uint32_t page, uint32_t pages) noexcept { return (uint64_t(page) << 32) | pages; }
    uint64_t page_offset(uint32_t page) const noexcept;
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
    uint32_t lid_at(uint32_t page, uint32_t index) const noexcept;
    void set_lids(uint32_t page, uint32_t first, const char* entries, uint32_t count) noexcept;
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

    using CounterField = std::atomic<uint64_t> Counters::*;
    static const std::vector<CounterField>& saved_counters();

    // Writer only.
    void open_posting_file();
    void grow_file(uint64_t new_file_pages);
    void start_new_file();
    void free_range(uint64_t key);
    void release_range(uint64_t key);
    void pin_ranges_for_save(uint64_t serial_num);
    void release_pinned_ranges();
    void update_page_counters() noexcept;
    ivf::SaveLayout layout() const noexcept;
    ivf::SaveState capture_save_state() const;
    std::string check_save_state(const ivf::SaveState& state) const;
    void apply_save_state(const ivf::SaveState& state);
    void load_save_file(FastOS_FileInterface& file);
    void check_cells(TypedCells cells) const;
    void encode_entry(char* dst, uint32_t docid, uint32_t version, TypedCells cells) const noexcept;
    uint32_t bump_version(uint32_t docid);
    uint32_t allocate_pages(uint32_t pages);
    void retire_posting(Posting posting);
    uint32_t slots_available() const noexcept;
    uint32_t create_posting(const char* entries, uint32_t count, TypedCells centroid_cells, double headroom);
    void retire_slot(uint32_t slot);
    void drop_entries(uint32_t docid, uint32_t count);
    Posting write_new_posting(const char* entries, uint32_t count, double headroom);
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
    // An empty posting_path gives an unlinked posting file and an index that is never saved.
    IvfIndex(const DocVectorAccess& vectors, DistanceFunctionFactory::UP distance_ff,
             vespalib::eval::CellType cell_type, size_t vector_size, search::attribute::DistanceMetric metric,
             const IvfIndexConfig& cfg, const std::string& posting_path);
    IvfIndex(const IvfIndex&) = delete;
    IvfIndex& operator=(const IvfIndex&) = delete;
    ~IvfIndex() override;

    const IvfIndexConfig& config() const noexcept { return _cfg; }

    // The posting file of an attribute with the given base file name: '<attribute name>.ivfpostings' in the
    // attribute's directory, which is the parent of a 'snapshot-<serial>' directory. Empty when the base file name
    // has no directory.
    static std::string posting_path_for(const std::string& base_file_name);
    // Removes the posting file that an earlier IVF index of the attribute left behind, if any.
    static void remove_posting_file(const std::string& base_file_name);
    // The generic header tag that marks an index save file written by this index.
    static const std::string& save_file_tag();

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
    std::unique_ptr<NearestNeighborIndexSaver> make_saver(vespalib::GenericHeader& header,
                                                          uint64_t                 serial_num) override;
    void on_snapshot_valid(uint64_t serial_num) noexcept override;
    uint64_t flush_disk_gain() const noexcept override;
    std::unique_ptr<NearestNeighborIndexLoader> make_loader(FastOS_FileInterface&          file,
                                                            const vespalib::GenericHeader& header) override;
    bool needs_build_after_load() const noexcept override;
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
    bool persistent() const noexcept { return _persistent; }
    const std::string& posting_path() const noexcept { return _posting_path; }
    IvfLoadSource load_source() const noexcept { return _load_source.load(std::memory_order_relaxed); }
    // File byte ranges (offset, bytes) of the live postings. Writer only; used by tests.
    std::vector<std::pair<uint64_t, uint64_t>> live_posting_ranges() const;
    // (length, live entries) of every live posting; reads each posting from the file. Writer only; used by the bench.
    std::vector<std::pair<uint32_t, uint32_t>> posting_fill() const;
    // Empty when live postings, free, pending, pinned and held page ranges exactly cover the pages below the limit
    // without overlap; else the first problem found. Writer only; used by tests.
    std::string check_storage() const;
};

} // namespace search::tensor
