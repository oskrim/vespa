// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class FastOS_FileInterface;

namespace search {
class BufferWriter;
}

namespace search::tensor::ivf {

constexpr uint32_t save_format_version = 1;

// A posting reference is (extent + 1) << 32 | length; 0 marks a free slot.
inline uint64_t make_posting_ref(uint32_t extent, uint32_t length) noexcept {
    return (static_cast<uint64_t>(extent + 1) << 32) | length;
}
inline uint32_t posting_ref_extent(uint64_t ref) noexcept {
    return static_cast<uint32_t>(ref >> 32) - 1;
}
inline uint32_t posting_ref_length(uint64_t ref) noexcept {
    return static_cast<uint32_t>(ref);
}

/**
 * Index properties that decide the posting file layout and the centroid encoding. A save file is only usable by an
 * index with the same layout.
 */
struct SaveLayout {
    uint32_t cell_type;
    uint32_t vector_size;
    uint32_t metric;
    uint32_t posting_capacity;
    uint32_t entry_bytes;
    uint32_t centroid_bytes;
    bool operator==(const SaveLayout&) const noexcept = default;
};

/**
 * The in-memory part of an IVF index as stored in the index save file of an attribute snapshot. Postings are
 * referenced by extent and length in the posting file whose header holds file_id.
 */
struct SaveState {
    SaveLayout            layout;
    uint64_t              file_id;
    uint32_t              slot_limit;
    uint32_t              extent_limit;
    uint32_t              gc_cursor;
    double                max_squared_norm; // negative when the metric keeps no norm
    std::vector<uint64_t> counters;
    std::string           rng_state;
    std::vector<uint64_t> refs;       // one posting reference per slot below slot_limit
    std::vector<char>     centroids;  // one centroid per live slot, in slot order
    std::vector<uint32_t> free_slots; // in allocation order: the last one is used first
    std::vector<uint32_t> versions;   // current version per docid
    std::vector<uint32_t> lids;       // docid of every entry of every live posting, in slot order

    SaveState();
    SaveState(SaveState&&) noexcept;
    SaveState& operator=(SaveState&&) noexcept;
    ~SaveState();
};

// Fills lids with the docids of the first length entries of an extent.
using LidReader = std::function<void(uint32_t extent, uint32_t length, uint32_t* lids)>;

// Writes a complete save file; the lids of live postings are fetched through lid_reader. Flushes the writer.
void write_save_file(BufferWriter& writer, const SaveState& state, const LidReader& lid_reader);

// Writes a save file that every reader rejects. Flushes the writer.
void write_incomplete_save_file(BufferWriter& writer);

// Reads a save file positioned after its generic header. Returns an empty string when the file is complete and its
// checksum matches, otherwise the reason why it cannot be used; state is then undefined.
std::string read_save_file(FastOS_FileInterface& file, SaveState& state);

} // namespace search::tensor::ivf
