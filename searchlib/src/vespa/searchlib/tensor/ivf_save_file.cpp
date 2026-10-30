// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "ivf_save_file.h"

#include <vespa/fastos/file.h>
#include <vespa/searchlib/util/bufferwriter.h>
#include <vespa/vespalib/util/exceptions.h>
#include <vespa/vespalib/util/stringfmt.h>

#include <xxhash.h>

#include <cinttypes>
#include <memory>

using vespalib::make_string;

namespace search::tensor::ivf {

namespace {

// File layout, all values in host byte order: magic, format version, complete flag, layout, file id, the counts,
// max squared norm, counters, rng state, refs, centroids, free slots, versions, lids, then the xxh3 checksum of
// everything before it.
constexpr uint64_t save_magic = 0x3145564153465649ULL; // "IVFSAVE1" read as little endian
constexpr uint32_t max_counters = 64;
constexpr uint32_t max_rng_bytes = 1u << 20;

struct HashState {
    struct Free {
        void operator()(XXH3_state_t* state) const noexcept { XXH3_freeState(state); }
    };
    std::unique_ptr<XXH3_state_t, Free> state;

    HashState() : state(XXH3_createState()) {
        if (!state || XXH3_64bits_reset(state.get()) != XXH_OK) {
            throw vespalib::IllegalStateException("IVF index: cannot initialize the save file checksum",
                                                  VESPA_STRLOC);
        }
    }
    void update(const void* data, size_t len) {
        if (XXH3_64bits_update(state.get(), data, len) != XXH_OK) {
            throw vespalib::IllegalStateException("IVF index: save file checksum update failed", VESPA_STRLOC);
        }
    }
    uint64_t digest() const noexcept { return XXH3_64bits_digest(state.get()); }
};

class HashingWriter {
    BufferWriter& _writer;
    HashState     _hash;

public:
    explicit HashingWriter(BufferWriter& writer) : _writer(writer), _hash() {}
    void write(const void* data, size_t len) {
        _writer.write(data, len);
        _hash.update(data, len);
    }
    template <typename T> void write_value(T value) { write(&value, sizeof(value)); }
    template <typename T> void write_vector(const std::vector<T>& values) {
        write(values.data(), values.size() * sizeof(T));
    }
    void finish() {
        uint64_t checksum = _hash.digest();
        _writer.write(&checksum, sizeof(checksum));
        _writer.flush();
    }
};

class CheckingReader {
    FastOS_FileInterface& _file;
    uint64_t              _remaining;
    HashState             _hash;
    std::string           _error;

public:
    explicit CheckingReader(FastOS_FileInterface& file) : _file(file), _remaining(0), _hash(), _error() {
        int64_t size = file.getSize();
        int64_t position = file.getPosition();
        if (size < 0 || position < 0 || position > size) {
            _error = "cannot get the file size";
        } else {
            _remaining = uint64_t(size - position);
        }
    }
    bool ok() const noexcept { return _error.empty(); }
    const std::string& error() const noexcept { return _error; }
    uint64_t remaining() const noexcept { return _remaining; }
    void fail(std::string reason) {
        if (_error.empty()) {
            _error = std::move(reason);
        }
    }
    // Each Read() either makes progress or ends the loop, so the loop runs at most len times.
    bool read_raw(void* data, size_t len) {
        if (!ok()) {
            return false;
        }
        if (len > _remaining) {
            fail("file is truncated");
            return false;
        }
        char*  dst = static_cast<char*>(data);
        size_t done = 0;
        while (done < len) {
            ssize_t got = _file.Read(dst + done, len - done);
            if (got <= 0) {
                fail((got == 0) ? "unexpected end of file" : "read error");
                return false;
            }
            done += size_t(got);
        }
        _remaining -= len;
        return true;
    }
    bool read(void* data, size_t len) {
        if (!read_raw(data, len)) {
            return false;
        }
        _hash.update(data, len);
        return true;
    }
    template <typename T> bool read_value(T& value) { return read(&value, sizeof(value)); }
    // Reads count elements, after checking that they fit in the rest of the file.
    template <typename T> bool read_vector(std::vector<T>& values, uint64_t count) {
        if (!ok()) {
            return false;
        }
        if (count > _remaining / sizeof(T)) {
            fail(make_string("%" PRIu64 " elements do not fit in the file", count));
            return false;
        }
        values.resize(count);
        return read(values.data(), count * sizeof(T));
    }
    bool check_digest() {
        uint64_t expected = _hash.digest();
        uint64_t stored = 0;
        if (!read_raw(&stored, sizeof(stored))) {
            return false;
        }
        if (stored != expected) {
            fail("checksum mismatch");
            return false;
        }
        if (_remaining != 0) {
            fail("trailing bytes after the checksum");
            return false;
        }
        return true;
    }
};

void write_layout(HashingWriter& out, const SaveLayout& layout) {
    out.write_value(layout.cell_type);
    out.write_value(layout.vector_size);
    out.write_value(layout.metric);
    out.write_value(layout.posting_capacity);
    out.write_value(layout.entry_bytes);
    out.write_value(layout.centroid_bytes);
    out.write_value(layout.size_classes_per_doubling);
    out.write_value(layout.min_posting_size);
}

bool read_layout(CheckingReader& in, SaveLayout& layout) {
    return in.read_value(layout.cell_type) && in.read_value(layout.vector_size) && in.read_value(layout.metric) &&
           in.read_value(layout.posting_capacity) && in.read_value(layout.entry_bytes) &&
           in.read_value(layout.centroid_bytes) && in.read_value(layout.size_classes_per_doubling) &&
           in.read_value(layout.min_posting_size);
}

} // namespace

SaveState::SaveState()
    : layout{0, 0, 0, 0, 0, 0, 0, 0},
      file_id(0),
      slot_limit(0),
      page_limit(0),
      gc_cursor(0),
      max_squared_norm(-1.0),
      counters(),
      rng_state(),
      refs(),
      centroids(),
      free_slots(),
      versions(),
      lids() {
}

SaveState::SaveState(SaveState&&) noexcept = default;
SaveState& SaveState::operator=(SaveState&&) noexcept = default;
SaveState::~SaveState() = default;

void write_save_file(BufferWriter& writer, const SaveState& state, const LidReader& lid_reader) {
    HashingWriter out(writer);
    out.write_value(save_magic);
    out.write_value(save_format_version);
    out.write_value(uint32_t(1));
    write_layout(out, state.layout);
    out.write_value(state.file_id);
    out.write_value(state.slot_limit);
    out.write_value(state.page_limit);
    out.write_value(state.gc_cursor);
    out.write_value(uint32_t(state.free_slots.size()));
    out.write_value(uint32_t(state.versions.size()));
    out.write_value(uint32_t(state.counters.size()));
    out.write_value(uint32_t(state.rng_state.size()));
    out.write_value(uint32_t(0));
    out.write_value(state.max_squared_norm);
    out.write_vector(state.counters);
    out.write(state.rng_state.data(), state.rng_state.size());
    out.write_vector(state.refs);
    out.write_vector(state.centroids);
    out.write_vector(state.free_slots);
    out.write_vector(state.versions);
    std::vector<uint32_t> lids;
    for (uint64_t ref : state.refs) {
        if (ref == 0) {
            continue;
        }
        lids.resize(posting_ref_length(ref));
        lid_reader(posting_ref_page(ref), posting_ref_length(ref), lids.data());
        out.write_vector(lids);
    }
    out.finish();
}

void write_incomplete_save_file(BufferWriter& writer) {
    HashingWriter out(writer);
    out.write_value(save_magic);
    out.write_value(save_format_version);
    out.write_value(uint32_t(0));
    out.finish();
}

std::string read_save_file(FastOS_FileInterface& file, SaveState& state) {
    CheckingReader in(file);
    uint64_t       magic = 0;
    uint32_t       version = 0;
    uint32_t       complete = 0;
    if (!in.read_value(magic) || !in.read_value(version) || !in.read_value(complete)) {
        return in.error();
    }
    if (magic != save_magic) {
        return "not an IVF index save file";
    }
    if (version != save_format_version) {
        return make_string("save file format %u, expected %u", version, save_format_version);
    }
    if (complete != 1) {
        return "the save did not complete (posting file sync failed)";
    }
    uint32_t free_slot_count = 0;
    uint32_t version_count = 0;
    uint32_t counter_count = 0;
    uint32_t rng_bytes = 0;
    uint32_t reserved = 0;
    if (!read_layout(in, state.layout) || !in.read_value(state.file_id) || !in.read_value(state.slot_limit) ||
        !in.read_value(state.page_limit) || !in.read_value(state.gc_cursor) || !in.read_value(free_slot_count) ||
        !in.read_value(version_count) || !in.read_value(counter_count) || !in.read_value(rng_bytes) ||
        !in.read_value(reserved) || !in.read_value(state.max_squared_norm))
    {
        return in.error();
    }
    if (counter_count > max_counters || rng_bytes > max_rng_bytes) {
        return make_string("bad header (%u counters, %u rng bytes)", counter_count, rng_bytes);
    }
    state.rng_state.resize(rng_bytes);
    if (!in.read_vector(state.counters, counter_count) || !in.read(state.rng_state.data(), rng_bytes) ||
        !in.read_vector(state.refs, state.slot_limit))
    {
        return in.error();
    }
    uint64_t live = 0;
    uint64_t entries = 0;
    for (uint64_t ref : state.refs) {
        if (ref != 0) {
            ++live;
            entries += posting_ref_length(ref);
        }
    }
    if (!in.read_vector(state.centroids, live * state.layout.centroid_bytes) ||
        !in.read_vector(state.free_slots, free_slot_count) || !in.read_vector(state.versions, version_count) ||
        !in.read_vector(state.lids, entries) || !in.check_digest())
    {
        return in.error();
    }
    return {};
}

} // namespace search::tensor::ivf
