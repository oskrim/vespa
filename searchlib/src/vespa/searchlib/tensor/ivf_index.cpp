// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "ivf_index.h"

#include "bound_distance_function.h"
#include "mips_distance_transform.h"
#include "nearest_neighbor_index_loader.h"
#include "nearest_neighbor_index_saver.h"

#include <vespa/eval/eval/value_type_spec.h>
#include <vespa/fastos/file.h>
#include <vespa/searchlib/queryeval/global_filter.h>
#include <vespa/searchlib/util/state_explorer_utils.h>
#include <vespa/vespalib/data/fileheader.h>
#include <vespa/vespalib/data/slime/cursor.h>
#include <vespa/vespalib/data/slime/inserter.h>
#include <vespa/vespalib/net/http/state_explorer.h>
#include <vespa/vespalib/util/deadline.h>
#include <vespa/vespalib/util/exceptions.h>
#include <vespa/vespalib/util/stringfmt.h>

#include <vespa/vespalib/util/generation_hold_list.hpp>
#include <vespa/vespalib/util/type_stable_vector.hpp>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <vespa/log/log.h>
LOG_SETUP(".searchlib.tensor.ivf_index");

using search::attribute::DistanceMetric;
using vespalib::Generation;
using vespalib::GenerationGuard;
using vespalib::GenericHeader;
using vespalib::IllegalArgumentException;
using vespalib::IllegalStateException;
using vespalib::make_string;
using vespalib::eval::CellType;
using vespalib::eval::CellTypeUtils;
using vespalib::eval::value_type::cell_type_to_name;

namespace search::tensor {

namespace {

constexpr size_t   entry_header_bytes = 2 * sizeof(uint32_t);
constexpr size_t   ref_bytes = sizeof(uint64_t);
constexpr uint32_t max_io_interrupts = 100;

// The posting file starts with a header that holds a random file id, which a save file must match. Extents follow.
constexpr size_t   file_header_bytes = 4096;
constexpr uint64_t posting_file_magic = 0x54534f5046564956ULL; // "VIVFPOST" read as little endian
constexpr uint32_t posting_file_version = 1;
constexpr size_t   file_id_offset = 16;
const std::string  posting_file_suffix(".ivfpostings");

size_t align_up_8(size_t value) noexcept {
    return (value + 7) & ~size_t(7);
}

void add_to(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
}

void sub_from(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
    uint64_t old_value = counter.load(std::memory_order_relaxed);
    counter.store(old_value >= value ? old_value - value : 0, std::memory_order_relaxed);
}

void set_to(std::atomic<uint64_t>& counter, uint64_t value) noexcept {
    counter.store(value, std::memory_order_relaxed);
}

uint64_t get(const std::atomic<uint64_t>& counter) noexcept {
    return counter.load(std::memory_order_relaxed);
}

bool parse_u32(const char* text, uint32_t& out) {
    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_double(const char* text, double& out) {
    char* end = nullptr;
    errno = 0;
    double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

const char* split_mode_name(IvfSplitMode mode) {
    return (mode == IvfSplitMode::KMEANS2) ? "kmeans2" : "medoid_balanced";
}

const char* filter_probe_mode_name(IvfFilterProbeMode mode) {
    return (mode == IvfFilterProbeMode::SCALE) ? "scale" : "adaptive";
}

const char* adaptive_stop_name(IvfAdaptiveStop stop) {
    return (stop == IvfAdaptiveStop::POSTINGS) ? "postings" : "entries";
}

class KnobReader {
    const IvfIndexConfig::Lookup& _lookup;

    const char* text(const char* knob) const {
        const char* value = _lookup(knob);
        return (value != nullptr && *value != '\0') ? value : nullptr;
    }
    static void warn(const char* knob, const char* value) {
        LOG(warning, "IVF index: ignoring invalid value '%2$s' of %1$s, using the default", knob, value);
    }

public:
    explicit KnobReader(const IvfIndexConfig::Lookup& lookup) : _lookup(lookup) {}
    void read(const char* knob, uint32_t& value) const {
        const char* value_text = text(knob);
        if (value_text != nullptr && !parse_u32(value_text, value)) {
            warn(knob, value_text);
        }
    }
    void read(const char* knob, double& value) const {
        const char* value_text = text(knob);
        if (value_text != nullptr && !parse_double(value_text, value)) {
            warn(knob, value_text);
        }
    }
    void read(const char* knob, bool& value) const {
        const char* value_text = text(knob);
        uint32_t    parsed = 0;
        if (value_text == nullptr) {
            return;
        }
        if (std::strcmp(value_text, "true") == 0 || std::strcmp(value_text, "false") == 0) {
            value = (value_text[0] == 't');
        } else if (parse_u32(value_text, parsed) && parsed <= 1) {
            value = (parsed == 1);
        } else {
            warn(knob, value_text);
        }
    }
    void read(const char* knob, IvfSplitMode& value) const {
        const char* value_text = text(knob);
        if (value_text == nullptr) {
            return;
        }
        if (std::strcmp(value_text, "kmeans2") == 0) {
            value = IvfSplitMode::KMEANS2;
        } else if (std::strcmp(value_text, "medoid_balanced") == 0) {
            value = IvfSplitMode::MEDOID_BALANCED;
        } else {
            warn(knob, value_text);
        }
    }
    void read(const char* knob, IvfFilterProbeMode& value) const {
        const char* value_text = text(knob);
        if (value_text == nullptr) {
            return;
        }
        if (std::strcmp(value_text, "scale") == 0) {
            value = IvfFilterProbeMode::SCALE;
        } else if (std::strcmp(value_text, "adaptive") == 0) {
            value = IvfFilterProbeMode::ADAPTIVE;
        } else {
            warn(knob, value_text);
        }
    }
    void read(const char* knob, IvfAdaptiveStop& value) const {
        const char* value_text = text(knob);
        if (value_text == nullptr) {
            return;
        }
        if (std::strcmp(value_text, "postings") == 0) {
            value = IvfAdaptiveStop::POSTINGS;
        } else if (std::strcmp(value_text, "entries") == 0) {
            value = IvfAdaptiveStop::ENTRIES;
        } else {
            warn(knob, value_text);
        }
    }
};

template <typename T> void check_range(const char* knob, T& value, T default_value, T lo, T hi) {
    if (!(value >= lo && value <= hi)) {
        LOG(warning, "IVF index: %s=%s is outside [%s, %s], using %s", knob, std::to_string(value).c_str(),
            std::to_string(lo).c_str(), std::to_string(hi).c_str(), std::to_string(default_value).c_str());
        value = default_value;
    }
}

int open_unlinked_file(const std::string& dir) {
    int fd = ::open(dir.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
    if (fd >= 0) {
        return fd;
    }
    int tmpfile_errno = errno;
    if (tmpfile_errno != EOPNOTSUPP && tmpfile_errno != EISDIR && tmpfile_errno != EINVAL) {
        throw IllegalStateException(
            make_string("IVF index: open(%s, O_TMPFILE) failed: %s", dir.c_str(), std::strerror(tmpfile_errno)),
            VESPA_STRLOC);
    }
    std::string       name_template = dir + "/ivf-index-XXXXXX";
    std::vector<char> name(name_template.begin(), name_template.end());
    name.push_back('\0');
    fd = ::mkostemp(name.data(), O_CLOEXEC);
    if (fd < 0) {
        throw IllegalStateException(
            make_string("IVF index: mkostemp(%s) failed: %s", name.data(), std::strerror(errno)), VESPA_STRLOC);
    }
    if (::unlink(name.data()) != 0) {
        int unlink_errno = errno;
        ::close(fd);
        throw IllegalStateException(
            make_string("IVF index: unlink(%s) failed: %s", name.data(), std::strerror(unlink_errno)), VESPA_STRLOC);
    }
    return fd;
}

void pread_fully(int fd, char* buf, size_t len, uint64_t offset) {
    size_t   done = 0;
    uint32_t interrupts = 0;
    while (done < len) {
        ssize_t got = ::pread(fd, buf + done, len - done, offset + done);
        if (got > 0) {
            done += got;
        } else if (got < 0 && errno == EINTR && ++interrupts < max_io_interrupts) {
            continue;
        } else {
            const char* reason = (got == 0) ? "unexpected end of file" : std::strerror(errno);
            throw IllegalStateException(
                make_string("IVF index: pread(%zu bytes at %" PRIu64 ") failed: %s", len, offset, reason),
                VESPA_STRLOC);
        }
    }
}

void pwrite_fully(int fd, const char* buf, size_t len, uint64_t offset) {
    size_t   done = 0;
    uint32_t interrupts = 0;
    while (done < len) {
        ssize_t put = ::pwrite(fd, buf + done, len - done, offset + done);
        if (put > 0) {
            done += put;
        } else if (put < 0 && errno == EINTR && ++interrupts < max_io_interrupts) {
            continue;
        } else {
            const char* reason = (put == 0) ? "no progress" : std::strerror(errno);
            throw IllegalStateException(
                make_string("IVF index: pwrite(%zu bytes at %" PRIu64 ") failed: %s", len, offset, reason),
                VESPA_STRLOC);
        }
    }
}

std::string unlinked_file_dir() {
    std::error_code ec;
    const char*     home = std::getenv("VESPA_HOME");
    if (home != nullptr && *home != '\0') {
        std::string dir = std::string(home) + "/var/tmp";
        if (std::filesystem::is_directory(dir, ec)) {
            return dir;
        }
    }
    auto tmp = std::filesystem::temp_directory_path(ec);
    return ec ? std::string("/tmp") : tmp.string();
}

void sync_directory(const std::string& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw IllegalStateException(make_string("IVF index: open(%s) failed: %s", dir.c_str(), std::strerror(errno)),
                                    VESPA_STRLOC);
    }
    int rc = ::fsync(fd);
    int sync_errno = errno;
    if (::close(fd) != 0 && rc == 0) {
        rc = -1;
        sync_errno = errno;
    }
    if (rc != 0) {
        throw IllegalStateException(
            make_string("IVF index: syncing directory %s failed: %s", dir.c_str(), std::strerror(sync_errno)),
            VESPA_STRLOC);
    }
}

uint64_t file_size_of(int fd) {
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        throw IllegalStateException(make_string("IVF index: fstat() failed: %s", std::strerror(errno)), VESPA_STRLOC);
    }
    return uint64_t(st.st_size);
}

// The file id in the posting file header, or 0 when the file has no valid header.
uint64_t read_file_id(int fd) {
    std::vector<char> header(file_id_offset + sizeof(uint64_t));
    ssize_t           got = -1;
    for (uint32_t interrupts = 0; interrupts < max_io_interrupts; ++interrupts) {
        got = ::pread(fd, header.data(), header.size(), 0);
        if (got >= 0 || errno != EINTR) {
            break;
        }
    }
    if (got < 0) {
        throw IllegalStateException(make_string("IVF index: reading the posting file header failed: %s",
                                                std::strerror(errno)),
                                    VESPA_STRLOC);
    }
    uint64_t magic = 0;
    uint32_t version = 0;
    uint64_t file_id = 0;
    if (size_t(got) < header.size()) {
        return 0;
    }
    std::memcpy(&magic, header.data(), sizeof(magic));
    std::memcpy(&version, header.data() + sizeof(magic), sizeof(version));
    std::memcpy(&file_id, header.data() + file_id_offset, sizeof(file_id));
    return (magic == posting_file_magic && version == posting_file_version) ? file_id : 0;
}

uint64_t new_file_id() {
    std::random_device device;
    uint64_t           id = (uint64_t(device()) << 32) ^ device();
    return (id != 0) ? id : 1;
}

const char* load_source_name(IvfLoadSource source) {
    switch (source) {
    case IvfLoadSource::SAVE_FILE:
        return "save_file";
    case IvfLoadSource::BULK_BUILD:
        return "bulk_build";
    case IvfLoadSource::NONE:
        break;
    }
    return "none";
}

void validate_config(size_t vector_size) {
    if (vector_size == 0) {
        throw IllegalArgumentException("IVF index: vector size must be positive", VESPA_STRLOC);
    }
}

IvfIndexConfig sanitized(const IvfIndexConfig& cfg) {
    IvfIndexConfig result(cfg);
    result.sanitize();
    return result;
}

using Neighbor = NearestNeighborIndex::Neighbor;

// Keeps the minimum distance per docid, then the k nearest docids, sorted by docid.
std::vector<Neighbor> select_top_k(std::vector<Neighbor>& candidates, uint32_t k, double distance_threshold) {
    std::sort(candidates.begin(), candidates.end(), [](const Neighbor& a, const Neighbor& b) {
        return (a.docid < b.docid) || ((a.docid == b.docid) && (a.distance < b.distance));
    });
    auto unique_end = std::unique(candidates.begin(), candidates.end(),
                                  [](const Neighbor& a, const Neighbor& b) { return a.docid == b.docid; });
    candidates.erase(unique_end, candidates.end());
    auto by_distance = [](const Neighbor& a, const Neighbor& b) {
        return (a.distance < b.distance) || ((a.distance == b.distance) && (a.docid < b.docid));
    };
    if (candidates.size() > k) {
        std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(), by_distance);
        candidates.resize(k);
    }
    std::vector<Neighbor> result;
    result.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (!(candidate.distance > distance_threshold)) {
            result.push_back(candidate);
        }
    }
    std::sort(result.begin(), result.end(), [](const Neighbor& a, const Neighbor& b) { return a.docid < b.docid; });
    return result;
}

class IvfIndexExplorer : public vespalib::StateExplorer {
    const IvfIndex& _index;

public:
    explicit IvfIndexExplorer(const IvfIndex& index) : _index(index) {}
    void get_state(const vespalib::slime::Inserter& inserter, bool full) const override {
        (void)full;
        _index.populate_state(inserter.insertObject());
    }
};

// Writes the state captured at init-save time. Lids of the saved postings are read from the live lid table on the
// flush thread; the extent pinning rule keeps them unchanged until a newer save is durable.
class IvfIndexSaver : public NearestNeighborIndexSaver {
    ivf::SaveState         _state;
    ivf::LidReader         _lid_reader;
    int                    _fd;
    std::string            _posting_path;
    std::atomic<uint64_t>& _saves_completed;
    uint64_t               _save_seq;

public:
    IvfIndexSaver(ivf::SaveState state, ivf::LidReader lid_reader, int fd, std::string posting_path,
                  std::atomic<uint64_t>& saves_completed, uint64_t save_seq)
        : _state(std::move(state)),
          _lid_reader(std::move(lid_reader)),
          _fd(fd),
          _posting_path(std::move(posting_path)),
          _saves_completed(saves_completed),
          _save_seq(save_seq) {}
    ~IvfIndexSaver() override;
    void save(BufferWriter& writer) const override {
        if (::fdatasync(_fd) != 0) {
            LOG(error, "IVF index: fdatasync(%s) failed: %s; the index save file is written as incomplete",
                _posting_path.c_str(), std::strerror(errno));
            ivf::write_incomplete_save_file(writer);
            return;
        }
        ivf::write_save_file(writer, _state, _lid_reader);
        // Saves of one attribute run one at a time, so this loop only retries after a spurious failure.
        uint64_t done = _saves_completed.load(std::memory_order_acquire);
        while (done < _save_seq &&
               !_saves_completed.compare_exchange_weak(done, _save_seq, std::memory_order_acq_rel)) {
        }
    }
};

IvfIndexSaver::~IvfIndexSaver() = default;

class IvfIndexLoader : public NearestNeighborIndexLoader {
    std::function<void()> _load;
    bool                  _done;

public:
    explicit IvfIndexLoader(std::function<void()> load) : _load(std::move(load)), _done(false) {}
    ~IvfIndexLoader() override;
    bool load_next() override {
        if (!_done) {
            _done = true;
            _load();
        }
        return false;
    }
};

IvfIndexLoader::~IvfIndexLoader() = default;

} // namespace

IvfIndexConfig::IvfIndexConfig()
    : posting_capacity(256),
      nprobe(32),
      reassign_range(0),
      max_postings(1u << 20),
      max_rebalance_steps(1000),
      split_mode(IvfSplitMode::KMEANS2),
      split_iters(5),
      split_min_side(0.1),
      replicas(1),
      replica_candidates(64),
      replica_eps(0.0),
      rng_factor(1.0),
      bulk_build(true),
      bulk_fill(0.5),
      bulk_sample(256 * 1024),
      bulk_iters(8),
      prune_eps(0.0),
      filter_min_ratio(0.1),
      filter_max_scale(16.0),
      bulk_coarse_probes(4),
      gc_steps(16),
      gc_stale_ratio(0.25),
      merge_fraction(0.125),
      split_live_fraction(0.5),
      bulk_coarse_k(0),
      prefetch(true),
      file_grow_extents(16),
      file_grow_fraction(0.25),
      medoid_sample(64),
      medoid_rounds(3),
      filter_probe_mode(IvfFilterProbeMode::SCALE),
      filter_max_examined(4096),
      filter_entry_read_density(0.0),
      filter_adaptive_stop(IvfAdaptiveStop::ENTRIES),
      filter_exact_ratio(0.0),
      k_entry_factor(4.0),
      rebuild_on_load(false) {
}

const std::vector<std::string>& IvfIndexConfig::knob_names() {
    static const std::vector<std::string> names = {
        "POSTING_CAPACITY",  "NPROBE",      "REASSIGN_RANGE",   "MAX_POSTINGS",     "MAX_REBALANCE_STEPS",
        "SPLIT_MODE",        "SPLIT_ITERS", "SPLIT_MIN_SIDE",   "REPLICAS",         "REPLICA_CANDIDATES",
        "REPLICA_EPS",       "RNG_FACTOR",  "BULK_BUILD",       "BULK_FILL",        "BULK_SAMPLE",
        "BULK_ITERS",        "PRUNE_EPS",   "FILTER_MIN_RATIO", "FILTER_MAX_SCALE",
        "BULK_COARSE_PROBES", "GC_STEPS",    "GC_STALE_RATIO",   "MERGE_FRACTION",   "SPLIT_LIVE_FRACTION",
        "BULK_COARSE_K",     "PREFETCH",    "FILE_GROW_EXTENTS", "FILE_GROW_FRACTION", "MEDOID_SAMPLE",
        "MEDOID_ROUNDS",     "FILTER_PROBE_MODE", "FILTER_MAX_EXAMINED", "FILTER_ENTRY_READ_DENSITY",
        "FILTER_ADAPTIVE_STOP", "FILTER_EXACT_RATIO", "K_ENTRY_FACTOR", "REBUILD_ON_LOAD"};
    return names;
}

IvfIndexConfig IvfIndexConfig::from_lookup(const Lookup& lookup) {
    IvfIndexConfig cfg;
    KnobReader     reader(lookup);
    reader.read("POSTING_CAPACITY", cfg.posting_capacity);
    reader.read("NPROBE", cfg.nprobe);
    reader.read("REASSIGN_RANGE", cfg.reassign_range);
    reader.read("MAX_POSTINGS", cfg.max_postings);
    reader.read("MAX_REBALANCE_STEPS", cfg.max_rebalance_steps);
    reader.read("SPLIT_MODE", cfg.split_mode);
    reader.read("SPLIT_ITERS", cfg.split_iters);
    reader.read("SPLIT_MIN_SIDE", cfg.split_min_side);
    reader.read("REPLICAS", cfg.replicas);
    reader.read("REPLICA_CANDIDATES", cfg.replica_candidates);
    reader.read("REPLICA_EPS", cfg.replica_eps);
    reader.read("RNG_FACTOR", cfg.rng_factor);
    reader.read("BULK_BUILD", cfg.bulk_build);
    reader.read("BULK_FILL", cfg.bulk_fill);
    reader.read("BULK_SAMPLE", cfg.bulk_sample);
    reader.read("BULK_ITERS", cfg.bulk_iters);
    reader.read("PRUNE_EPS", cfg.prune_eps);
    reader.read("FILTER_MIN_RATIO", cfg.filter_min_ratio);
    reader.read("FILTER_MAX_SCALE", cfg.filter_max_scale);
    reader.read("BULK_COARSE_PROBES", cfg.bulk_coarse_probes);
    reader.read("GC_STEPS", cfg.gc_steps);
    reader.read("GC_STALE_RATIO", cfg.gc_stale_ratio);
    reader.read("MERGE_FRACTION", cfg.merge_fraction);
    reader.read("SPLIT_LIVE_FRACTION", cfg.split_live_fraction);
    reader.read("BULK_COARSE_K", cfg.bulk_coarse_k);
    reader.read("PREFETCH", cfg.prefetch);
    reader.read("FILE_GROW_EXTENTS", cfg.file_grow_extents);
    reader.read("FILE_GROW_FRACTION", cfg.file_grow_fraction);
    reader.read("MEDOID_SAMPLE", cfg.medoid_sample);
    reader.read("MEDOID_ROUNDS", cfg.medoid_rounds);
    reader.read("FILTER_PROBE_MODE", cfg.filter_probe_mode);
    reader.read("FILTER_MAX_EXAMINED", cfg.filter_max_examined);
    reader.read("FILTER_ENTRY_READ_DENSITY", cfg.filter_entry_read_density);
    reader.read("FILTER_ADAPTIVE_STOP", cfg.filter_adaptive_stop);
    reader.read("FILTER_EXACT_RATIO", cfg.filter_exact_ratio);
    reader.read("K_ENTRY_FACTOR", cfg.k_entry_factor);
    reader.read("REBUILD_ON_LOAD", cfg.rebuild_on_load);
    cfg.sanitize();
    return cfg;
}

IvfIndexConfig IvfIndexConfig::from_params(const search::attribute::IvfIndexParams& params) {
    auto kebab = [](const std::string& knob) {
        std::string name(knob);
        std::transform(name.begin(), name.end(), name.begin(),
                       [](char c) { return c == '_' ? '-' : static_cast<char>(std::tolower(c)); });
        return name;
    };
    for (const auto& [name, value] : params.params()) {
        const auto& knobs = knob_names();
        if (std::none_of(knobs.begin(), knobs.end(), [&](const std::string& knob) { return kebab(knob) == name; })) {
            LOG(warning, "IVF index: ignoring unknown parameter '%s'", name.c_str());
        }
    }
    return from_lookup([&params, &kebab](const char* knob) -> const char* {
        std::string name = kebab(knob);
        for (const auto& [param_name, value] : params.params()) {
            if (param_name == name) {
                return value.c_str();
            }
        }
        return nullptr;
    });
}

void IvfIndexConfig::sanitize() {
    const IvfIndexConfig def;
    check_range("POSTING_CAPACITY", posting_capacity, def.posting_capacity, 4u, 65536u);
    check_range("NPROBE", nprobe, def.nprobe, 1u, 1u << 20);
    check_range("REASSIGN_RANGE", reassign_range, def.reassign_range, 0u, 1024u);
    check_range("MAX_POSTINGS", max_postings, def.max_postings, 2u, 1u << 26);
    check_range("MAX_REBALANCE_STEPS", max_rebalance_steps, def.max_rebalance_steps, 1u, 1u << 24);
    check_range("SPLIT_ITERS", split_iters, def.split_iters, 1u, 100u);
    check_range("SPLIT_MIN_SIDE", split_min_side, def.split_min_side, 0.0, 0.5);
    check_range("REPLICAS", replicas, def.replicas, 1u, 16u);
    check_range("REPLICA_CANDIDATES", replica_candidates, def.replica_candidates, 1u, 1024u);
    if (replica_candidates < replicas) {
        LOG(warning, "IVF index: VESPA_IVF_REPLICA_CANDIDATES=%u is below VESPA_IVF_REPLICAS=%u, using %u",
            replica_candidates, replicas, std::max(def.replica_candidates, replicas));
        replica_candidates = std::max(def.replica_candidates, replicas);
    }
    check_range("REPLICA_EPS", replica_eps, def.replica_eps, 0.0, 100.0);
    check_range("RNG_FACTOR", rng_factor, def.rng_factor, 0.0, 100.0);
    check_range("BULK_FILL", bulk_fill, def.bulk_fill, 0.05, 1.0);
    check_range("BULK_SAMPLE", bulk_sample, def.bulk_sample, 1000u, 1u << 26);
    check_range("BULK_ITERS", bulk_iters, def.bulk_iters, 1u, 100u);
    check_range("PRUNE_EPS", prune_eps, def.prune_eps, 0.0, 1e6);
    check_range("FILTER_MIN_RATIO", filter_min_ratio, def.filter_min_ratio, 1e-6, 1.0);
    check_range("FILTER_MAX_SCALE", filter_max_scale, def.filter_max_scale, 1.0, 1e4);
    check_range("BULK_COARSE_PROBES", bulk_coarse_probes, def.bulk_coarse_probes, 1u, 65536u);
    check_range("GC_STEPS", gc_steps, def.gc_steps, 0u, 65536u);
    check_range("GC_STALE_RATIO", gc_stale_ratio, def.gc_stale_ratio, 0.01, 1.0);
    check_range("MERGE_FRACTION", merge_fraction, def.merge_fraction, 0.0, 0.5);
    check_range("SPLIT_LIVE_FRACTION", split_live_fraction, def.split_live_fraction, 0.0, 1.0);
    check_range("BULK_COARSE_K", bulk_coarse_k, def.bulk_coarse_k, 0u, 1u << 20);
    check_range("FILE_GROW_EXTENTS", file_grow_extents, def.file_grow_extents, 1u, 1u << 20);
    check_range("FILE_GROW_FRACTION", file_grow_fraction, def.file_grow_fraction, 0.0, 4.0);
    check_range("MEDOID_SAMPLE", medoid_sample, def.medoid_sample, 2u, 4096u);
    check_range("MEDOID_ROUNDS", medoid_rounds, def.medoid_rounds, 0u, 100u);
    check_range("FILTER_MAX_EXAMINED", filter_max_examined, def.filter_max_examined, 1u, 1u << 20);
    check_range("FILTER_ENTRY_READ_DENSITY", filter_entry_read_density, def.filter_entry_read_density, 0.0, 1.0);
    check_range("FILTER_EXACT_RATIO", filter_exact_ratio, def.filter_exact_ratio, 0.0, 1.0);
    check_range("K_ENTRY_FACTOR", k_entry_factor, def.k_entry_factor, 0.0, 1000.0);
}

namespace ivf {

PreparedAdd::PreparedAdd(uint32_t docid_in, ReadGuard read_guard_in, ReadGuard index_read_guard_in) noexcept
    : docid(docid_in),
      read_guard(std::move(read_guard_in)),
      index_read_guard(std::move(index_read_guard_in)),
      slots() {
}

PreparedAdd::~PreparedAdd() = default;

} // namespace ivf

IvfIndex::IvfIndex(const DocVectorAccess& vectors, DistanceFunctionFactory::UP distance_ff,
                   vespalib::eval::CellType cell_type, size_t vector_size, DistanceMetric metric,
                   const IvfIndexConfig& cfg, const std::string& posting_path)
    : _vectors(vectors),
      _distance_ff(std::move(distance_ff)),
      _cfg(sanitized(cfg)),
      _cell_type(cell_type),
      _vector_size(vector_size),
      _vector_bytes(CellTypeUtils::mem_size(cell_type, vector_size)),
      _entry_bytes(align_up_8(entry_header_bytes + _vector_bytes)),
      _extent_bytes(_entry_bytes * _cfg.posting_capacity),
      _centroid_bytes(_vector_bytes),
      _space(*_distance_ff, cell_type, vector_size, metric),
      _posting_path(posting_path),
      _persistent(false),
      _fd(-1),
      _file_ready(false),
      _file_id(0),
      _disk_file_id(0),
      _slot_table(nullptr),
      _slot_table_bytes(0),
      _slot_refs(nullptr),
      _centroids(nullptr),
      _slot_limit(0),
      _extent_lids(nullptr),
      _lid_table_bytes(0),
      _max_extents(0),
      _versions(),
      _version_limit(0),
      _generation_handler(),
      _slot_hold(),
      _extent_hold(),
      _free_slots(),
      _free_extents(),
      _pending_extents(),
      _pinned_extents(),
      _saves_started(0),
      _saves_completed(0),
      _extent_limit(0),
      _file_extents(0),
      _gc_cursor(0),
      _reassign_queue(),
      _entry_buf(_entry_bytes, 0),
      _rng(0x1f2e3d4c),
      _counters(),
      _fadvise_warned(false),
      _load_source(IvfLoadSource::NONE),
      _load_seconds(0.0),
      _load_rejected() {
    assert(_distance_ff);
    validate_config(_vector_size);
    _slot_table_bytes = size_t(_cfg.max_postings) * (ref_bytes + _centroid_bytes);
    _slot_table = ::mmap(nullptr, _slot_table_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (_slot_table == MAP_FAILED) {
        _slot_table = nullptr;
        throw IllegalStateException(
            make_string("IVF index: mmap(%zu bytes) failed: %s", _slot_table_bytes, std::strerror(errno)),
            VESPA_STRLOC);
    }
    _slot_refs = static_cast<uint64_t*>(_slot_table);
    _centroids = static_cast<char*>(_slot_table) + size_t(_cfg.max_postings) * ref_bytes;
    // Live postings plus extents on hold never exceed two extents per posting slot. Extents pinned by saves come on
    // top; they are few compared to max_postings unless saves keep failing.
    _max_extents = 2 * _cfg.max_postings;
    _lid_table_bytes = size_t(_max_extents) * _cfg.posting_capacity * sizeof(uint32_t);
    void* lids = ::mmap(nullptr, _lid_table_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                        -1, 0);
    if (lids == MAP_FAILED) {
        int mmap_errno = errno;
        ::munmap(_slot_table, _slot_table_bytes);
        throw IllegalStateException(
            make_string("IVF index: mmap(%zu bytes) failed: %s", _lid_table_bytes, std::strerror(mmap_errno)),
            VESPA_STRLOC);
    }
    _extent_lids = static_cast<uint32_t*>(lids);
    try {
        open_posting_file();
    } catch (...) {
        if (_fd >= 0) {
            ::close(_fd);
        }
        ::munmap(_extent_lids, _lid_table_bytes);
        ::munmap(_slot_table, _slot_table_bytes);
        throw;
    }
}

IvfIndex::~IvfIndex() {
    _slot_hold.reclaim_all();
    _extent_hold.reclaim_all();
    if (_fd >= 0 && ::close(_fd) != 0) {
        LOG(warning, "IVF index: close() failed: %s", std::strerror(errno));
    }
    if (_slot_table != nullptr && ::munmap(_slot_table, _slot_table_bytes) != 0) {
        LOG(warning, "IVF index: munmap() failed: %s", std::strerror(errno));
    }
    if (_extent_lids != nullptr && ::munmap(_extent_lids, _lid_table_bytes) != 0) {
        LOG(warning, "IVF index: munmap() failed: %s", std::strerror(errno));
    }
}

uint64_t IvfIndex::acquire_ref(uint32_t slot) const noexcept {
    return std::atomic_ref<uint64_t>(_slot_refs[slot]).load(std::memory_order_acquire);
}

void IvfIndex::release_ref(uint32_t slot, uint64_t ref) noexcept {
    std::atomic_ref<uint64_t>(_slot_refs[slot]).store(ref, std::memory_order_release);
}

vespalib::eval::TypedCells IvfIndex::centroid(uint32_t slot) const noexcept {
    return {_centroids + size_t(slot) * _centroid_bytes, _cell_type, _vector_size};
}

vespalib::eval::TypedCells IvfIndex::entry_cells(const char* entry) const noexcept {
    return {entry + entry_header_bytes, _cell_type, _vector_size};
}

uint32_t IvfIndex::entry_docid(const char* entry) noexcept {
    uint32_t docid;
    std::memcpy(&docid, entry, sizeof(docid));
    return docid;
}

uint32_t IvfIndex::entry_version(const char* entry) noexcept {
    uint32_t version;
    std::memcpy(&version, entry + sizeof(uint32_t), sizeof(version));
    return version;
}

uint32_t IvfIndex::acquire_version(uint32_t docid) const noexcept {
    uint32_t limit = _version_limit.load(std::memory_order_acquire);
    return (docid < limit) ? _versions.acquire_elem_ref(docid).load_acquire() : 0;
}

bool IvfIndex::entry_is_live(const char* entry) const noexcept {
    uint32_t version = entry_version(entry);
    return (version != 0) && (version == acquire_version(entry_docid(entry)));
}

bool IvfIndex::slot_is_live(uint32_t slot) const noexcept {
    return (slot < _slot_limit.load(std::memory_order_acquire)) && (acquire_ref(slot) != 0);
}

IvfIndex::SlotDistances IvfIndex::top_slots(const BoundDistanceFunction& df, uint32_t count, uint32_t exclude_a,
                                            uint32_t exclude_b, Stats* stats) const {
    SlotDistances heap;
    if (count == 0) {
        return heap;
    }
    uint32_t limit = _slot_limit.load(std::memory_order_acquire);
    heap.reserve(std::min(count, limit));
    for (uint32_t slot = 0; slot < limit; ++slot) {
        if (slot == exclude_a || slot == exclude_b || acquire_ref(slot) == 0) {
            continue;
        }
        double distance = df.calc(centroid(slot));
        if (stats != nullptr) {
            stats->count_computed_distance();
        }
        if (heap.size() < count) {
            heap.emplace_back(distance, slot);
            std::push_heap(heap.begin(), heap.end());
        } else if (distance < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {distance, slot};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort_heap(heap.begin(), heap.end());
    return heap;
}

void IvfIndex::read_posting(Posting posting, std::vector<char>& buf) const {
    size_t bytes = size_t(posting.length) * _entry_bytes;
    buf.resize(bytes);
    pread_fully(_fd, buf.data(), bytes, extent_offset(posting.extent));
}

uint32_t IvfIndex::lid_at(uint32_t extent, uint32_t index) const noexcept {
    return std::atomic_ref<uint32_t>(_extent_lids[size_t(extent) * _cfg.posting_capacity + index])
        .load(std::memory_order_relaxed);
}

// Writer only; the posting reference published afterwards (release) makes the lids visible to readers.
void IvfIndex::set_lids(uint32_t extent, uint32_t first, const char* entries, uint32_t count) noexcept {
    for (uint32_t i = 0; i < count; ++i) {
        std::atomic_ref<uint32_t>(_extent_lids[size_t(extent) * _cfg.posting_capacity + first + i])
            .store(entry_docid(entries + size_t(i) * _entry_bytes), std::memory_order_relaxed);
    }
}

std::vector<NearestNeighborIndex::Neighbor>
IvfIndex::search(Stats& stats, uint32_t k, const BoundDistanceFunction& df, const BoundDistanceFunction& centroid_df,
                 const GlobalFilter* filter, uint32_t nprobe, const vespalib::Deadline& doom,
                 double distance_threshold, IvfProbeStats* probe_stats) const {
    if (filter != nullptr && filter->size() > 0 &&
        double(filter->count()) < _cfg.filter_exact_ratio * double(filter->size())) {
        return search_exact(stats, k, df, *filter, doom, distance_threshold);
    }
    if (filter != nullptr && _cfg.filter_probe_mode == IvfFilterProbeMode::ADAPTIVE) {
        return search_adaptive(stats, k, df, centroid_df, *filter, nprobe, doom, distance_threshold, probe_stats);
    }
    auto     guard = _generation_handler.takeGuard();
    uint32_t filter_limit = 0;
    if (filter != nullptr) {
        filter_limit = filter->size();
        double ratio = (filter_limit > 0) ? double(filter->count()) / filter_limit : 1.0;
        double scale = std::min(1.0 / std::max(ratio, _cfg.filter_min_ratio), _cfg.filter_max_scale);
        nprobe = static_cast<uint32_t>(std::min(std::ceil(nprobe * scale), double(_cfg.max_postings)));
    }
    auto scan_start = std::chrono::steady_clock::now();
    auto probes = top_slots(centroid_df, nprobe, no_slot, no_slot, &stats);
    if (probe_stats != nullptr) {
        probe_stats->centroid_scan_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - scan_start)
                .count();
    }
    if (_cfg.prune_eps > 0.0 && !probes.empty()) {
        double limit = probes.front().first + _cfg.prune_eps * std::abs(probes.front().first);
        auto beyond = std::find_if(probes.begin(), probes.end(), [limit](const auto& p) { return p.first > limit; });
        probes.erase(beyond, probes.end());
    }
    std::vector<uint64_t> refs;
    refs.reserve(probes.size());
    for (const auto& probe : probes) {
        uint64_t ref = acquire_ref(probe.second);
        if (ref != 0) {
            refs.push_back(ref);
        }
    }
    for (uint64_t ref : _cfg.prefetch ? refs : std::vector<uint64_t>()) {
        Posting posting = decode_ref(ref);
        if (posting.length == 0) {
            continue;
        }
        int rc = ::posix_fadvise(_fd, off_t(extent_offset(posting.extent)),
                                 off_t(size_t(posting.length) * _entry_bytes), POSIX_FADV_WILLNEED);
        if (rc != 0 && !_fadvise_warned.exchange(true, std::memory_order_relaxed)) {
            LOG(warning, "IVF index: posix_fadvise(WILLNEED) failed: %s", std::strerror(rc));
        }
    }
    std::vector<char> buf;
    buf.reserve(_extent_bytes);
    std::vector<Neighbor> candidates;
    for (uint64_t ref : refs) {
        if (doom.is_missed()) {
            break;
        }
        Posting posting = decode_ref(ref);
        read_posting(posting, buf);
        if (probe_stats != nullptr) {
            ++probe_stats->postings_probed;
            probe_stats->entries_scanned += posting.length;
        }
        for (uint32_t i = 0; i < posting.length; ++i) {
            const char* entry = buf.data() + size_t(i) * _entry_bytes;
            stats.count_visited_node();
            if (!entry_is_live(entry)) {
                continue;
            }
            uint32_t docid = entry_docid(entry);
            if (filter != nullptr && (docid >= filter_limit || !filter->check(docid))) {
                continue;
            }
            double distance = df.calc(entry_cells(entry));
            stats.count_computed_distance();
            candidates.emplace_back(docid, distance);
        }
    }
    return select_top_k(candidates, k, distance_threshold);
}

// Exact search over the docs that pass the filter, reading their vectors from the attribute (like Vespa's exact
// fallback), for filters so selective that probing postings would read more than the matching docs themselves.
std::vector<NearestNeighborIndex::Neighbor>
IvfIndex::search_exact(Stats& stats, uint32_t k, const BoundDistanceFunction& df, const GlobalFilter& filter,
                       const vespalib::Deadline& doom, double distance_threshold) const {
    std::vector<Neighbor> candidates;
    uint32_t              limit = filter.size();
    for (uint32_t docid = 1; docid < limit; ++docid) {
        if (!filter.check(docid)) {
            continue;
        }
        if ((docid % 1024) == 0 && doom.is_missed()) {
            break;
        }
        auto   vectors = _vectors.get_vectors(docid);
        double best = std::numeric_limits<double>::max();
        for (uint32_t subspace = 0; subspace < vectors.subspaces(); ++subspace) {
            best = std::min(best, df.calc(vectors.cells(subspace)));
            stats.count_computed_distance();
        }
        if (vectors.subspaces() > 0) {
            candidates.emplace_back(docid, best);
        }
    }
    return select_top_k(candidates, k, distance_threshold);
}

// Walks postings in centroid order, counts filter matches from the lid table without reading the posting, and reads
// postings that contain matches until the stop rule holds: nprobe such postings, or as many matching entries as an
// unfiltered query reads (nprobe times the mean posting length). Both rules also need at least k matches.
std::vector<NearestNeighborIndex::Neighbor>
IvfIndex::search_adaptive(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                          const BoundDistanceFunction& centroid_df, const GlobalFilter& filter, uint32_t nprobe,
                          const vespalib::Deadline& doom, double distance_threshold,
                          IvfProbeStats* probe_stats) const {
    auto     guard = _generation_handler.takeGuard();
    uint32_t filter_limit = filter.size();
    auto     scan_start = std::chrono::steady_clock::now();
    auto     order = top_slots(centroid_df, _cfg.filter_max_examined, no_slot, no_slot, &stats);
    if (probe_stats != nullptr) {
        probe_stats->centroid_scan_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - scan_start)
                .count();
    }
    struct Selected {
        Posting               posting;
        std::vector<uint32_t> matches;
    };
    std::vector<Selected> selected;
    uint64_t              matched = 0;
    uint64_t              postings_now = std::max<uint64_t>(1, get(_counters.postings));
    uint64_t entries_target = std::max<uint64_t>(k, uint64_t(nprobe) * get(_counters.entries) / postings_now);
    for (const auto& probe : order) {
        bool done = (_cfg.filter_adaptive_stop == IvfAdaptiveStop::POSTINGS) ? (selected.size() >= nprobe)
                                                                              : (matched >= entries_target);
        if (done && matched >= k) {
            break;
        }
        uint64_t ref = acquire_ref(probe.second);
        if (ref == 0) {
            continue;
        }
        Posting posting = decode_ref(ref);
        if (probe_stats != nullptr) {
            ++probe_stats->postings_examined;
        }
        std::vector<uint32_t> matches;
        for (uint32_t i = 0; i < posting.length; ++i) {
            uint32_t lid = lid_at(posting.extent, i);
            if (lid < filter_limit && filter.check(lid)) {
                matches.push_back(i);
            }
        }
        if (!matches.empty()) {
            matched += matches.size();
            selected.push_back({posting, std::move(matches)});
        }
    }
    if (_cfg.prefetch) {
        for (const auto& sel : selected) {
            int rc = ::posix_fadvise(_fd, off_t(extent_offset(sel.posting.extent)),
                                     off_t(size_t(sel.posting.length) * _entry_bytes), POSIX_FADV_WILLNEED);
            if (rc != 0 && !_fadvise_warned.exchange(true, std::memory_order_relaxed)) {
                LOG(warning, "IVF index: posix_fadvise(WILLNEED) failed: %s", std::strerror(rc));
            }
        }
    }
    std::vector<char>     buf;
    std::vector<Neighbor> candidates;
    for (const auto& sel : selected) {
        if (doom.is_missed()) {
            break;
        }
        bool whole = double(sel.matches.size()) >= _cfg.filter_entry_read_density * sel.posting.length;
        if (whole) {
            read_posting(sel.posting, buf);
        } else {
            buf.resize(size_t(sel.posting.length) * _entry_bytes);
            for (uint32_t i : sel.matches) {
                pread_fully(_fd, buf.data() + size_t(i) * _entry_bytes, _entry_bytes,
                            extent_offset(sel.posting.extent) + uint64_t(i) * _entry_bytes);
            }
        }
        if (probe_stats != nullptr) {
            ++probe_stats->postings_probed;
            probe_stats->entries_scanned += whole ? sel.posting.length : sel.matches.size();
        }
        for (uint32_t i : sel.matches) {
            const char* entry = buf.data() + size_t(i) * _entry_bytes;
            stats.count_visited_node();
            uint32_t docid = entry_docid(entry);
            if (!entry_is_live(entry) || docid >= filter_limit || !filter.check(docid)) {
                continue;
            }
            candidates.emplace_back(docid, df.calc(entry_cells(entry)));
            stats.count_computed_distance();
        }
    }
    return select_top_k(candidates, k, distance_threshold);
}

void IvfIndex::check_cells(TypedCells cells) const {
    if (cells.type != _cell_type || cells.size != _vector_size) {
        throw IllegalStateException(make_string("IVF index: vector has %zu cells of type %s, expected %zu of %s",
                                                size_t(cells.size), cell_type_to_name(cells.type).c_str(),
                                                _vector_size, cell_type_to_name(_cell_type).c_str()),
                                    VESPA_STRLOC);
    }
}

void IvfIndex::encode_entry(char* dst, uint32_t docid, uint32_t version, TypedCells cells) const noexcept {
    std::memcpy(dst, &docid, sizeof(docid));
    std::memcpy(dst + sizeof(docid), &version, sizeof(version));
    std::memcpy(dst + entry_header_bytes, cells.data, _vector_bytes);
    std::memset(dst + entry_header_bytes + _vector_bytes, 0, _entry_bytes - entry_header_bytes - _vector_bytes);
}

uint32_t IvfIndex::bump_version(uint32_t docid) {
    if (docid >= _versions.size()) {
        _versions.ensure_size(size_t(docid) + 1, AtomicVersion());
        _version_limit.store(_versions.size(), std::memory_order_release);
    }
    auto&    elem = _versions[docid];
    uint32_t version = elem.load_relaxed() + 1;
    elem.store_release(version);
    return version;
}

std::string IvfIndex::posting_path_for(const std::string& base_file_name) {
    std::filesystem::path base(base_file_name);
    std::filesystem::path dir = base.parent_path();
    if (dir.filename().string().starts_with("snapshot-")) {
        dir = dir.parent_path();
    }
    if (dir.empty() || base.filename().empty()) {
        return {};
    }
    return (dir / (base.filename().string() + posting_file_suffix)).string();
}

void IvfIndex::remove_posting_file(const std::string& base_file_name) {
    std::string path = posting_path_for(base_file_name);
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    if (std::filesystem::remove(path, ec)) {
        LOG(info, "IVF index: removed posting file %s, the attribute has no IVF index", path.c_str());
    } else if (ec) {
        LOG(warning, "IVF index: removing posting file %s failed: %s", path.c_str(), ec.message().c_str());
    }
}

const std::string& IvfIndex::save_file_tag() {
    static const std::string tag("ivf.save_format");
    return tag;
}

const std::vector<IvfIndex::CounterField>& IvfIndex::saved_counters() {
    static const std::vector<CounterField> fields = {
        &Counters::postings,         &Counters::entries,         &Counters::live_entries,    &Counters::splits,
        &Counters::split_fallbacks,  &Counters::compactions,     &Counters::reassign_checks, &Counters::moves,
        &Counters::dropped_steps,    &Counters::dropped_entries, &Counters::prepared_retired, &Counters::bytes_written,
        &Counters::bulk_builds,      &Counters::gc_checks,       &Counters::gc_compactions,  &Counters::merges};
    return fields;
}

uint64_t IvfIndex::extent_offset(uint32_t extent) const noexcept {
    return file_header_bytes + uint64_t(extent) * _extent_bytes;
}

// Takes an exclusive lock on the named posting file, so that two index instances never share it; the second one
// gets an unlinked file instead.
void IvfIndex::open_posting_file() {
    if (!_posting_path.empty()) {
        std::string     dir = std::filesystem::path(_posting_path).parent_path().string();
        std::error_code ec;
        if (std::filesystem::is_directory(dir, ec)) {
            int fd = ::open(_posting_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
            if (fd < 0) {
                throw IllegalStateException(
                    make_string("IVF index: open(%s) failed: %s", _posting_path.c_str(), std::strerror(errno)),
                    VESPA_STRLOC);
            }
            if (::flock(fd, LOCK_EX | LOCK_NB) == 0) {
                _fd = fd;
                _persistent = true;
                sync_directory(dir);
                _disk_file_id = read_file_id(_fd);
                return;
            }
            int lock_errno = errno;
            if (::close(fd) != 0) {
                LOG(warning, "IVF index: close(%s) failed: %s", _posting_path.c_str(), std::strerror(errno));
            }
            LOG(warning, "IVF index: cannot lock posting file %s (%s); using an unlinked file, not saved",
                _posting_path.c_str(),
                (lock_errno == EWOULDBLOCK) ? "used by another index" : std::strerror(lock_errno));
        } else {
            LOG(warning, "IVF index: directory %s does not exist; using an unlinked posting file, not saved",
                dir.c_str());
        }
        _posting_path.clear();
    } else {
        LOG(info, "IVF index: the attribute has no directory yet; using an unlinked posting file until a restart");
    }
    _fd = open_unlinked_file(unlinked_file_dir());
}

// Starts an empty posting file with a new file id, which invalidates every save file written for the old content.
void IvfIndex::start_new_file() {
    assert(!_file_ready && _extent_limit == 0);
    uint64_t file_id = new_file_id();
    if (::ftruncate(_fd, 0) != 0) {
        throw IllegalStateException(make_string("IVF index: ftruncate() failed: %s", std::strerror(errno)),
                                    VESPA_STRLOC);
    }
    std::vector<char> header(file_header_bytes, 0);
    std::memcpy(header.data(), &posting_file_magic, sizeof(posting_file_magic));
    std::memcpy(header.data() + sizeof(posting_file_magic), &posting_file_version, sizeof(posting_file_version));
    std::memcpy(header.data() + file_id_offset, &file_id, sizeof(file_id));
    pwrite_fully(_fd, header.data(), header.size(), 0);
    if (_persistent && ::fdatasync(_fd) != 0) {
        throw IllegalStateException(make_string("IVF index: fdatasync() failed: %s", std::strerror(errno)),
                                    VESPA_STRLOC);
    }
    _file_id = file_id;
    _disk_file_id = file_id;
    _file_extents = 0;
    _file_ready = true;
    set_to(_counters.file_bytes, file_header_bytes);
}

/*
 * Extent pinning. A save references postings by (extent, length) and is the snapshot loaded after a crash until a
 * newer save is durable. Appends write at a posting's current length, which only grows while its extent is live,
 * and every other write (rewrite, split, merge target, GC sweep, bulk build) goes to an extent from the free list.
 * So saved bytes stay intact as long as no extent that a loadable save references is on the free list: a freed
 * extent waits in _pending_extents, is pinned by the next save, and is released when a later save starts after the
 * pinning save completed.
 */
void IvfIndex::release_extent(uint32_t extent) {
    (_persistent ? _pending_extents : _free_extents).push_back(extent);
}

// The flush of a save ends before the next save of the attribute starts, so a completed save is the durable
// snapshot here and the extents freed before it was captured can be reused.
void IvfIndex::pin_extents_for_save() {
    uint64_t                   completed = _saves_completed.load(std::memory_order_acquire);
    std::vector<PinnedExtents> still_pinned;
    for (auto& batch : _pinned_extents) {
        if (batch.save_seq <= completed) {
            _free_extents.insert(_free_extents.end(), batch.extents.begin(), batch.extents.end());
        } else {
            still_pinned.push_back(std::move(batch));
        }
    }
    _pinned_extents = std::move(still_pinned);
    ++_saves_started;
    _pinned_extents.push_back({_saves_started, std::move(_pending_extents)});
    _pending_extents.clear();
    if (_pinned_extents.size() > 2) {
        LOG(warning, "IVF index %s: %zu saves have not completed; their freed extents are not reused",
            _posting_path.c_str(), _pinned_extents.size() - 1);
    }
    update_extent_counters();
}

void IvfIndex::update_extent_counters() noexcept {
    uint64_t pinned = _pending_extents.size();
    for (const auto& batch : _pinned_extents) {
        pinned += batch.extents.size();
    }
    set_to(_counters.free_extents, _free_extents.size());
    set_to(_counters.pinned_extents, pinned);
}

ivf::SaveLayout IvfIndex::layout() const noexcept {
    return {static_cast<uint32_t>(_cell_type),
            static_cast<uint32_t>(_vector_size),
            static_cast<uint32_t>(_space.metric),
            _cfg.posting_capacity,
            static_cast<uint32_t>(_entry_bytes),
            static_cast<uint32_t>(_centroid_bytes)};
}

// Slots on hold come after the free slots, so the load leaves them in the order they get after the next reclaim.
ivf::SaveState IvfIndex::capture_save_state() const {
    ivf::SaveState state;
    uint32_t       slot_limit = _slot_limit.load(std::memory_order_relaxed);
    state.layout = layout();
    state.file_id = _file_id;
    state.slot_limit = slot_limit;
    state.extent_limit = _extent_limit;
    state.gc_cursor = _gc_cursor;
    auto* mips = dynamic_cast<MipsDistanceFunctionFactoryBase*>(_distance_ff.get());
    state.max_squared_norm = (mips != nullptr) ? mips->get_max_squared_norm_store().get_max() : -1.0;
    for (CounterField field : saved_counters()) {
        state.counters.push_back(get(_counters.*field));
    }
    std::ostringstream rng;
    rng << _rng;
    state.rng_state = rng.str();
    std::vector<bool> is_free(slot_limit, false);
    for (uint32_t slot : _free_slots) {
        assert(slot < slot_limit);
        is_free[slot] = true;
    }
    state.free_slots = _free_slots;
    state.refs.resize(slot_limit);
    for (uint32_t slot = 0; slot < slot_limit; ++slot) {
        uint64_t ref = acquire_ref(slot);
        state.refs[slot] = ref;
        if (ref != 0) {
            const char* src = _centroids + size_t(slot) * _centroid_bytes;
            state.centroids.insert(state.centroids.end(), src, src + _centroid_bytes);
        } else if (!is_free[slot]) {
            state.free_slots.push_back(slot);
        }
    }
    state.versions.resize(_versions.get_size());
    for (size_t docid = 0; docid < state.versions.size(); ++docid) {
        state.versions[docid] = _versions.get_elem_ref(docid).load_relaxed();
    }
    return state;
}

// Checks a save file against this index and its posting file. Returns the reason it cannot be used, or "".
std::string IvfIndex::check_save_state(const ivf::SaveState& state) const {
    if (state.layout != layout()) {
        return "cell type, dimensions, distance metric or posting-capacity changed";
    }
    if (state.file_id == 0 || state.file_id != _disk_file_id) {
        return "the posting file was rewritten or replaced after the save";
    }
    if (state.slot_limit > _cfg.max_postings || state.extent_limit > _max_extents) {
        return make_string("%u slots / %u extents exceed max-postings %u", state.slot_limit, state.extent_limit,
                           _cfg.max_postings);
    }
    if (state.counters.size() != saved_counters().size() || state.gc_cursor > state.slot_limit) {
        return "bad header values";
    }
    uint64_t          file_bytes = file_size_of(_fd);
    uint64_t          free_slots = 0;
    std::vector<bool> extent_used(state.extent_limit, false);
    for (uint32_t slot = 0; slot < state.slot_limit; ++slot) {
        uint64_t ref = state.refs[slot];
        if (ref == 0) {
            ++free_slots;
            continue;
        }
        Posting posting = decode_ref(ref);
        if (posting.extent >= state.extent_limit || posting.length == 0 ||
            posting.length > _cfg.posting_capacity || extent_used[posting.extent])
        {
            return make_string("bad posting reference in slot %u", slot);
        }
        extent_used[posting.extent] = true;
        if (extent_offset(posting.extent) + uint64_t(posting.length) * _entry_bytes > file_bytes) {
            return "the posting file is shorter than the saved postings";
        }
    }
    std::vector<bool> listed(state.slot_limit, false);
    for (uint32_t slot : state.free_slots) {
        if (slot >= state.slot_limit || state.refs[slot] != 0 || listed[slot]) {
            return "bad free slot list";
        }
        listed[slot] = true;
    }
    if (state.free_slots.size() != free_slots) {
        return "bad free slot list";
    }
    for (uint32_t lid : state.lids) {
        if (lid >= state.versions.size()) {
            return "a posting entry has a docid without a version";
        }
    }
    std::mt19937       rng;
    std::istringstream rng_state(state.rng_state);
    rng_state >> rng;
    if (rng_state.fail()) {
        return "bad random generator state";
    }
    return {};
}

void IvfIndex::apply_save_state(const ivf::SaveState& state) {
    assert(!_file_ready && _slot_limit.load(std::memory_order_relaxed) == 0 && _versions.size() == 0);
    const char*       centroid = state.centroids.data();
    const uint32_t*   lids = state.lids.data();
    std::vector<bool> extent_used(state.extent_limit, false);
    for (uint32_t slot = 0; slot < state.slot_limit; ++slot) {
        uint64_t ref = state.refs[slot];
        release_ref(slot, ref);
        if (ref == 0) {
            continue;
        }
        Posting posting = decode_ref(ref);
        std::memcpy(_centroids + size_t(slot) * _centroid_bytes, centroid, _centroid_bytes);
        centroid += _centroid_bytes;
        for (uint32_t i = 0; i < posting.length; ++i) {
            std::atomic_ref<uint32_t>(_extent_lids[size_t(posting.extent) * _cfg.posting_capacity + i])
                .store(lids[i], std::memory_order_relaxed);
        }
        lids += posting.length;
        extent_used[posting.extent] = true;
    }
    assert(centroid == state.centroids.data() + state.centroids.size());
    assert(lids == state.lids.data() + state.lids.size());
    _slot_limit.store(state.slot_limit, std::memory_order_release);
    if (!state.versions.empty()) {
        _versions.ensure_size(state.versions.size(), AtomicVersion());
        for (size_t docid = 0; docid < state.versions.size(); ++docid) {
            _versions[docid].store_release(state.versions[docid]);
        }
    }
    _version_limit.store(_versions.size(), std::memory_order_release);
    _free_slots = state.free_slots;
    _extent_limit = state.extent_limit;
    _free_extents.clear();
    for (uint32_t extent = state.extent_limit; extent-- > 0;) {
        if (!extent_used[extent]) {
            _free_extents.push_back(extent);
        }
    }
    _gc_cursor = state.gc_cursor;
    std::istringstream rng_state(state.rng_state);
    rng_state >> _rng;
    const auto& fields = saved_counters();
    for (size_t i = 0; i < fields.size(); ++i) {
        set_to(_counters.*fields[i], state.counters[i]);
    }
    auto* mips = dynamic_cast<MipsDistanceFunctionFactoryBase*>(_distance_ff.get());
    if (mips != nullptr && state.max_squared_norm >= 0.0) {
        (void)mips->get_max_squared_norm_store().get_max(state.max_squared_norm);
    }
    _file_id = state.file_id;
    _file_ready = true;
    uint64_t file_bytes = file_size_of(_fd);
    uint64_t extent_bytes_in_file = (file_bytes > file_header_bytes) ? file_bytes - file_header_bytes : 0;
    _file_extents = static_cast<uint32_t>(
        std::min<uint64_t>(extent_bytes_in_file / _extent_bytes, std::numeric_limits<uint32_t>::max() - 1));
    set_to(_counters.file_bytes, extent_offset(_file_extents));
    if (_file_extents < _extent_limit) {
        grow_file(_extent_limit);
    }
    set_to(_counters.extents, _extent_limit);
    update_extent_counters();
}

void IvfIndex::load_save_file(FastOS_FileInterface& file) {
    auto        start = std::chrono::steady_clock::now();
    std::string reason;
    if (_cfg.rebuild_on_load) {
        reason = "rebuild-on-load is set";
    } else if (!_persistent) {
        reason = "the posting file is not available";
    } else if (_file_ready || _slot_limit.load(std::memory_order_relaxed) != 0) {
        reason = "the index is not empty";
    }
    ivf::SaveState state;
    if (reason.empty()) {
        reason = ivf::read_save_file(file, state);
    }
    if (reason.empty()) {
        reason = check_save_state(state);
    }
    if (!reason.empty()) {
        LOG(warning, "IVF index: not using the saved index %s: %s; building the index from the attribute",
            file.GetFileName(), reason.c_str());
        _load_rejected = reason;
        _load_source.store(IvfLoadSource::NONE, std::memory_order_relaxed);
        return;
    }
    apply_save_state(state);
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    _load_rejected.clear();
    _load_source.store(IvfLoadSource::SAVE_FILE, std::memory_order_relaxed);
    _load_seconds.store(elapsed.count(), std::memory_order_relaxed);
    LOG(info, "IVF index: loaded %" PRIu64 " postings with %" PRIu64 " entries from %s in %.2f s",
        get(_counters.postings), get(_counters.entries), file.GetFileName(), elapsed.count());
}

void IvfIndex::grow_file(uint64_t new_file_extents) {
    if (new_file_extents <= _file_extents) {
        throw IllegalStateException("IVF index: extent id space exhausted", VESPA_STRLOC);
    }
    off_t offset = off_t(extent_offset(_file_extents));
    off_t len = off_t((new_file_extents - _file_extents) * _extent_bytes);
    if (::fallocate(_fd, 0, offset, len) != 0) {
        int fallocate_errno = errno;
        if (fallocate_errno != EOPNOTSUPP || ::ftruncate(_fd, offset + len) != 0) {
            throw IllegalStateException(make_string("IVF index: growing file to %" PRIu64 " bytes failed: %s",
                                                    uint64_t(offset + len), std::strerror(errno)),
                                        VESPA_STRLOC);
        }
    }
    _file_extents = new_file_extents;
    set_to(_counters.file_bytes, extent_offset(_file_extents));
}

uint32_t IvfIndex::allocate_extent() {
    if (!_file_ready) {
        start_new_file();
    }
    if (!_free_extents.empty()) {
        uint32_t extent = _free_extents.back();
        _free_extents.pop_back();
        set_to(_counters.free_extents, _free_extents.size());
        return extent;
    }
    if (_extent_limit == _file_extents) {
        uint64_t grow = std::max<uint64_t>(_cfg.file_grow_extents, uint64_t(_file_extents * _cfg.file_grow_fraction));
        grow_file(std::min<uint64_t>(uint64_t(_file_extents) + grow, std::numeric_limits<uint32_t>::max() - 1));
    }
    if (_extent_limit >= _max_extents) {
        throw IllegalStateException(make_string("IVF index: more than %u extents", _max_extents), VESPA_STRLOC);
    }
    uint32_t extent = _extent_limit++;
    set_to(_counters.extents, _extent_limit);
    return extent;
}

void IvfIndex::retire_extent(uint32_t extent) {
    _extent_hold.insert(extent);
}

uint32_t IvfIndex::slots_available() const noexcept {
    return _free_slots.size() + (_cfg.max_postings - _slot_limit.load(std::memory_order_relaxed));
}

uint32_t IvfIndex::write_new_extent(const char* entries, uint32_t count) {
    assert(count > 0 && count <= _cfg.posting_capacity);
    uint32_t extent = allocate_extent();
    pwrite_fully(_fd, entries, size_t(count) * _entry_bytes, extent_offset(extent));
    add_to(_counters.bytes_written, size_t(count) * _entry_bytes);
    set_lids(extent, 0, entries, count);
    return extent;
}

uint32_t IvfIndex::create_posting(const char* entries, uint32_t count, TypedCells centroid_cells) {
    if (slots_available() == 0) {
        return no_slot;
    }
    uint32_t extent = write_new_extent(entries, count);
    uint32_t limit = _slot_limit.load(std::memory_order_relaxed);
    uint32_t slot = limit;
    if (!_free_slots.empty()) {
        slot = _free_slots.back();
        _free_slots.pop_back();
    }
    ivf::write_as_centroid(_space, centroid_cells, _centroids + size_t(slot) * _centroid_bytes);
    release_ref(slot, make_ref(extent, count));
    if (slot == limit) {
        _slot_limit.store(limit + 1, std::memory_order_release);
    }
    add_to(_counters.postings, 1);
    add_to(_counters.entries, count);
    return slot;
}

void IvfIndex::retire_slot(uint32_t slot) {
    uint64_t ref = acquire_ref(slot);
    assert(ref != 0);
    Posting posting = decode_ref(ref);
    release_ref(slot, 0);
    retire_extent(posting.extent);
    _slot_hold.insert(slot);
    sub_from(_counters.postings, 1);
    sub_from(_counters.entries, posting.length);
}

void IvfIndex::drop_entries(uint32_t docid, uint32_t count) {
    add_to(_counters.dropped_entries, count);
    LOG(debug, "IVF index: no free posting slot (max_postings=%u), dropped %u entries of docid %u", _cfg.max_postings,
        count, docid);
}

ivf::ReplicaRule IvfIndex::replica_rule() const noexcept {
    return {_cfg.replicas, _cfg.replica_candidates, _cfg.replica_eps, _cfg.rng_factor};
}

std::vector<uint32_t> IvfIndex::select_postings(TypedCells cells) const {
    auto     df = _space.bind(cells);
    uint32_t candidates = (_cfg.replicas == 1) ? 1 : _cfg.replica_candidates;
    auto     nearest = top_slots(*df, candidates, no_slot, no_slot, nullptr);
    if (_cfg.replicas == 1 || nearest.empty()) {
        return nearest.empty() ? std::vector<uint32_t>() : std::vector<uint32_t>{nearest.front().second};
    }
    return ivf::select_replicas(_space, nearest, replica_rule(),
                                [this](uint32_t slot) noexcept { return centroid(slot); });
}

void IvfIndex::add_entries(uint32_t docid, VectorBundle vectors, const PreparedSlots* prepared) {
    uint32_t version = bump_version(docid);
    append_copies(docid, version, vectors, prepared);
    add_to(_counters.live_entries, vectors.subspaces());
    run_rebalance();
}

uint32_t IvfIndex::append_copies(uint32_t docid, uint32_t version, VectorBundle vectors,
                                 const PreparedSlots* prepared) {
    uint32_t appended = 0;
    for (uint32_t subspace = 0; subspace < vectors.subspaces(); ++subspace) {
        TypedCells cells = vectors.cells(subspace);
        check_cells(cells);
        std::vector<uint32_t> slots;
        if (prepared != nullptr) {
            slots = (*prepared)[subspace];
            if (std::any_of(slots.begin(), slots.end(), [this](uint32_t slot) { return !slot_is_live(slot); })) {
                add_to(_counters.prepared_retired, 1);
                slots.clear();
            }
        }
        if (slots.empty()) {
            slots = select_postings(cells);
        }
        encode_entry(_entry_buf.data(), docid, version, cells);
        if (slots.empty()) {
            append_entry(no_slot, _entry_buf.data());
            ++appended;
        }
        for (uint32_t slot : slots) {
            append_entry(slot, _entry_buf.data());
            ++appended;
        }
    }
    return appended;
}

void IvfIndex::append_entry(uint32_t slot, const char* entry) {
    if (slot == no_slot) {
        if (create_posting(entry, 1, entry_cells(entry)) == no_slot) {
            drop_entries(entry_docid(entry), 1);
        }
        return;
    }
    Posting posting = decode_ref(acquire_ref(slot));
    if (posting.length < _cfg.posting_capacity) {
        uint64_t offset = extent_offset(posting.extent) + uint64_t(posting.length) * _entry_bytes;
        pwrite_fully(_fd, entry, _entry_bytes, offset);
        add_to(_counters.bytes_written, _entry_bytes);
        set_lids(posting.extent, posting.length, entry, 1);
        release_ref(slot, make_ref(posting.extent, posting.length + 1));
        add_to(_counters.entries, 1);
        return;
    }
    rebuild_full_posting(slot, entry);
}

void IvfIndex::rebuild_full_posting(uint32_t slot, const char* pending_entry) {
    Posting           posting = decode_ref(acquire_ref(slot));
    std::vector<char> buf;
    read_posting(posting, buf);
    std::vector<char> live;
    live.reserve(buf.size() + _entry_bytes);
    for (uint32_t i = 0; i < posting.length; ++i) {
        const char* entry = buf.data() + size_t(i) * _entry_bytes;
        if (entry_is_live(entry)) {
            live.insert(live.end(), entry, entry + _entry_bytes);
        }
    }
    live.insert(live.end(), pending_entry, pending_entry + _entry_bytes);
    uint32_t count = live.size() / _entry_bytes;
    bool     can_split = slots_available() >= 2;
    if (count > _cfg.split_live_fraction * _cfg.posting_capacity && can_split) {
        split_posting(slot, live.data(), count);
        return;
    }
    if (count > _cfg.posting_capacity) {
        drop_entries(entry_docid(pending_entry), 1);
        --count;
    }
    uint32_t extent = write_new_extent(live.data(), count);
    release_ref(slot, make_ref(extent, count));
    retire_extent(posting.extent);
    add_to(_counters.entries, count);
    sub_from(_counters.entries, posting.length);
    add_to(_counters.compactions, 1);
}

ivf::TwoMeansResult IvfIndex::split_points(const std::vector<TypedCells>& points) {
    if (_cfg.split_mode == IvfSplitMode::MEDOID_BALANCED) {
        return ivf::medoid_balanced_split(_space, points, _cfg.medoid_sample, _cfg.medoid_rounds);
    }
    return ivf::two_means(_space, points, _rng, _cfg.split_iters, _cfg.split_min_side);
}

void IvfIndex::split_posting(uint32_t slot, const char* entries, uint32_t count) {
    std::vector<TypedCells> points;
    points.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        points.push_back(entry_cells(entries + size_t(i) * _entry_bytes));
    }
    auto              split = split_points(points);
    std::vector<char> group_1;
    std::vector<char> group_2;
    for (uint32_t i = 0; i < count; ++i) {
        const char* entry = entries + size_t(i) * _entry_bytes;
        auto&       group = (split.side[i] == 0) ? group_1 : group_2;
        group.insert(group.end(), entry, entry + _entry_bytes);
    }
    ReassignTask task;
    task.centroids.resize(3 * _centroid_bytes);
    std::memcpy(task.centroids.data(), centroid(slot).data, _centroid_bytes);
    std::memcpy(task.centroids.data() + _centroid_bytes, split.centroid_0.data(), _centroid_bytes);
    std::memcpy(task.centroids.data() + 2 * _centroid_bytes, split.centroid_1.data(), _centroid_bytes);
    task.slot_a1 =
        create_posting(group_1.data(), group_1.size() / _entry_bytes, _space.cells(split.centroid_0.data()));
    task.slot_a2 =
        create_posting(group_2.data(), group_2.size() / _entry_bytes, _space.cells(split.centroid_1.data()));
    assert(task.slot_a1 != no_slot && task.slot_a2 != no_slot);
    retire_slot(slot);
    add_to(_counters.splits, 1);
    if (split.balanced_fallback) {
        add_to(_counters.split_fallbacks, 1);
    }
    _reassign_queue.push_back(std::move(task));
}

void IvfIndex::run_rebalance() {
    uint32_t budget = _cfg.max_rebalance_steps;
    while (!_reassign_queue.empty()) {
        if (budget == 0) {
            add_to(_counters.dropped_steps, _reassign_queue.size());
            LOG(debug, "IVF index: rebalance step bound %u reached, dropped %zu reassign checks",
                _cfg.max_rebalance_steps, _reassign_queue.size());
            _reassign_queue.clear();
            break;
        }
        ReassignTask task = std::move(_reassign_queue.front());
        _reassign_queue.pop_front();
        process_reassign(task, budget);
    }
}

void IvfIndex::process_reassign(const ReassignTask& task, uint32_t& budget) {
    assert(budget > 0);
    --budget;
    add_to(_counters.reassign_checks, 1);
    TypedCells      old_centroid = _space.cells(task.centroids.data());
    TypedCells      c1 = _space.cells(task.centroids.data() + _centroid_bytes);
    TypedCells      c2 = _space.cells(task.centroids.data() + 2 * _centroid_bytes);
    auto            df_old = _space.bind(old_centroid);
    auto            df_1 = _space.bind(c1);
    auto            df_2 = _space.bind(c2);
    ReassignContext ctx{*df_old, *df_1, *df_2, task.slot_a1, task.slot_a2};
    auto            nearby = top_slots(*df_old, _cfg.reassign_range, task.slot_a1, task.slot_a2, nullptr);
    std::unordered_set<uint32_t> reassigned;
    reassign_from(task.slot_a1, ctx, reassigned, budget);
    reassign_from(task.slot_a2, ctx, reassigned, budget);
    for (const auto& near : nearby) {
        reassign_from(near.second, ctx, reassigned, budget);
    }
}

// Follows SPFresh: a candidate vector whose posting is no longer among its selected postings is re-added to the
// selected postings with a new version, which makes all its old copies stale.
void IvfIndex::reassign_from(uint32_t slot, const ReassignContext& ctx, std::unordered_set<uint32_t>& reassigned,
                             uint32_t& budget) {
    if (budget == 0) {
        add_to(_counters.dropped_steps, 1);
        return;
    }
    --budget;
    if (!slot_is_live(slot)) {
        return;
    }
    Posting           posting = decode_ref(acquire_ref(slot));
    std::vector<char> buf;
    read_posting(posting, buf);
    bool is_new = (slot == ctx.slot_a1 || slot == ctx.slot_a2);
    auto df_current = _space.bind(centroid(slot));
    for (uint32_t i = 0; i < posting.length; ++i) {
        if (budget == 0) {
            add_to(_counters.dropped_steps, 1);
            break;
        }
        const char* entry = buf.data() + size_t(i) * _entry_bytes;
        uint32_t    docid = entry_docid(entry);
        if (!entry_is_live(entry) || reassigned.contains(docid)) {
            continue;
        }
        TypedCells cells = entry_cells(entry);
        double     d_old = ctx.df_old.calc(cells);
        double     d_1 = ctx.df_1.calc(cells);
        double     d_2 = ctx.df_2.calc(cells);
        double     d_current = df_current->calc(cells);
        bool       candidate = is_new ? (d_old < d_current)
                                      : !((d_old <= d_1 && d_old <= d_2) || (d_current <= d_1 && d_current <= d_2));
        if (!candidate) {
            continue;
        }
        auto selected = select_postings(cells);
        if (std::find(selected.begin(), selected.end(), slot) != selected.end()) {
            continue;
        }
        reassigned.insert(docid);
        --budget;
        uint32_t appended = append_copies(docid, bump_version(docid), _vectors.get_vectors(docid), nullptr);
        add_to(_counters.moves, appended);
        budget -= std::min(budget, appended);
    }
}

void IvfIndex::add_document(uint32_t docid) {
    add_entries(docid, _vectors.get_vectors(docid), nullptr);
}

bool IvfIndex::bulk_build(const std::vector<uint32_t>& lids, vespalib::Executor* executor) {
    if (!_cfg.bulk_build || _slot_limit.load(std::memory_order_relaxed) != 0) {
        return false;
    }
    auto                    start = std::chrono::steady_clock::now();
    std::vector<uint32_t>   item_lids;
    std::vector<TypedCells> points;
    for (uint32_t lid : lids) {
        auto vectors = _vectors.get_vectors(lid);
        for (uint32_t subspace = 0; subspace < vectors.subspaces(); ++subspace) {
            check_cells(vectors.cells(subspace));
            item_lids.push_back(lid);
            points.push_back(vectors.cells(subspace));
        }
    }
    uint64_t target_size = std::max<uint64_t>(1, std::lround(_cfg.posting_capacity * _cfg.bulk_fill));
    uint64_t clusters_wanted = (points.size() + target_size - 1) / target_size;
    if (clusters_wanted * std::max(2u, _cfg.replicas) > _cfg.max_postings) {
        LOG(warning, "IVF index: %zu entries need about %" PRIu64 " postings, max_postings is %u; adding one by one",
            points.size(), clusters_wanted, _cfg.max_postings);
        return false;
    }
    for (uint32_t lid : lids) {
        bump_version(lid);
    }
    add_to(_counters.live_entries, points.size());
    add_to(_counters.bulk_builds, 1);
    if (points.empty()) {
        return true;
    }
    uint32_t clusters = 0;
    auto     assignment = ivf::two_level_partition(_space, points, clusters_wanted, _cfg.bulk_sample, _cfg.bulk_iters,
                                                   _cfg.bulk_coarse_k, _cfg.bulk_coarse_probes, replica_rule(), _rng,
                                                   executor, clusters);
    std::vector<uint32_t> offsets(clusters + 1, 0);
    for (uint32_t cluster : assignment) {
        if (cluster != ivf::no_cluster) {
            ++offsets[cluster + 1];
        }
    }
    for (uint32_t c = 0; c < clusters; ++c) {
        offsets[c + 1] += offsets[c];
    }
    uint32_t              replicas = assignment.size() / points.size();
    std::vector<uint32_t> order(offsets[clusters]);
    std::vector<uint32_t> fill(offsets.begin(), offsets.end() - 1);
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] != ivf::no_cluster) {
            order[fill[assignment[i]]++] = i / replicas;
        }
    }
    for (uint32_t c = 0; c < clusters; ++c) {
        if (offsets[c] < offsets[c + 1]) {
            write_bulk_cluster(std::vector<uint32_t>(order.begin() + offsets[c], order.begin() + offsets[c + 1]),
                               item_lids, points);
        }
    }
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    _load_source.store(IvfLoadSource::BULK_BUILD, std::memory_order_relaxed);
    _load_seconds.store(elapsed.count(), std::memory_order_relaxed);
    LOG(info,
        "IVF index: bulk built %zu vectors of %zu documents into %" PRIu64 " entries in %" PRIu64
        " postings in %.1f s",
        points.size(), lids.size(), get(_counters.entries), get(_counters.postings), elapsed.count());
    return true;
}

// Splits the cluster in two until every part fits in a posting.
void IvfIndex::write_bulk_cluster(std::vector<uint32_t> members, const std::vector<uint32_t>& item_lids,
                                  const std::vector<TypedCells>& points) {
    std::vector<std::vector<uint32_t>> work;
    work.push_back(std::move(members));
    size_t max_steps = 2 * work.back().size();
    for (size_t step = 0; !work.empty(); ++step) {
        if (step >= max_steps) {
            throw IllegalStateException("IVF index: bulk build split did not converge", VESPA_STRLOC);
        }
        std::vector<uint32_t> group = std::move(work.back());
        work.pop_back();
        if (group.size() <= _cfg.posting_capacity) {
            write_bulk_posting(group, item_lids, points);
            continue;
        }
        std::vector<TypedCells> group_points;
        group_points.reserve(group.size());
        for (uint32_t i : group) {
            group_points.push_back(points[i]);
        }
        auto                  split = split_points(group_points);
        std::vector<uint32_t> side_0;
        std::vector<uint32_t> side_1;
        for (size_t j = 0; j < group.size(); ++j) {
            (split.side[j] == 0 ? side_0 : side_1).push_back(group[j]);
        }
        assert(!side_0.empty() && !side_1.empty());
        work.push_back(std::move(side_0));
        work.push_back(std::move(side_1));
    }
}

void IvfIndex::write_bulk_posting(const std::vector<uint32_t>& members, const std::vector<uint32_t>& item_lids,
                                  const std::vector<TypedCells>& points) {
    ivf::CentroidAccumulator acc(_space);
    std::vector<char>        entries(members.size() * _entry_bytes);
    for (size_t j = 0; j < members.size(); ++j) {
        uint32_t lid = item_lids[members[j]];
        acc.add(points[members[j]]);
        encode_entry(entries.data() + j * _entry_bytes, lid, _versions[lid].load_relaxed(), points[members[j]]);
    }
    std::vector<char> centroid_bytes(_centroid_bytes);
    acc.write(centroid_bytes.data());
    if (create_posting(entries.data(), members.size(), _space.cells(centroid_bytes.data())) == no_slot) {
        drop_entries(item_lids[members.front()], members.size());
    }
}

std::unique_ptr<PrepareResult> IvfIndex::prepare_add_document(uint32_t docid, VectorBundle vectors,
                                                              GenerationGuard read_guard) const {
    auto result = std::make_unique<ivf::PreparedAdd>(docid, std::move(read_guard), _generation_handler.takeGuard());
    uint32_t subspaces = vectors.subspaces();
    result->slots.reserve(subspaces);
    for (uint32_t subspace = 0; subspace < subspaces; ++subspace) {
        result->slots.push_back(select_postings(vectors.cells(subspace)));
    }
    return result;
}

void IvfIndex::complete_add_document(uint32_t docid, std::unique_ptr<PrepareResult> prepare_result) {
    auto  vectors = _vectors.get_vectors(docid);
    auto* prepared = dynamic_cast<ivf::PreparedAdd*>(prepare_result.get());
    if (prepared != nullptr && prepared->docid == docid && prepared->slots.size() == vectors.subspaces()) {
        add_entries(docid, vectors, &prepared->slots);
    } else {
        add_entries(docid, vectors, nullptr);
    }
}

void IvfIndex::remove_document(uint32_t docid) {
    if (docid >= _versions.size()) {
        return;
    }
    auto& elem = _versions[docid];
    elem.store_release(elem.load_relaxed() + 1);
    sub_from(_counters.live_entries, _vectors.get_vectors(docid).subspaces());
}

void IvfIndex::assign_generation(Generation current_gen) {
    // The version vector transfers hold lists on reallocation, using the generation that is current after this call.
    _versions.setGeneration(current_gen + 1);
    _slot_hold.assign_generation(current_gen);
    _extent_hold.assign_generation(current_gen);
}

void IvfIndex::reclaim_memory(Generation oldest_used_gen) {
    _versions.reclaim_memory(oldest_used_gen);
    _slot_hold.reclaim(oldest_used_gen, [this](uint32_t slot) { _free_slots.push_back(slot); });
    _extent_hold.reclaim(oldest_used_gen, [this](uint32_t extent) { release_extent(extent); });
    update_extent_counters();
}

GenerationGuard IvfIndex::make_generation_read_guard() const {
    return _generation_handler.takeGuard();
}

void IvfIndex::inc_generation() {
    auto current_gen = _generation_handler.getCurrentGeneration();
    assign_generation(current_gen);
    _generation_handler.incGeneration();
}

void IvfIndex::reclaim_unused_memory() {
    _generation_handler.update_oldest_used_generation();
    reclaim_memory(_generation_handler.get_oldest_used_generation());
}

bool IvfIndex::consider_compact(const CompactionStrategy&) {
    return sweep_postings(_cfg.gc_steps);
}

// Examines up to steps postings round-robin, so every posting is checked again after about postings / steps commits.
bool IvfIndex::sweep_postings(uint32_t steps) {
    uint32_t limit = _slot_limit.load(std::memory_order_relaxed);
    bool     changed = false;
    for (uint32_t i = 0; i < std::min(steps, limit); ++i) {
        uint32_t slot = _gc_cursor;
        _gc_cursor = (_gc_cursor + 1 < limit) ? _gc_cursor + 1 : 0;
        if (slot_is_live(slot)) {
            changed = maintain_posting(slot) || changed;
        }
    }
    if (changed) {
        run_rebalance();
    }
    return changed;
}

// Merges a posting with few live entries into its neighbors, or rewrites a posting that is mostly stale entries.
bool IvfIndex::maintain_posting(uint32_t slot) {
    add_to(_counters.gc_checks, 1);
    Posting           posting = decode_ref(acquire_ref(slot));
    std::vector<char> buf;
    read_posting(posting, buf);
    std::vector<char> live;
    live.reserve(buf.size());
    for (uint32_t i = 0; i < posting.length; ++i) {
        const char* entry = buf.data() + size_t(i) * _entry_bytes;
        if (entry_is_live(entry)) {
            live.insert(live.end(), entry, entry + _entry_bytes);
        }
    }
    uint32_t count = live.size() / _entry_bytes;
    uint32_t merge_below = static_cast<uint32_t>(std::lround(_cfg.posting_capacity * _cfg.merge_fraction));
    if (count < merge_below && get(_counters.postings) > 1) {
        merge_posting(slot, live, count);
        return true;
    }
    if (count < posting.length && double(posting.length - count) >= _cfg.gc_stale_ratio * posting.length) {
        rewrite_posting(slot, posting, live, count);
        return true;
    }
    return false;
}

void IvfIndex::rewrite_posting(uint32_t slot, Posting old, const std::vector<char>& live, uint32_t count) {
    if (count == 0) {
        retire_slot(slot);
        add_to(_counters.gc_compactions, 1);
        return;
    }
    uint32_t extent = write_new_extent(live.data(), count);
    release_ref(slot, make_ref(extent, count));
    retire_extent(old.extent);
    add_to(_counters.entries, count);
    sub_from(_counters.entries, old.length);
    add_to(_counters.gc_compactions, 1);
}

// Retires the posting first so that its entries go to their nearest remaining posting.
void IvfIndex::merge_posting(uint32_t slot, const std::vector<char>& live, uint32_t count) {
    retire_slot(slot);
    add_to(_counters.merges, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const char* entry = live.data() + size_t(i) * _entry_bytes;
        auto        df = _space.bind(entry_cells(entry));
        auto        nearest = top_slots(*df, 1, no_slot, no_slot, nullptr);
        append_entry(nearest.empty() ? no_slot : nearest.front().second, entry);
        add_to(_counters.moves, 1);
    }
}

vespalib::MemoryUsage IvfIndex::update_stat(const CompactionStrategy&) {
    return memory_usage();
}

vespalib::MemoryUsage IvfIndex::memory_usage() const {
    vespalib::MemoryUsage result = _versions.getMemoryUsage();
    size_t slot_bytes = size_t(_slot_limit.load(std::memory_order_relaxed)) * (ref_bytes + _centroid_bytes);
    size_t lid_bytes = size_t(get(_counters.extents)) * _cfg.posting_capacity * sizeof(uint32_t);
    result.incAllocatedBytes(slot_bytes + lid_bytes);
    result.incUsedBytes(slot_bytes + lid_bytes);
    return result;
}

void IvfIndex::populate_address_space_usage(search::AddressSpaceUsage&) const {
}

std::unique_ptr<vespalib::StateExplorer> IvfIndex::make_state_explorer() const {
    return std::make_unique<IvfIndexExplorer>(*this);
}

void IvfIndex::shrink_lid_space(uint32_t) {
    // Versions are kept for all docids so that stale entries never become live again.
}

// Called by the writer between operations, so the reassign queue is empty and the captured state is consistent with
// the attribute snapshot taken at the same time.
std::unique_ptr<NearestNeighborIndexSaver> IvfIndex::make_saver(GenericHeader& header) {
    if (!_persistent) {
        return {};
    }
    assert(_reassign_queue.empty());
    if (!_file_ready) {
        start_new_file();
    }
    pin_extents_for_save();
    header.putTag(GenericHeader::Tag(save_file_tag(), ivf::save_format_version));
    ivf::LidReader lid_reader = [this](uint32_t extent, uint32_t length, uint32_t* lids) noexcept {
        for (uint32_t i = 0; i < length; ++i) {
            lids[i] = lid_at(extent, i);
        }
    };
    return std::make_unique<IvfIndexSaver>(capture_save_state(), std::move(lid_reader), _fd, _posting_path,
                                           _saves_completed, _saves_started);
}

std::unique_ptr<NearestNeighborIndexLoader> IvfIndex::make_loader(FastOS_FileInterface& file,
                                                                  const GenericHeader&) {
    return std::make_unique<IvfIndexLoader>([this, &file]() { load_save_file(file); });
}

bool IvfIndex::needs_build_after_load() const noexcept {
    return _load_source.load(std::memory_order_relaxed) != IvfLoadSource::SAVE_FILE;
}

std::vector<std::pair<uint64_t, uint64_t>> IvfIndex::live_posting_ranges() const {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    uint32_t                                   limit = _slot_limit.load(std::memory_order_relaxed);
    for (uint32_t slot = 0; slot < limit; ++slot) {
        uint64_t ref = acquire_ref(slot);
        if (ref != 0) {
            Posting posting = decode_ref(ref);
            ranges.emplace_back(extent_offset(posting.extent), uint64_t(posting.length) * _entry_bytes);
        }
    }
    return ranges;
}

// Scales nprobe by explore_k / k, and reads at least k_entry_factor * explore_k entries so that large targetHits
// get enough candidates.
uint32_t IvfIndex::nprobe_for_explore(uint32_t k, uint32_t explore_k) const noexcept {
    double scaled = std::ceil(double(_cfg.nprobe) * explore_k / std::max(k, 1u));
    double mean_entries = double(get(_counters.entries)) / std::max<uint64_t>(1, get(_counters.postings));
    double for_hits = std::ceil(_cfg.k_entry_factor * explore_k / std::max(1.0, mean_entries));
    return static_cast<uint32_t>(std::clamp(std::max(scaled, for_hits), double(_cfg.nprobe), double(_cfg.max_postings)));
}

std::vector<NearestNeighborIndex::Neighbor> IvfIndex::find_top_k(Stats& stats, uint32_t k,
                                                                 const BoundDistanceFunction& df, uint32_t explore_k,
                                                                 double, bool, const vespalib::Deadline& doom,
                                                                 double distance_threshold) const {
    return search(stats, k, df, df, nullptr, nprobe_for_explore(k, explore_k), doom, distance_threshold, nullptr);
}

std::vector<NearestNeighborIndex::Neighbor>
IvfIndex::find_top_k_with_filter(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                 const GlobalFilter& filter, bool, double, uint32_t explore_k, double, bool,
                                 const vespalib::Deadline& doom, double distance_threshold) const {
    return search(stats, k, df, df, filter.ptr_if_active(), nprobe_for_explore(k, explore_k), doom,
                  distance_threshold, nullptr);
}

std::vector<NearestNeighborIndex::Neighbor>
IvfIndex::find_top_k_with_nprobe(Stats& stats, uint32_t k, const BoundDistanceFunction& df,
                                 const GlobalFilter* filter, uint32_t nprobe, const vespalib::Deadline& doom,
                                 double distance_threshold, IvfProbeStats& probe_stats) const {
    const GlobalFilter* active = (filter != nullptr) ? filter->ptr_if_active() : nullptr;
    return search(stats, k, df, df, active, nprobe, doom, distance_threshold, &probe_stats);
}

std::vector<NearestNeighborIndex::Neighbor>
IvfIndex::find_top_k_with_query(Stats& stats, uint32_t k, const BoundDistanceFunction& df, TypedCells query,
                                const GlobalFilter* filter, uint32_t nprobe, const vespalib::Deadline& doom,
                                double distance_threshold, IvfProbeStats& probe_stats) const {
    const GlobalFilter* active = (filter != nullptr) ? filter->ptr_if_active() : nullptr;
    auto                centroid_df = _space.dff.for_query_vector(query);
    return search(stats, k, df, *centroid_df, active, nprobe, doom, distance_threshold, &probe_stats);
}

void IvfIndex::drop_file_cache() const {
    if (::fdatasync(_fd) != 0) {
        throw IllegalStateException(make_string("IVF index: fdatasync() failed: %s", std::strerror(errno)),
                                    VESPA_STRLOC);
    }
    int rc = ::posix_fadvise(_fd, 0, 0, POSIX_FADV_DONTNEED);
    if (rc != 0) {
        throw IllegalStateException(make_string("IVF index: posix_fadvise(DONTNEED) failed: %s", std::strerror(rc)),
                                    VESPA_STRLOC);
    }
}

uint32_t IvfIndex::check_consistency(uint32_t) const noexcept {
    return 0;
}

IvfIndexStats IvfIndex::get_stats() const noexcept {
    return {get(_counters.postings),        _slot_limit.load(std::memory_order_relaxed),
            get(_counters.entries),         get(_counters.live_entries),
            get(_counters.splits),          get(_counters.split_fallbacks),
            get(_counters.compactions),     get(_counters.reassign_checks),
            get(_counters.moves),           get(_counters.dropped_steps),
            get(_counters.dropped_entries), get(_counters.prepared_retired),
            get(_counters.extents),         get(_counters.free_extents),
            get(_counters.file_bytes),      get(_counters.bytes_written),
            get(_counters.bulk_builds),     get(_counters.gc_checks),
            get(_counters.gc_compactions),  get(_counters.merges),
            get(_counters.pinned_extents)};
}

void IvfIndex::populate_state(vespalib::slime::Cursor& object) const {
    StateExplorerUtils::memory_usage_to_slime(memory_usage(), object.setObject("memory_usage"));
    auto stats = get_stats();
    object.setLong("postings", stats.postings);
    object.setLong("slot_limit", stats.slot_limit);
    object.setLong("entries", stats.entries);
    object.setLong("live_entries", stats.live_entries);
    object.setLong("stale_entries", stats.entries > stats.live_entries ? stats.entries - stats.live_entries : 0);
    object.setLong("splits", stats.splits);
    object.setLong("split_fallbacks", stats.split_fallbacks);
    object.setLong("compactions", stats.compactions);
    object.setLong("reassign_checks", stats.reassign_checks);
    object.setLong("moves", stats.moves);
    object.setLong("dropped_steps", stats.dropped_steps);
    object.setLong("dropped_entries", stats.dropped_entries);
    object.setLong("prepared_retired", stats.prepared_retired);
    object.setLong("extents", stats.extents);
    object.setLong("free_extents", stats.free_extents);
    object.setLong("file_bytes", stats.file_bytes);
    object.setLong("bytes_written", stats.bytes_written);
    object.setLong("bulk_builds", stats.bulk_builds);
    object.setLong("gc_checks", stats.gc_checks);
    object.setLong("gc_compactions", stats.gc_compactions);
    object.setLong("merges", stats.merges);
    object.setLong("pinned_extents", stats.pinned_extents);
    object.setBool("persistent", _persistent);
    object.setString("posting_file", _posting_path);
    object.setLong("saves_completed", _saves_completed.load(std::memory_order_relaxed));
    auto& load = object.setObject("load");
    load.setString("source", load_source_name(_load_source.load(std::memory_order_relaxed)));
    load.setDouble("seconds", _load_seconds.load(std::memory_order_relaxed));
    load.setString("rejected_reason", _load_rejected);
    auto& cfg = object.setObject("cfg");
    cfg.setLong("posting_capacity", _cfg.posting_capacity);
    cfg.setLong("nprobe", _cfg.nprobe);
    cfg.setLong("reassign_range", _cfg.reassign_range);
    cfg.setLong("max_postings", _cfg.max_postings);
    cfg.setLong("max_rebalance_steps", _cfg.max_rebalance_steps);
    cfg.setString("split_mode", split_mode_name(_cfg.split_mode));
    cfg.setLong("split_iters", _cfg.split_iters);
    cfg.setDouble("split_min_side", _cfg.split_min_side);
    cfg.setLong("replicas", _cfg.replicas);
    cfg.setLong("replica_candidates", _cfg.replica_candidates);
    cfg.setDouble("replica_eps", _cfg.replica_eps);
    cfg.setDouble("rng_factor", _cfg.rng_factor);
    cfg.setBool("bulk_build", _cfg.bulk_build);
    cfg.setDouble("bulk_fill", _cfg.bulk_fill);
    cfg.setLong("bulk_sample", _cfg.bulk_sample);
    cfg.setLong("bulk_iters", _cfg.bulk_iters);
    cfg.setDouble("prune_eps", _cfg.prune_eps);
    cfg.setDouble("filter_min_ratio", _cfg.filter_min_ratio);
    cfg.setDouble("filter_max_scale", _cfg.filter_max_scale);
    cfg.setLong("bulk_coarse_probes", _cfg.bulk_coarse_probes);
    cfg.setLong("gc_steps", _cfg.gc_steps);
    cfg.setDouble("gc_stale_ratio", _cfg.gc_stale_ratio);
    cfg.setDouble("merge_fraction", _cfg.merge_fraction);
    cfg.setDouble("split_live_fraction", _cfg.split_live_fraction);
    cfg.setLong("bulk_coarse_k", _cfg.bulk_coarse_k);
    cfg.setBool("prefetch", _cfg.prefetch);
    cfg.setLong("file_grow_extents", _cfg.file_grow_extents);
    cfg.setDouble("file_grow_fraction", _cfg.file_grow_fraction);
    cfg.setLong("medoid_sample", _cfg.medoid_sample);
    cfg.setLong("medoid_rounds", _cfg.medoid_rounds);
    cfg.setString("filter_probe_mode", filter_probe_mode_name(_cfg.filter_probe_mode));
    cfg.setLong("filter_max_examined", _cfg.filter_max_examined);
    cfg.setDouble("filter_entry_read_density", _cfg.filter_entry_read_density);
    cfg.setString("filter_adaptive_stop", adaptive_stop_name(_cfg.filter_adaptive_stop));
    cfg.setDouble("filter_exact_ratio", _cfg.filter_exact_ratio);
    cfg.setDouble("k_entry_factor", _cfg.k_entry_factor);
    cfg.setBool("rebuild_on_load", _cfg.rebuild_on_load);
    cfg.setLong("entry_bytes", _entry_bytes);
}

} // namespace search::tensor
