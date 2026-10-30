// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace search::tensor::ivf {

constexpr uint32_t page_bytes = 4096;

/**
 * Posting size classes in whole 4 KiB pages, so every posting starts on a page. A class holds
 * min(max_entries, pages * page_bytes / entry_bytes) entries. Page counts grow by about 2^(1/per_doubling) per class,
 * from the pages of min_entries up to the pages of max_entries; per_doubling 0 gives one class.
 */
class SizeClasses {
public:
    static constexpr uint32_t max_classes = 255;

    SizeClasses(uint32_t entry_bytes, uint32_t max_entries, uint32_t min_entries, uint32_t per_doubling);

    uint32_t count() const noexcept { return _pages.size(); }
    uint32_t largest() const noexcept { return count() - 1; }
    uint32_t pages(uint32_t cls) const noexcept { return _pages[cls]; }
    uint32_t capacity(uint32_t cls) const noexcept { return _capacity[cls]; }
    // Smallest class with room for max(n + 1, ceil(n * (1 + headroom))) entries, or the largest class.
    uint32_t fit(uint32_t n, double headroom) const noexcept;
    // First lid table slot of a posting at this page. Postings with disjoint pages get disjoint lid slots.
    uint64_t first_lid_slot(uint32_t page) const noexcept { return uint64_t(page) * page_bytes / _entry_bytes; }

private:
    uint32_t              _entry_bytes;
    std::vector<uint32_t> _pages;
    std::vector<uint32_t> _capacity;
};

/**
 * Best-fit allocator of page ranges below a high-water mark (limit). Freed ranges merge with free neighbors, and a
 * free range that ends at the limit lowers it. Ties between equally sized free ranges go to the lowest page.
 */
class PageAllocator {
public:
    PageAllocator();
    ~PageAllocator();

    // First page of a new range, or nullopt when no free range fits and the limit would pass max_limit.
    std::optional<uint32_t> allocate(uint32_t pages, uint64_t max_limit);
    void free(uint32_t page, uint32_t pages);
    // Restarts with the used ranges (page, pages), which must be disjoint: the gaps between them are free and the
    // limit is the end of the last one.
    void reset(std::vector<std::pair<uint32_t, uint32_t>> used);

    uint32_t limit() const noexcept { return _limit; }
    uint64_t free_pages() const noexcept { return _free_pages; }
    size_t free_ranges() const noexcept { return _by_page.size(); }
    // (first page, pages) of every free range, in page order.
    std::vector<std::pair<uint32_t, uint32_t>> free_list() const;
    // Empty when the invariants hold, else the first one broken. Linear time; for tests.
    const char* check() const;

private:
    void insert_free(uint32_t page, uint32_t pages);
    void erase_free(std::map<uint32_t, uint32_t>::iterator it);

    std::map<uint32_t, uint32_t>            _by_page; // first page -> pages
    std::set<std::pair<uint32_t, uint32_t>> _by_size; // (pages, first page)
    uint32_t                                _limit;
    uint64_t                                _free_pages;
};

} // namespace search::tensor::ivf
