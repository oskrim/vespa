// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "ivf_page_allocator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>

namespace search::tensor::ivf {

SizeClasses::SizeClasses(uint32_t entry_bytes, uint32_t max_entries, uint32_t min_entries, uint32_t per_doubling)
    : _entry_bytes(entry_bytes), _pages(), _capacity() {
    assert(entry_bytes > 0 && max_entries > 0);
    auto pages_for = [entry_bytes](uint32_t entries) {
        return static_cast<uint32_t>((uint64_t(entries) * entry_bytes + page_bytes - 1) / page_bytes);
    };
    auto capacity_of = [entry_bytes, max_entries](uint32_t pages) {
        return static_cast<uint32_t>(std::min<uint64_t>(max_entries, uint64_t(pages) * page_bytes / entry_bytes));
    };
    const uint32_t last = pages_for(max_entries);
    const double   step = (per_doubling == 0) ? 1.0 : std::exp2(1.0 / per_doubling);
    uint32_t       pages = (per_doubling == 0) ? last : pages_for(std::clamp(min_entries, 1u, max_entries));
    while (true) {
        uint32_t capacity = capacity_of(pages);
        if (capacity > 0 && (_capacity.empty() || capacity > _capacity.back())) {
            _pages.push_back(pages);
            _capacity.push_back(capacity);
        }
        if (pages >= last) {
            break;
        }
        pages = std::min(last, std::max(pages + 1, static_cast<uint32_t>(std::lround(pages * step))));
    }
    assert(!_pages.empty() && _pages.size() <= max_classes && _capacity.back() == max_entries);
}

uint32_t SizeClasses::fit(uint32_t n, double headroom) const noexcept {
    uint64_t want = std::max<uint64_t>(uint64_t(n) + 1, static_cast<uint64_t>(std::ceil(n * (1.0 + headroom))));
    auto     it = std::lower_bound(_capacity.begin(), _capacity.end(), want);
    return (it == _capacity.end()) ? largest() : static_cast<uint32_t>(it - _capacity.begin());
}

PageAllocator::PageAllocator() : _by_page(), _by_size(), _limit(0), _free_pages(0) {
}

PageAllocator::~PageAllocator() = default;

void PageAllocator::insert_free(uint32_t page, uint32_t pages) {
    _by_page.emplace(page, pages);
    _by_size.emplace(pages, page);
    _free_pages += pages;
}

void PageAllocator::erase_free(std::map<uint32_t, uint32_t>::iterator it) {
    _by_size.erase({it->second, it->first});
    _free_pages -= it->second;
    _by_page.erase(it);
}

std::optional<uint32_t> PageAllocator::allocate(uint32_t pages, uint64_t max_limit) {
    assert(pages > 0);
    auto best = _by_size.lower_bound({pages, 0});
    if (best != _by_size.end()) {
        auto [size, page] = *best;
        erase_free(_by_page.find(page));
        if (size > pages) {
            insert_free(page + pages, size - pages);
        }
        return page;
    }
    if (uint64_t(_limit) + pages > max_limit) {
        return std::nullopt;
    }
    uint32_t page = _limit;
    _limit += pages;
    return page;
}

void PageAllocator::free(uint32_t page, uint32_t pages) {
    assert(pages > 0 && uint64_t(page) + pages <= _limit);
    uint32_t start = page;
    uint32_t end = page + pages;
    auto     next = _by_page.lower_bound(page);
    assert(next == _by_page.end() || next->first >= end);
    if (next != _by_page.begin()) {
        auto prev = std::prev(next);
        assert(prev->first + prev->second <= start);
        if (prev->first + prev->second == start) {
            start = prev->first;
            erase_free(prev);
        }
    }
    if (next != _by_page.end() && next->first == end) {
        end += next->second;
        erase_free(next);
    }
    if (end == _limit) {
        _limit = start;
    } else {
        insert_free(start, end - start);
    }
}

void PageAllocator::reset(std::vector<std::pair<uint32_t, uint32_t>> used) {
    _by_page.clear();
    _by_size.clear();
    _free_pages = 0;
    std::sort(used.begin(), used.end());
    uint32_t cursor = 0;
    for (auto [page, pages] : used) {
        assert(page >= cursor && pages > 0);
        if (page > cursor) {
            insert_free(cursor, page - cursor);
        }
        cursor = page + pages;
    }
    _limit = cursor;
}

const char* PageAllocator::check() const {
    if (_by_page.size() != _by_size.size()) {
        return "the two free indexes differ in size";
    }
    uint64_t total = 0;
    uint64_t prev_end = 0;
    bool     first = true;
    for (auto [page, pages] : _by_page) {
        if (pages == 0) {
            return "empty free range";
        }
        if (!first && page <= prev_end) {
            return "free ranges overlap or touch";
        }
        if (uint64_t(page) + pages >= _limit) {
            return "a free range reaches the limit";
        }
        if (!_by_size.contains({pages, page})) {
            return "a free range is missing from the size index";
        }
        total += pages;
        prev_end = uint64_t(page) + pages;
        first = false;
    }
    return (total == _free_pages) ? "" : "free page count is wrong";
}

} // namespace search::tensor::ivf
