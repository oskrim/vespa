// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <cstdint>
#include <vector>

namespace search::fef {
class BlueprintResolver;
}
namespace search::queryeval {
class IRequestContext;
}
namespace search::tensor {
class ITensorAttribute;
}

namespace proton::matching {

/**
 * Hints the kernel to start reading the tensors of documents that are about to be ranked, for the tensor
 * attributes a rank program reads with attribute(name). Paged (mmap'ed) tensor attributes are otherwise read one
 * page fault at a time while documents are scored one by one.
 */
class PagedAttributePrefetcher {
    std::vector<const search::tensor::ITensorAttribute*> _attributes;
    uint32_t                                             _window;

public:
    /**
     * @param window how many documents to hint ahead of the one being scored; 0 hints all documents up front.
     */
    PagedAttributePrefetcher(const search::queryeval::IRequestContext& request_context,
                             const search::fef::BlueprintResolver& resolver, uint32_t window);
    ~PagedAttributePrefetcher();

    // Prototype switches: VESPA_PREFETCH_PAGED_ATTRIBUTES=0 turns prefetch off,
    // VESPA_PREFETCH_PAGED_ATTRIBUTES_WINDOW sets the window (unset or 0: all hits up front).
    static bool     enabled_from_env();
    static uint32_t window_from_env();

    bool empty() const noexcept { return _attributes.empty(); }

    // Number of documents to hint before scoring the first of num_docs documents.
    uint32_t lookahead(uint32_t num_docs) const noexcept {
        return ((_window == 0) || (_window > num_docs)) ? num_docs : _window;
    }
    void advise(uint32_t docid) const noexcept;
};

} // namespace proton::matching
