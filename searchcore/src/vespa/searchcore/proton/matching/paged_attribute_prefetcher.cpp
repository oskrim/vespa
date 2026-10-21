// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "paged_attribute_prefetcher.h"

#include <vespa/searchcommon/attribute/iattributevector.h>
#include <vespa/searchlib/fef/blueprint.h>
#include <vespa/searchlib/fef/blueprintresolver.h>
#include <vespa/searchlib/fef/featurenameparser.h>
#include <vespa/searchlib/queryeval/irequestcontext.h>
#include <vespa/searchlib/tensor/i_tensor_attribute.h>

#include <cstdlib>
#include <set>
#include <string>
#include <string_view>

namespace proton::matching {

PagedAttributePrefetcher::PagedAttributePrefetcher(const search::queryeval::IRequestContext& request_context,
                                                   const search::fef::BlueprintResolver& resolver, uint32_t window)
    : _attributes(), _window(window) {
    std::set<std::string> names;
    for (const auto& spec : resolver.getExecutorSpecs()) {
        if (spec.blueprint->getBaseName() != "attribute") {
            continue;
        }
        search::fef::FeatureNameParser parser(spec.blueprint->getName());
        if (parser.valid() && !parser.parameters().empty()) {
            names.insert(parser.parameters()[0]);
        }
    }
    for (const auto& name : names) {
        const auto* attribute = request_context.getAttribute(name);
        const auto* tensor_attribute = (attribute != nullptr) ? attribute->asTensorAttribute() : nullptr;
        if (tensor_attribute != nullptr) {
            _attributes.push_back(tensor_attribute);
        }
    }
}

PagedAttributePrefetcher::~PagedAttributePrefetcher() = default;

bool PagedAttributePrefetcher::enabled_from_env() {
    static const bool enabled = [] {
        const char* value = std::getenv("VESPA_PREFETCH_PAGED_ATTRIBUTES");
        return (value == nullptr) || (std::string_view(value) != "0");
    }();
    return enabled;
}

uint32_t PagedAttributePrefetcher::window_from_env() {
    static const uint32_t window = [] {
        const char* value = std::getenv("VESPA_PREFETCH_PAGED_ATTRIBUTES_WINDOW");
        return (value == nullptr) ? 0u : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
    }();
    return window;
}

void PagedAttributePrefetcher::advise(uint32_t docid) const noexcept {
    for (const auto* attribute : _attributes) {
        attribute->advise_will_need(docid);
    }
}

} // namespace proton::matching
