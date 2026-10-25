// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <string>
#include <utility>
#include <vector>

namespace search::attribute {

/**
 * Parameters of the experimental disk-based IVF index, as the name/value pairs given in the schema's 'ivf' index
 * block (kebab case names, e.g. "posting-capacity"). The index validates names and values when it is created.
 */
class IvfIndexParams {
public:
    using Params = std::vector<std::pair<std::string, std::string>>;

private:
    Params _params;

public:
    IvfIndexParams() noexcept : _params() {}
    explicit IvfIndexParams(Params params) noexcept : _params(std::move(params)) {}
    [[nodiscard]] const Params& params() const noexcept { return _params; }
    bool operator==(const IvfIndexParams& rhs) const noexcept = default;
};

} // namespace search::attribute
