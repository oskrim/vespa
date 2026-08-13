// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace search::common {
struct FieldSortSpec;
}

namespace search::attribute {

class IAttributeVector;
class ISortBlobWriter;

struct MapSortField {
    std::string map_name;
    std::string key;
};

std::optional<MapSortField> parse_map_sort_field(std::string_view field);

std::unique_ptr<ISortBlobWriter> make_sort_blob_writer(const IAttributeVector*              vector,
                                                       const search::common::FieldSortSpec& field_sort_spec);

std::unique_ptr<ISortBlobWriter>
make_map_value_sort_blob_writer(const IAttributeVector* key_vector, const IAttributeVector* value_vector,
                                std::string_view key, const search::common::FieldSortSpec& field_sort_spec);

} // namespace search::attribute
