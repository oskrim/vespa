// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "make_sort_blob_writer.h"

#include "numeric_sort_blob_writer.h"
#include "string_sort_blob_writer.h"
#include "string_to_number.h"

#include <vespa/searchcommon/attribute/i_sort_blob_writer.h>
#include <vespa/searchcommon/attribute/iattributevector.h>
#include <vespa/vespalib/util/exceptions.h>
#include <vespa/vespalib/util/issue.h>

#include <vector>

using search::attribute::BasicType;
using search::common::FieldSortSpec;
using vespalib::IllegalArgumentException;
using vespalib::Issue;

namespace search::attribute {

namespace {

std::optional<std::string> decode_map_key(std::string_view encoded) {
    if (encoded.empty()) {
        return {};
    }
    if (encoded.front() != '"') {
        return encoded.find('"') == std::string_view::npos ? std::optional<std::string>(encoded) : std::nullopt;
    }
    if (encoded.size() < 2 || encoded.back() != '"') {
        return {};
    }
    std::string key;
    key.reserve(encoded.size() - 2);
    for (size_t i = 1; i + 1 < encoded.size(); ++i) {
        char c = encoded[i];
        if (c == '\\') {
            if (++i + 1 >= encoded.size() || (encoded[i] != '\\' && encoded[i] != '"')) {
                return {};
            }
            c = encoded[i];
        } else if (c == '"') {
            return {};
        }
        key.push_back(c);
    }
    return key;
}

std::optional<size_t> find_key(const IAttributeVector& keys, uint32_t docid, std::string_view selected_key,
                               std::vector<const char*>& values) {
    if (docid >= keys.getNumDocs()) {
        return {};
    }
    const uint32_t count = keys.getValueCount(docid);
    values.resize(count);
    if (keys.get(docid, values.data(), count) != count) {
        return {};
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] != nullptr && selected_key == values[i]) {
            return i;
        }
    }
    return {};
}

template <typename T, typename ReadType, bool ascending>
class NumericMapValueSortBlobWriter final : public ISortBlobWriter {
    const IAttributeVector&             _keys;
    const IAttributeVector&             _values;
    std::string                         _selected_key;
    NumericSortBlobWriter<T, ascending> _writer;
    std::vector<const char*>             _key_buffer;
    std::vector<ReadType>                _value_buffer;

public:
    NumericMapValueSortBlobWriter(const IAttributeVector& keys, const IAttributeVector& values,
                                  std::string_view selected_key, const FieldSortSpec& spec)
        : _keys(keys),
          _values(values),
          _selected_key(selected_key),
          _writer(spec._missing_policy, search::string_to_number<T>(spec._missing_value), false),
          _key_buffer(),
          _value_buffer() {
    }

    long write(uint32_t docid, void* buf, long available) override {
        _writer.reset();
        if (docid < _values.getNumDocs()) {
            const auto index = find_key(_keys, docid, _selected_key, _key_buffer);
            const uint32_t value_count = _values.getValueCount(docid);
            if (index.has_value() && _key_buffer.size() == value_count) {
                _value_buffer.resize(value_count);
                if (_values.get(docid, _value_buffer.data(), value_count) == value_count) {
                    _writer.candidate(static_cast<T>(_value_buffer[*index]));
                }
            }
        }
        return available < 0 ? -1 : _writer.write(buf, static_cast<size_t>(available));
    }
};

template <bool ascending>
class StringMapValueSortBlobWriter final : public ISortBlobWriter {
    const IAttributeVector&           _keys;
    const IAttributeVector&           _values;
    std::string                       _selected_key;
    StringSortBlobWriter<ascending> _writer;
    std::vector<const char*>          _key_buffer;
    std::vector<const char*>          _value_buffer;

public:
    StringMapValueSortBlobWriter(const IAttributeVector& keys, const IAttributeVector& values,
                                 std::string_view selected_key, const FieldSortSpec& spec)
        : _keys(keys),
          _values(values),
          _selected_key(selected_key),
          _writer(spec._converter.get(), spec._missing_policy, spec._missing_value, false),
          _key_buffer(),
          _value_buffer() {
    }

    long write(uint32_t docid, void* buf, long available) override {
        if (available < 0) {
            return -1;
        }
        _writer.reset(buf, static_cast<size_t>(available));
        if (docid < _values.getNumDocs()) {
            const auto index = find_key(_keys, docid, _selected_key, _key_buffer);
            const uint32_t value_count = _values.getValueCount(docid);
            if (index.has_value() && _key_buffer.size() == value_count) {
                _value_buffer.resize(value_count);
                if (_values.get(docid, _value_buffer.data(), value_count) == value_count &&
                    !_writer.candidate(_value_buffer[*index])) {
                    return -1;
                }
            }
        }
        return _writer.write();
    }
};

template <typename T, typename ReadType>
std::unique_ptr<ISortBlobWriter> make_numeric_map_writer(const IAttributeVector& keys,
                                                         const IAttributeVector& values,
                                                         std::string_view selected_key,
                                                         const FieldSortSpec& spec) {
    if (spec.is_ascending()) {
        return std::make_unique<NumericMapValueSortBlobWriter<T, ReadType, true>>(keys, values, selected_key, spec);
    }
    return std::make_unique<NumericMapValueSortBlobWriter<T, ReadType, false>>(keys, values, selected_key, spec);
}

} // namespace

std::optional<MapSortField> parse_map_sort_field(std::string_view field) {
    const size_t open = field.find('{');
    const size_t close = field.rfind('}');
    if (open == std::string_view::npos || open == 0 || close != field.size() - 1 ||
        field.find('{', open + 1) != std::string_view::npos || field.find('}', open) != close) {
        return {};
    }
    auto key = decode_map_key(field.substr(open + 1, close - open - 1));
    if (!key.has_value() || key->empty()) {
        return {};
    }
    return MapSortField{std::string(field.substr(0, open)), std::move(*key)};
}

std::unique_ptr<search::attribute::ISortBlobWriter> make_sort_blob_writer(const IAttributeVector* vector,
                                                                          const FieldSortSpec&    field_sort_spec) {
    if (vector == nullptr) {
        return {};
    }
    try {
        return vector->make_sort_blob_writer(field_sort_spec.is_ascending(), field_sort_spec._converter.get(),
                                             field_sort_spec._missing_policy, field_sort_spec._missing_value);
    } catch (const IllegalArgumentException& e) {
        Issue::report("make_sort_blob_writer: For attribute vector %s (basic type %s): %s", vector->getName().c_str(),
                      BasicType(vector->getBasicType()).asString(), e.message());
        return {};
    }
}

std::unique_ptr<ISortBlobWriter>
make_map_value_sort_blob_writer(const IAttributeVector* key_vector, const IAttributeVector* value_vector,
                                std::string_view key, const FieldSortSpec& field_sort_spec) {
    if (key_vector == nullptr || value_vector == nullptr || !key_vector->isStringType() ||
        key_vector->getCollectionType() != CollectionType::ARRAY ||
        value_vector->getCollectionType() != CollectionType::ARRAY || key.empty()) {
        return {};
    }
    try {
        switch (value_vector->getBasicType()) {
        case BasicType::BOOL:
        case BasicType::UINT2:
        case BasicType::UINT4:
        case BasicType::INT8:
            return make_numeric_map_writer<int8_t, IAttributeVector::largeint_t>(*key_vector, *value_vector, key,
                                                                                 field_sort_spec);
        case BasicType::INT16:
            return make_numeric_map_writer<int16_t, IAttributeVector::largeint_t>(*key_vector, *value_vector, key,
                                                                                  field_sort_spec);
        case BasicType::INT32:
            return make_numeric_map_writer<int32_t, IAttributeVector::largeint_t>(*key_vector, *value_vector, key,
                                                                                  field_sort_spec);
        case BasicType::INT64:
            return make_numeric_map_writer<int64_t, IAttributeVector::largeint_t>(*key_vector, *value_vector, key,
                                                                                  field_sort_spec);
        case BasicType::FLOAT:
            return make_numeric_map_writer<float, double>(*key_vector, *value_vector, key, field_sort_spec);
        case BasicType::DOUBLE:
            return make_numeric_map_writer<double, double>(*key_vector, *value_vector, key, field_sort_spec);
        case BasicType::STRING:
            if (field_sort_spec.is_ascending()) {
                return std::make_unique<StringMapValueSortBlobWriter<true>>(*key_vector, *value_vector, key,
                                                                           field_sort_spec);
            }
            return std::make_unique<StringMapValueSortBlobWriter<false>>(*key_vector, *value_vector, key,
                                                                        field_sort_spec);
        default:
            return {};
        }
    } catch (const IllegalArgumentException& e) {
        Issue::report("make_map_value_sort_blob_writer: For attribute vector %s (basic type %s): %s",
                      value_vector->getName().c_str(), BasicType(value_vector->getBasicType()).asString(), e.message());
        return {};
    }
}

} // namespace search::attribute
