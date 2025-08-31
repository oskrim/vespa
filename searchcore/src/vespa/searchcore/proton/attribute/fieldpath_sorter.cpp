// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "fieldpath_sorter.h"
#include <vespa/document/base/fieldpath.h>
#include <vespa/document/datatype/documenttype.h>
#include <algorithm>
#include <cassert>

using document::Document;
using document::DocumentType;
using document::FieldPath;
using document::FieldValue;
using search::RankedHit;

namespace proton {

void
FieldpathSorter::sortByFieldpath(RankedHit hits[], 
                                uint32_t hit_count,
                                const std::vector<Document::SP>& documents,
                                const DocumentType& doc_type,
                                const std::vector<FieldpathSortSpec>& sort_specs)
{
    if (hit_count == 0 || sort_specs.empty()) {
        return;
    }
    
    assert(documents.size() >= hit_count);
    
    // Create sort data with extracted fieldpath values
    std::vector<FieldpathSortData> sort_data;
    sort_data.reserve(hit_count);
    
    for (uint32_t i = 0; i < hit_count; ++i) {
        FieldpathSortData data;
        data.doc_index = i;
        data.original_hit = hits[i];
        data.extracted_values.reserve(sort_specs.size());
        
        // Extract values for each fieldpath sort spec
        DocumentFieldExtractor extractor(*documents[i]);
        
        for (const auto& spec : sort_specs) {
            FieldPath fieldPath;
            try {
                doc_type.buildFieldPath(fieldPath, spec.fieldpath_expression);
                
                if (DocumentFieldExtractor::isSupported(fieldPath)) {
                    auto extracted_value = extractor.getFieldValue(fieldPath);
                    data.extracted_values.push_back(std::move(extracted_value));
                } else {
                    // Unsupported fieldpath, use null value
                    data.extracted_values.push_back(std::unique_ptr<FieldValue>());
                }
            } catch (...) {
                // Failed to parse fieldpath, use null value
                data.extracted_values.push_back(std::unique_ptr<FieldValue>());
            }
        }
        
        sort_data.push_back(std::move(data));
    }
    
    // Sort the data
    std::sort(sort_data.begin(), sort_data.end(), 
              [&sort_specs](const FieldpathSortData& a, const FieldpathSortData& b) {
                  return compareFieldpathSortData(a, b, sort_specs);
              });
    
    // Copy sorted results back to original array
    for (uint32_t i = 0; i < hit_count; ++i) {
        hits[i] = sort_data[i].original_hit;
    }
}

bool
FieldpathSorter::compareFieldpathSortData(const FieldpathSortData& a, 
                                         const FieldpathSortData& b,
                                         const std::vector<FieldpathSortSpec>& sort_specs)
{
    for (size_t i = 0; i < sort_specs.size(); ++i) {
        const auto& spec = sort_specs[i];
        const auto& value_a = a.extracted_values[i];
        const auto& value_b = b.extracted_values[i];
        
        // Handle null values (missing fieldpath values)
        if (!value_a && !value_b) {
            continue; // Both null, equal
        }
        if (!value_a) {
            return !spec.ascending; // Null sorts last in ascending, first in descending
        }
        if (!value_b) {
            return spec.ascending; // Null sorts last in ascending, first in descending
        }
        
        // Compare actual values
        int cmp = value_a->compare(*value_b);
        if (cmp != 0) {
            return spec.ascending ? (cmp < 0) : (cmp > 0);
        }
    }
    
    // All sort fields are equal, maintain stable sort by using original order
    return a.doc_index < b.doc_index;
}

}