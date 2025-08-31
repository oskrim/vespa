// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#pragma once

#include "document_field_extractor.h"
#include <vespa/document/base/documentid.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/searchlib/common/rankedhit.h>
#include <memory>
#include <vector>

namespace proton {

/**
 * Fieldpath-aware sorter that can sort search results using fieldpath expressions
 * like myMap{myKey}. This works alongside the existing attribute-based sorting
 * by providing document-aware sorting capabilities.
 */
class FieldpathSorter {
public:
    struct FieldpathSortSpec {
        std::string fieldpath_expression;
        bool ascending;
        
        FieldpathSortSpec(const std::string& expr, bool asc) 
            : fieldpath_expression(expr), ascending(asc) {}
    };
    
    /**
     * Sort hits using fieldpath expressions.
     * @param hits Array of hits to sort (modified in place)
     * @param hit_count Number of hits to sort
     * @param documents Document objects corresponding to the hits
     * @param doc_type Document type for fieldpath parsing
     * @param sort_specs Fieldpath sort specifications
     */
    static void sortByFieldpath(search::RankedHit hits[],
                                uint32_t hit_count,
                                const std::vector<document::Document::SP>& documents,
                                const document::DocumentType& doc_type,
                                const std::vector<FieldpathSortSpec>& sort_specs);

private:
    struct FieldpathSortData {
        uint32_t doc_index;
        std::vector<std::unique_ptr<document::FieldValue>> extracted_values;
        search::RankedHit original_hit;
    };
    
    static bool compareFieldpathSortData(const FieldpathSortData& a, 
                                        const FieldpathSortData& b,
                                        const std::vector<FieldpathSortSpec>& sort_specs);
};

}