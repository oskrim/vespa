// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "document_scorer.h"
#include "paged_attribute_prefetcher.h"

#include <vespa/searchlib/fef/rank_program.h>

#include <algorithm>
#include <cassert>

using search::feature_t;
using search::fef::FeatureResolver;
using search::fef::LazyValue;
using search::fef::RankProgram;
using search::queryeval::SearchIterator;

namespace proton::matching {

namespace {

LazyValue extractScoreFeature(const RankProgram& rankProgram) {
    FeatureResolver resolver(rankProgram.get_seeds());
    assert(resolver.num_features() == 1u);
    return resolver.resolve(0);
}

} // namespace

DocumentScorer::DocumentScorer(RankProgram& rankProgram, SearchIterator& searchItr,
                               const PagedAttributePrefetcher* prefetcher)
    : _searchItr(searchItr), _scoreFeature(extractScoreFeature(rankProgram)), _prefetcher(prefetcher) {
}

void DocumentScorer::score(TaggedHits& hits) {
    if (hits.empty()) {
        return;
    }
    auto sort_on_docid = [](const TaggedHit& a, const TaggedHit& b) { return (a.first.first < b.first.first); };
    std::sort(hits.begin(), hits.end(), sort_on_docid);
    // Hint the first `lookahead` hits now, then keep the hint `lookahead` hits ahead of the one being scored.
    const size_t lookahead = (_prefetcher != nullptr) ? _prefetcher->lookahead(hits.size()) : 0;
    for (size_t i = 0; i < lookahead; ++i) {
        _prefetcher->advise(hits[i].first.first);
    }
    _searchItr.initRange(hits.front().first.first, hits.back().first.first + 1);
    for (size_t i = 0; i < hits.size(); ++i) {
        if ((_prefetcher != nullptr) && (i + lookahead < hits.size())) {
            _prefetcher->advise(hits[i + lookahead].first.first);
        }
        hits[i].first.second = doScore(hits[i].first.first);
    }
}

} // namespace proton::matching
