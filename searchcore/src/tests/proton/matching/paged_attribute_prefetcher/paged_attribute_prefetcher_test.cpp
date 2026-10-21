// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/searchcore/proton/matching/paged_attribute_prefetcher.h>
#include <vespa/searchlib/fef/blueprintfactory.h>
#include <vespa/searchlib/fef/blueprintresolver.h>
#include <vespa/searchlib/fef/test/indexenvironment.h>
#include <vespa/searchlib/queryeval/fake_requestcontext.h>
#include <vespa/vespalib/gtest/gtest.h>

using proton::matching::PagedAttributePrefetcher;
using search::fef::BlueprintFactory;
using search::fef::BlueprintResolver;
using search::queryeval::FakeRequestContext;

namespace {

struct Fixture {
    BlueprintFactory                    factory;
    search::fef::test::IndexEnvironment index_env;
    BlueprintResolver                   resolver;
    FakeRequestContext                  request_context;
    Fixture() : factory(), index_env(), resolver(factory, index_env), request_context() {}
    PagedAttributePrefetcher make(uint32_t window) { return {request_context, resolver, window}; }
};

} // namespace

TEST(PagedAttributePrefetcherTest, configured_window_is_off_by_default) {
    EXPECT_EQ(-1, PagedAttributePrefetcher::configured_window());
}

TEST(PagedAttributePrefetcherTest, configured_window_follows_the_latest_setting) {
    PagedAttributePrefetcher::set_configured_window(32);
    EXPECT_EQ(32, PagedAttributePrefetcher::configured_window());
    PagedAttributePrefetcher::set_configured_window(-1);
    EXPECT_EQ(-1, PagedAttributePrefetcher::configured_window());
}

TEST(PagedAttributePrefetcherTest, window_zero_hints_all_hits_up_front) {
    Fixture f;
    auto    prefetcher = f.make(0);
    EXPECT_EQ(100u, prefetcher.lookahead(100));
    EXPECT_EQ(0u, prefetcher.lookahead(0));
}

TEST(PagedAttributePrefetcherTest, window_limits_the_lookahead) {
    Fixture f;
    auto    prefetcher = f.make(32);
    EXPECT_EQ(32u, prefetcher.lookahead(100));
    EXPECT_EQ(10u, prefetcher.lookahead(10));
}

TEST(PagedAttributePrefetcherTest, rank_program_without_attribute_features_has_nothing_to_hint) {
    Fixture f;
    EXPECT_TRUE(f.make(0).empty());
}

GTEST_MAIN_RUN_ALL_TESTS()
