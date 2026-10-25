# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(searchlib_ivf_index_test_app TEST
    SOURCES
    ivf_index_test.cpp
    DEPENDS
    searchlib_test
    vespa_searchlib
    GTest::gtest
)
vespa_add_test(NAME searchlib_ivf_index_test_app COMMAND searchlib_ivf_index_test_app)

vespa_add_executable(searchlib_ivf_index_bench_app
    SOURCES
    ivf_index_bench.cpp
    DEPENDS
    vespa_searchlib
)
