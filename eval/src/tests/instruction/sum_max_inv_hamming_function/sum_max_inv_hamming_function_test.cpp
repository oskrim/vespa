// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/function.h>
#include <vespa/eval/eval/lazy_params.h>
#include <vespa/eval/eval/make_tensor_function.h>
#include <vespa/eval/eval/optimize_tensor_function.h>
#include <vespa/eval/eval/tensor_function.h>
#include <vespa/eval/eval/test/eval_fixture.h>
#include <vespa/eval/eval/test/gen_spec.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/eval/instruction/sum_max_inv_hamming_function.h>
#include <vespa/eval/streamed/streamed_value_builder_factory.h>
#include <vespa/vespalib/gtest/gtest.h>
#include <vespa/vespalib/util/benchmark_timer.h>

#include <cstdint>
#include <string_view>

using namespace vespalib;
using namespace vespalib::eval;
using namespace vespalib::eval::test;

const ValueBuilderFactory& prod_factory = FastValueBuilderFactory::get();
bool bench = false;

std::string main_expr = "reduce(reduce(1/(1+reduce(hamming(a,b),sum,z)),max,y),sum,x)";
std::string alt_expr = "reduce(reduce(1/(reduce(hamming(a,b),sum,z)+1),max,y),sum,x)";
std::string chunked_expr = "reduce(reduce(reduce(1/(1+reduce(hamming(a,b),sum,z)),max,y),sum,x),max,c)";

//-----------------------------------------------------------------------------

void assert_optimized(const TensorSpec& a, const TensorSpec& b, size_t vec_size,
                      const std::string& expr = main_expr) {
    EvalFixture::ParamRepo param_repo;
    param_repo.add("a", a);
    param_repo.add("b", b);
    EvalFixture slow_fixture(prod_factory, expr, param_repo, false);
    EvalFixture fast_fixture(prod_factory, expr, param_repo, true);
    EXPECT_EQ(slow_fixture.result(), EvalFixture::ref(main_expr, param_repo));
    EXPECT_EQ(fast_fixture.result(), EvalFixture::ref(main_expr, param_repo));
    auto info = fast_fixture.find_all<SumMaxInvHammingFunction>();
    if (info.size() == 1) {
        EXPECT_TRUE(info[0]->result_is_mutable());
        EXPECT_EQ(info[0]->vec_size(), vec_size);
    }
    EXPECT_EQ(info.size(), 1);
}

void assert_not_optimized(const TensorSpec& a, const TensorSpec& b, const std::string& expr = main_expr) {
    EvalFixture::ParamRepo param_repo;
    param_repo.add("a", a);
    param_repo.add("b", b);
    EvalFixture slow_fixture(prod_factory, expr, param_repo, false);
    EvalFixture fast_fixture(prod_factory, expr, param_repo, true);
    EXPECT_EQ(slow_fixture.result(), EvalFixture::ref(expr, param_repo));
    EXPECT_EQ(fast_fixture.result(), EvalFixture::ref(expr, param_repo));
    auto info = fast_fixture.find_all<SumMaxInvHammingFunction>();
    EXPECT_EQ(info.size(), 0);
}

void assert_chunked_optimized(const TensorSpec& a, const TensorSpec& b, size_t vec_size,
                              size_t chunk_dimension, const std::string& expr = chunked_expr,
                              const ValueBuilderFactory& factory = prod_factory) {
    EvalFixture::ParamRepo param_repo;
    param_repo.add("a", a);
    param_repo.add("b", b);
    EvalFixture slow_fixture(factory, expr, param_repo, false);
    EvalFixture fast_fixture(factory, expr, param_repo, true);
    EXPECT_EQ(slow_fixture.result(), EvalFixture::ref(expr, param_repo));
    EXPECT_EQ(fast_fixture.result(), EvalFixture::ref(expr, param_repo));
    auto info = fast_fixture.find_all<MaxSumMaxInvHammingFunction>();
    if (info.size() == 1) {
        EXPECT_TRUE(info[0]->result_is_mutable());
        EXPECT_EQ(info[0]->vec_size(), vec_size);
        EXPECT_EQ(info[0]->chunk_dimension(), chunk_dimension);
    }
    EXPECT_EQ(info.size(), 1);
}

void assert_chunked_not_optimized(const TensorSpec& a, const TensorSpec& b,
                                  const std::string& expr = chunked_expr) {
    EvalFixture::ParamRepo param_repo;
    param_repo.add("a", a);
    param_repo.add("b", b);
    EvalFixture fixture(prod_factory, expr, param_repo, true);
    EXPECT_EQ(fixture.result(), EvalFixture::ref(expr, param_repo));
    EXPECT_TRUE(fixture.find_all<MaxSumMaxInvHammingFunction>().empty());
}

//-----------------------------------------------------------------------------

GenSpec make_spec(const std::string& desc, CellType cell_type) {
    return GenSpec::from_desc(desc).cells(cell_type).seq(
        Seq({0x1f, 0x2e, 0x3d, 0x4c, 0x5b, 0x6a, 0x79, 0x88, 0x97, 0xa6, 0xb5, 0xc4, 0xd3, 0xe2, 0xf1}));
}

GenSpec query = make_spec("x3_1z7", CellType::INT8);
GenSpec document = make_spec("y5_1z7", CellType::INT8);

TEST(SumMaxInvHamming, expression_can_be_optimized) {
    assert_optimized(query, document, 7);
}

TEST(SumMaxInvHamming, input_values_can_be_reordered) {
    assert_optimized(document, query, 7);
}

TEST(SumMaxInvHamming, expression_can_have_alternative_form) {
    assert_optimized(query, document, 7, alt_expr);
    assert_optimized(document, query, 7, alt_expr);
}

TEST(SumMaxInvHamming, optimization_works_with_empty_tensors) {
    auto empty_query = make_spec("x0_0z7", CellType::INT8);
    auto empty_document = make_spec("y0_0z7", CellType::INT8);
    assert_optimized(empty_query, document, 7);
    assert_optimized(query, empty_document, 7);
    assert_optimized(empty_query, empty_document, 7);
}

TEST(SumMaxInvHamming, the_hamming_dimension_may_be_trivial) {
    GenSpec trivial_query = make_spec("x3_1z1", CellType::INT8);
    GenSpec trivial_document = make_spec("y5_1z1", CellType::INT8);
    assert_optimized(trivial_query, trivial_document, 1);
}

//-----------------------------------------------------------------------------

TEST(SumMaxInvHamming, other_dimensions_may_be_indexed_as_long_as_hamming_dimension_has_stride_1) {
    auto dense_query = make_spec("x3z7", CellType::INT8);
    auto dense_document = make_spec("y5z7", CellType::INT8);
    assert_optimized(dense_query, dense_document, 7);

    std::string outer_expr = "reduce(reduce(1/(1+reduce(hamming(a,b),sum,y)),max,x),sum,z)";
    auto        dense_query2 = make_spec("x3y7", CellType::INT8);
    auto        dense_document2 = make_spec("y7z5", CellType::INT8);
    assert_not_optimized(dense_query2, dense_document2);
}

//-----------------------------------------------------------------------------

TEST(SumMaxInvHamming, all_cells_must_be_int8) {
    for (auto ct : CellTypeUtils::list_types()) {
        if (ct != CellType::INT8) {
            assert_not_optimized(query.cpy().cells(ct), document);
            assert_not_optimized(query, document.cpy().cells(ct));
            assert_not_optimized(query.cpy().cells(ct), document.cpy().cells(ct));
        }
    }
}

TEST(SumMaxInvHamming, extra_dimensions_are_not_allowed) {
    GenSpec query_es = make_spec("a1_1x3_1z7", CellType::INT8);
    GenSpec query_ed = make_spec("x3_1w1z7", CellType::INT8);
    GenSpec document_es = make_spec("a1_1y5_1z7", CellType::INT8);
    GenSpec document_ed = make_spec("y5_1w1z7", CellType::INT8);
    assert_not_optimized(query_es, document);
    assert_not_optimized(query, document_es);
    assert_not_optimized(query_ed, document);
    assert_not_optimized(query, document_ed);
    assert_not_optimized(query_es, document_es);
    assert_not_optimized(query_ed, document_ed);
}

TEST(SumMaxInvHamming, chunked_expression_can_be_optimized) {
    auto chunked_document = make_spec("c2_1y5_1z7", CellType::INT8);
    assert_chunked_optimized(query, chunked_document, 7, 0);
    assert_chunked_optimized(chunked_document, query, 7, 0);
    assert_chunked_optimized(query, chunked_document, 7, 0, chunked_expr,
                             StreamedValueBuilderFactory::get());
}

TEST(SumMaxInvHamming, chunk_dimension_can_follow_token_dimension) {
    std::string expr = "reduce(reduce(reduce(1/(1+reduce(hamming(a,b),sum,z)),max,c),sum,x),max,y)";
    auto chunked_document = make_spec("c5_1y2_1z7", CellType::INT8);
    assert_chunked_optimized(query, chunked_document, 7, 1, expr);
}

TEST(SumMaxInvHamming, chunked_optimization_works_with_empty_tensors) {
    auto empty_query = make_spec("x0_0z7", CellType::INT8);
    auto chunked_document = make_spec("c2_1y5_1z7", CellType::INT8);
    auto empty_document = make_spec("c0_0y0_0z7", CellType::INT8);
    assert_chunked_optimized(empty_query, chunked_document, 7, 0);
    assert_chunked_optimized(query, empty_document, 7, 0);
    assert_chunked_optimized(empty_query, empty_document, 7, 0);
}

TEST(SumMaxInvHamming, chunked_optimization_preserves_generic_float_accumulation) {
    auto many_query_vectors = GenSpec::from_desc("x32_1z16").cells(CellType::INT8).seq(Seq({0}));
    auto one_document_vector = GenSpec::from_desc("c1_1y1_1z16")
                                   .cells(CellType::INT8)
                                   .seq(Seq({3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    EvalFixture::ParamRepo param_repo;
    param_repo.add("a", many_query_vectors);
    param_repo.add("b", one_document_vector);
    EvalFixture generic_fixture(prod_factory, chunked_expr, param_repo, false);
    EvalFixture fused_fixture(prod_factory, chunked_expr, param_repo, true);
    EXPECT_EQ(fused_fixture.result(), generic_fixture.result());
    EXPECT_EQ(fused_fixture.find_all<MaxSumMaxInvHammingFunction>().size(), 1);
}

TEST(SumMaxInvHamming, chunked_optimization_requires_exact_supported_types) {
    assert_chunked_not_optimized(query, make_spec("c2y5_1z7", CellType::INT8));
    assert_chunked_not_optimized(query, make_spec("c2_1y5z7", CellType::INT8));
    assert_chunked_not_optimized(query, make_spec("c2_1d3_1y5_1z7", CellType::INT8));
    assert_chunked_not_optimized(make_spec("x3_1z7", CellType::DOUBLE),
                                 make_spec("c2_1y5_1z7", CellType::DOUBLE));
}

TEST(SumMaxInvHamming, similar_expressions_are_not_optimized) {
    assert_not_optimized(query, document, "reduce(reduce(1*(1+reduce(hamming(a,b),sum,z)),max,y),sum,x)");
    assert_not_optimized(query, document, "reduce(reduce(1/(1-reduce(hamming(a,b),sum,z)),max,y),sum,x)");
    assert_not_optimized(query, document, "reduce(reduce(1/(1+reduce(hamming(a,b),max,z)),max,y),sum,x)");
    assert_not_optimized(query, document, "reduce(reduce(1/(1+reduce(hamming(a,b),sum,z)),sum,y),sum,x)");
    assert_not_optimized(query, document, "reduce(reduce(1/(1+reduce(hamming(a,b),sum,z)),max,y),max,x)");
    assert_not_optimized(query, document, "reduce(reduce(1/(1+reduce(hamming(a,b),sum,y)),max,z),sum,x)");
    assert_not_optimized(query, document, "reduce(reduce(1/(1+reduce(hamming(a,b),sum,x)),max,y),sum,z)");
}

struct BenchmarkFixture {
    std::shared_ptr<Function const> function;
    NodeTypes node_types;
    Stash stash;
    const TensorFunction& plain_function;
    const TensorFunction& tensor_function;
    InterpretedFunction interpreted_function;
    InterpretedFunction::Context context;
    std::vector<Value::UP> values;
    SimpleObjectParams params;

    BenchmarkFixture(const std::string& expr, const TensorSpec& query_spec, const TensorSpec& document_spec,
                     bool optimized)
        : function(Function::parse(expr)),
          node_types(*function, {ValueType::from_spec(query_spec.type()), ValueType::from_spec(document_spec.type())}),
          stash(),
          plain_function(make_tensor_function(prod_factory, function->root(), node_types, stash)),
          tensor_function(optimized ? optimize_tensor_function(prod_factory, plain_function, stash) : plain_function),
          interpreted_function(prod_factory, tensor_function),
          context(interpreted_function),
          values(),
          params({}) {
        values.push_back(value_from_spec(query_spec, prod_factory));
        values.push_back(value_from_spec(document_spec, StreamedValueBuilderFactory::get()));
        params.params.emplace_back(*values[0]);
        params.params.emplace_back(*values[1]);
    }

    double estimate_cost_ms() {
        auto evaluate = [&]() {
            const Value& result = interpreted_function.eval(context, params);
            (void)result;
        };
        return BenchmarkTimer::benchmark(evaluate, 0.25) * 1000.0;
    }

    TensorSpec result() {
        return spec_from_value(interpreted_function.eval(context, params));
    }
};

void benchmark_shape(size_t chunks, size_t tokens) {
    auto pseudo_random = [](uint32_t salt) {
        return [salt](size_t idx) noexcept {
            uint32_t value = uint32_t(idx) + salt;
            value ^= value >> 16;
            value *= 0x7feb352d;
            value ^= value >> 15;
            value *= 0x846ca68b;
            value ^= value >> 16;
            return double(int8_t(value));
        };
    };
    auto query_spec = GenSpec::from_desc("x32_1z16").cells(CellType::INT8).seq(pseudo_random(17));
    auto document_spec = GenSpec::from_desc("c" + std::to_string(chunks) + "_1y" + std::to_string(tokens) + "_1z16")
                             .cells(CellType::INT8)
                             .seq(pseudo_random(31));
    auto flat_document_spec = GenSpec::from_desc("y" + std::to_string(chunks * tokens) + "_1z16")
                                  .cells(CellType::INT8)
                                  .seq(pseudo_random(31));
    BenchmarkFixture generic(chunked_expr, query_spec, document_spec, false);
    BenchmarkFixture fused(chunked_expr, query_spec, document_spec, true);
    BenchmarkFixture flat_fused(main_expr, query_spec, flat_document_spec, true);
    EXPECT_EQ(generic.result(), fused.result());
    const double generic_ms = generic.estimate_cost_ms();
    const double fused_ms = fused.estimate_cost_ms();
    const double flat_fused_ms = flat_fused.estimate_cost_ms();
    fprintf(stderr, "chunks=%zu tokens=%zu generic=%.3f ms fused=%.3f ms flat-fused=%.3f ms speedup=%.2fx\n",
            chunks, tokens, generic_ms, fused_ms, flat_fused_ms, generic_ms / fused_ms);
}

TEST(SumMaxInvHamming, benchmark_chunked_expression) {
    if (!bench) {
        return;
    }
    benchmark_shape(8, 64);
    benchmark_shape(32, 128);
    benchmark_shape(301, 138);
}

//-----------------------------------------------------------------------------

int main(int argc, char** argv) {
    if ((argc > 1) && (std::string_view(argv[1]) == "bench")) {
        bench = true;
        ++argv;
        --argc;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
