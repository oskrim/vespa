// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "sum_max_inv_hamming_function.h"

#include <vespa/eval/eval/value.h>
#include <vespa/vespalib/util/binary_hamming_distance.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace vespalib::eval {

using namespace tensor_function;
using namespace operation;

namespace {

void my_sum_max_inv_hamming_op(InterpretedFunction::State& state, uint64_t vec_size) {
    double result = 0.0;
    auto   query_cells = state.peek(1).cells().unsafe_typify<int8_t>();
    auto   document_cells = state.peek(0).cells().unsafe_typify<int8_t>();
    if ((query_cells.size() > 0) && (document_cells.size() > 0)) {
        for (const int8_t* query = query_cells.data(); query < query_cells.data() + query_cells.size();
             query += vec_size)
        {
            float max_inv_hamming = aggr::Max<float>::null_value();
            for (const int8_t* document = document_cells.data();
                 document < document_cells.data() + document_cells.size(); document += vec_size)
            {
                float my_inv_hamming = 1.0f / (1.0f + binary_hamming_distance(query, document, vec_size));
                max_inv_hamming = aggr::Max<float>::combine(max_inv_hamming, my_inv_hamming);
            }
            result += max_inv_hamming;
        }
    }
    state.pop_pop_push(state.stash.create<DoubleValue>(result));
}

struct ChunkedParams {
    uint32_t vec_size;
    uint32_t chunk_dimension;
};

void my_max_sum_max_inv_hamming_op(InterpretedFunction::State& state, uint64_t param_value) {
    const auto params = std::bit_cast<ChunkedParams>(param_value);
    const auto query_cells = state.peek(1).cells().unsafe_typify<int8_t>();
    const auto& document = state.peek(0);
    const auto document_cells = document.cells().unsafe_typify<int8_t>();
    assert(params.vec_size > 0);
    assert((query_cells.size() % params.vec_size) == 0);
    assert((document_cells.size() % params.vec_size) == 0);
    assert(document.index().size() == (document_cells.size() / params.vec_size));
    const size_t query_vectors = query_cells.size() / params.vec_size;
    const size_t max_distance = params.vec_size * 8;
    if (query_vectors == 0) {
        state.pop_pop_push(state.stash.create<DoubleValue>(0.0));
        return;
    }
    float result = aggr::Max<float>::null_value();
    std::unordered_map<uint32_t, size_t> chunk_indexes;
    std::vector<size_t> best_distances;
    size_t num_chunks = 0;

    auto view = document.index().create_view({});
    std::array<string_id, 2> address;
    std::array<string_id*, 2> address_refs = {&address[0], &address[1]};
    size_t subspace = 0;
    view->lookup({});
    while (view->next_result(address_refs, subspace)) {
        auto [pos, inserted] = chunk_indexes.try_emplace(address[params.chunk_dimension].value(), num_chunks);
        if (inserted) {
            ++num_chunks;
            assert(num_chunks <= (best_distances.max_size() / query_vectors));
            best_distances.resize(num_chunks * query_vectors, max_distance);
        }
        const int8_t* document_vector = document_cells.data() + (subspace * params.vec_size);
        size_t* chunk_distances = best_distances.data() + (pos->second * query_vectors);
        for (size_t query_idx = 0; query_idx < query_vectors; ++query_idx) {
            if (chunk_distances[query_idx] != 0) {
                const int8_t* query_vector = query_cells.data() + (query_idx * params.vec_size);
                chunk_distances[query_idx] = std::min(chunk_distances[query_idx],
                                                      binary_hamming_distance(query_vector, document_vector,
                                                                              params.vec_size));
            }
        }
    }
    if (num_chunks == 0) {
        result = 0.0f;
    }
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        float chunk_score = 0.0f;
        const size_t* chunk_distances = best_distances.data() + (chunk_idx * query_vectors);
        for (size_t query_idx = 0; query_idx < query_vectors; ++query_idx) {
            chunk_score += 1.0f / (1.0f + chunk_distances[query_idx]);
        }
        result = aggr::Max<float>::combine(result, chunk_score);
    }
    state.pop_pop_push(state.stash.create<DoubleValue>(result));
}

const Reduce* check_reduce(const TensorFunction& expr, Aggr aggr) {
    if (auto reduce = as<Reduce>(expr)) {
        if ((reduce->aggr() == aggr) && (reduce->dimensions().size() == 1)) {
            return reduce;
        }
    }
    return nullptr;
}

const Join* check_join(const TensorFunction& expr, op2_t op) {
    if (auto join = as<Join>(expr)) {
        if (join->function() == op) {
            return join;
        }
    }
    return nullptr;
}

bool is_one(const TensorFunction& expr) {
    if (expr.result_type().is_double()) {
        if (auto const_value = as<ConstValue>(expr)) {
            return (const_value->value().as_double() == 1.0);
        }
    }
    return false;
}

// 1/(1+x) -> x
// 1/(x+1) -> x
const TensorFunction* check_inv(const TensorFunction& expr) {
    if (auto div = check_join(expr, Div::f)) {
        if (is_one(div->lhs())) {
            if (auto add = check_join(div->rhs(), Add::f)) {
                if (is_one(add->lhs())) {
                    return &add->rhs();
                }
                if (is_one(add->rhs())) {
                    return &add->lhs();
                }
            }
        }
    }
    return nullptr;
}

bool check_params(const ValueType& res_type, const ValueType& query, const ValueType& document,
                  const std::string& sum_dim, const std::string& max_dim, const std::string& ham_dim) {
    return (res_type.is_double() && (query.dimensions().size() == 2) && (query.cell_type() == CellType::INT8) &&
            (document.dimensions().size() == 2) && (document.cell_type() == CellType::INT8) &&
            query.has_dimension(sum_dim) && (query.stride_of(ham_dim) == 1) && document.has_dimension(max_dim) &&
            (document.stride_of(ham_dim) == 1));
}

bool check_chunked_params(const ValueType& res_type, const ValueType& query, const ValueType& document,
                          const std::string& chunk_dim, const std::string& sum_dim, const std::string& max_dim,
                          const std::string& ham_dim) {
    return (res_type.is_double() && (query.dimensions().size() == 2) && (query.cell_type() == CellType::INT8) &&
            (document.dimensions().size() == 3) && (document.cell_type() == CellType::INT8) &&
            query.has_dimension(sum_dim) && (query.stride_of(ham_dim) == 1) &&
            document.has_dimension(chunk_dim) && document.has_dimension(max_dim) &&
            document.dimensions()[document.dimension_index(chunk_dim)].is_mapped() &&
            document.dimensions()[document.dimension_index(max_dim)].is_mapped() &&
            (document.stride_of(ham_dim) == 1));
}

size_t get_dim_size(const ValueType& type, const std::string& dim) {
    size_t npos = ValueType::Dimension::npos;
    size_t idx = type.dimension_index(dim);
    assert(idx != npos);
    return type.dimensions()[idx].size;
}

size_t get_mapped_dim_index(const ValueType& type, const std::string& dim) {
    const auto mapped_dimensions = type.mapped_dimensions();
    for (size_t idx = 0; idx < mapped_dimensions.size(); ++idx) {
        if (mapped_dimensions[idx].name == dim) {
            return idx;
        }
    }
    abort();
}

const TensorFunction* try_optimize_chunked(const TensorFunction& expr, Stash& stash) {
    if (auto chunk_reduce = check_reduce(expr, Aggr::MAX)) {
        if (auto sum_reduce = check_reduce(chunk_reduce->child(), Aggr::SUM)) {
            if (auto max_reduce = check_reduce(sum_reduce->child(), Aggr::MAX)) {
                if (auto inverted = check_inv(max_reduce->child())) {
                    if (auto ham_reduce = check_reduce(*inverted, Aggr::SUM)) {
                        if (auto ham = check_join(ham_reduce->child(), Hamming::f)) {
                            const auto& chunk_dim = chunk_reduce->dimensions()[0];
                            const auto& sum_dim = sum_reduce->dimensions()[0];
                            const auto& max_dim = max_reduce->dimensions()[0];
                            const auto& ham_dim = ham_reduce->dimensions()[0];
                            if (check_chunked_params(expr.result_type(), ham->lhs().result_type(),
                                                     ham->rhs().result_type(), chunk_dim, sum_dim, max_dim, ham_dim)) {
                                const auto& document_type = ham->rhs().result_type();
                                return &stash.create<MaxSumMaxInvHammingFunction>(
                                    expr.result_type(), ham->lhs(), ham->rhs(),
                                    get_dim_size(ham->lhs().result_type(), ham_dim),
                                    get_mapped_dim_index(document_type, chunk_dim));
                            }
                            if (check_chunked_params(expr.result_type(), ham->rhs().result_type(),
                                                     ham->lhs().result_type(), chunk_dim, sum_dim, max_dim, ham_dim)) {
                                const auto& document_type = ham->lhs().result_type();
                                return &stash.create<MaxSumMaxInvHammingFunction>(
                                    expr.result_type(), ham->rhs(), ham->lhs(),
                                    get_dim_size(ham->rhs().result_type(), ham_dim),
                                    get_mapped_dim_index(document_type, chunk_dim));
                            }
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

} // namespace

SumMaxInvHammingFunction::SumMaxInvHammingFunction(const ValueType& res_type_in, const TensorFunction& query,
                                                   const TensorFunction& document, size_t vec_size)
    : tensor_function::Op2(res_type_in, query, document), _vec_size(vec_size) {
}

InterpretedFunction::Instruction SumMaxInvHammingFunction::compile_self(const ValueBuilderFactory&, Stash&) const {
    return InterpretedFunction::Instruction(my_sum_max_inv_hamming_op, _vec_size);
}

MaxSumMaxInvHammingFunction::MaxSumMaxInvHammingFunction(const ValueType& res_type_in, const TensorFunction& query,
                                                         const TensorFunction& document, size_t vec_size,
                                                         size_t chunk_dimension)
    : tensor_function::Op2(res_type_in, query, document),
      _vec_size(vec_size),
      _chunk_dimension(chunk_dimension) {
    assert(_vec_size > 0);
    assert(_vec_size <= std::numeric_limits<uint32_t>::max());
    assert(_chunk_dimension <= std::numeric_limits<uint32_t>::max());
}

InterpretedFunction::Instruction
MaxSumMaxInvHammingFunction::compile_self(const ValueBuilderFactory&, Stash&) const {
    static_assert(sizeof(ChunkedParams) == sizeof(uint64_t));
    const auto params = std::bit_cast<uint64_t>(ChunkedParams{uint32_t(_vec_size), uint32_t(_chunk_dimension)});
    return InterpretedFunction::Instruction(my_max_sum_max_inv_hamming_op, params);
}

const TensorFunction& SumMaxInvHammingFunction::optimize(const TensorFunction& expr, Stash& stash) {
    if (const TensorFunction* chunked = try_optimize_chunked(expr, stash)) {
        return *chunked;
    }
    if (auto sum_reduce = check_reduce(expr, Aggr::SUM)) {
        if (auto max_reduce = check_reduce(sum_reduce->child(), Aggr::MAX)) {
            if (auto inverted = check_inv(max_reduce->child())) {
                if (auto ham_reduce = check_reduce(*inverted, Aggr::SUM)) {
                    if (auto ham = check_join(ham_reduce->child(), Hamming::f)) {
                        const auto& sum_dim = sum_reduce->dimensions()[0];
                        const auto& max_dim = max_reduce->dimensions()[0];
                        const auto& ham_dim = ham_reduce->dimensions()[0];
                        if (check_params(expr.result_type(), ham->lhs().result_type(), ham->rhs().result_type(),
                                         sum_dim, max_dim, ham_dim))
                        {
                            size_t vec_size = get_dim_size(ham->lhs().result_type(), ham_dim);
                            return stash.create<SumMaxInvHammingFunction>(expr.result_type(), ham->lhs(), ham->rhs(),
                                                                          vec_size);
                        }
                        if (check_params(expr.result_type(), ham->rhs().result_type(), ham->lhs().result_type(),
                                         sum_dim, max_dim, ham_dim))
                        {
                            size_t vec_size = get_dim_size(ham->rhs().result_type(), ham_dim);
                            return stash.create<SumMaxInvHammingFunction>(expr.result_type(), ham->rhs(), ham->lhs(),
                                                                          vec_size);
                        }
                    }
                }
            }
        }
    }
    return expr;
}

} // namespace vespalib::eval
