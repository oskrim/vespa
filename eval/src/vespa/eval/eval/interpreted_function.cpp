// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "interpreted_function.h"
#include "node_visitor.h"
#include "node_traverser.h"
#include "tensor_nodes.h"
#include "tensor_engine.h"
#include "make_tensor_function.h"
#include "compile_tensor_function.h"
#include "simple_tensor_engine.h"
#include <vespa/vespalib/util/classname.h>
#include <vespa/eval/eval/llvm/compile_cache.h>
#include <vespa/vespalib/util/benchmark_timer.h>
#include <set>

namespace vespalib::eval {

namespace {

const Function *get_lambda(const nodes::Node &node) {
    if (auto ptr = nodes::as<nodes::TensorMap>(node)) {
        return &ptr->lambda();
    }
    if (auto ptr = nodes::as<nodes::TensorJoin>(node)) {
        return &ptr->lambda();
    }
    if (auto ptr = nodes::as<nodes::TensorMerge>(node)) {
        return &ptr->lambda();
    }
    return nullptr;
}

} // namespace vespalib::<unnamed>


InterpretedFunction::State::State(EngineOrFactory engine_in)
    : engine(engine_in),
      params(nullptr),
      stash(),
      stack(),
      program_offset(0),
      if_cnt(0)
{
}

InterpretedFunction::State::~State() = default;

void
InterpretedFunction::State::init(const LazyParams &params_in) {
    params = &params_in;
    stash.clear();
    stack.clear();
    program_offset = 0;
    if_cnt = 0;
}

InterpretedFunction::Context::Context(const InterpretedFunction &ifun)
    : _state(ifun._tensor_engine)
{
}

InterpretedFunction::InterpretedFunction(EngineOrFactory engine, const TensorFunction &function)
    : _program(),
      _stash(),
      _tensor_engine(engine)
{
    _program = compile_tensor_function(engine, function, _stash);
}

InterpretedFunction::InterpretedFunction(EngineOrFactory engine, const nodes::Node &root, const NodeTypes &types)
    : _program(),
      _stash(),
      _tensor_engine(engine)
{
    const TensorFunction &plain_fun = make_tensor_function(engine, root, types, _stash);
    const TensorFunction &optimized = engine.optimize(plain_fun, _stash);
    _program = compile_tensor_function(engine, optimized, _stash);
}

InterpretedFunction::~InterpretedFunction() = default;

const Value &
InterpretedFunction::eval(Context &ctx, const LazyParams &params) const
{
    State &state = ctx._state;
    state.init(params);
    while (state.program_offset < _program.size()) {
        _program[state.program_offset++].perform(state);
    }
    assert(state.stack.size() == 1);
    return state.stack.back();
}

double
InterpretedFunction::estimate_cost_us(const std::vector<double> &params, double budget) const
{
    Context ctx(*this);
    SimpleParams lazy_params(params);
    auto actual = [&](){eval(ctx, lazy_params);};
    return BenchmarkTimer::benchmark(actual, budget) * 1000.0 * 1000.0;
}

Function::Issues
InterpretedFunction::detect_issues(const Function &function)
{
    struct NotSupported : NodeTraverser {
        std::vector<vespalib::string> issues;
        bool open(const nodes::Node &) override { return true; }
        void close(const nodes::Node &node) override {
            auto lambda = get_lambda(node);
            if (lambda && CompiledFunction::detect_issues(*lambda)) {
                issues.push_back(make_string("lambda function that cannot be compiled within %s",
                                getClassName(node).c_str()));
            }
        }
    } checker;
    function.root().traverse(checker);
    return Function::Issues(std::move(checker.issues));
}

InterpretedFunction::EvalSingle::EvalSingle(EngineOrFactory engine, Instruction op)
    : _state(engine),
      _op(op)
{
}

const Value &
InterpretedFunction::EvalSingle::eval(const std::vector<Value::CREF> &stack)
{
    _state.stash.clear();
    _state.stack = stack;
    _op.perform(_state);
    assert(_state.stack.size() == 1);
    return _state.stack.back();
}

}
