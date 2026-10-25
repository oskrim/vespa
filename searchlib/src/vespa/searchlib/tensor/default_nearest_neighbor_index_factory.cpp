// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "default_nearest_neighbor_index_factory.h"

#include "distance_function_factory.h"
#include "hnsw_index.h"
#include "inv_log_level_generator.h"
#include "ivf_index.h"
#include "random_level_generator.h"

#include <vespa/searchcommon/attribute/config.h>

#include <vespa/log/log.h>
LOG_SETUP(".searchlib.tensor.default_nearest_neighbor_index_factory");

namespace search::tensor {

using vespalib::eval::ValueType;

namespace {

class LevelZeroGenerator : public RandomLevelGenerator {
    uint32_t max_level() override { return 0; }
};

RandomLevelGenerator::UP make_random_level_generator(uint32_t m) {
    return std::make_unique<InvLogLevelGenerator>(m);
}

} // namespace

std::unique_ptr<NearestNeighborIndex> DefaultNearestNeighborIndexFactory::make(
    const DocVectorAccess& vectors, size_t vector_size, bool multi_vector_index, vespalib::eval::CellType cell_type,
    const search::attribute::HnswIndexParams&           params,
    const std::optional<attribute::QuantizationParams>& quant_params,
    const std::optional<attribute::IvfIndexParams>& ivf_params, const std::string& base_file_name) const {
    uint32_t        m = params.max_links_per_node();
    HnswIndexConfig cfg(m * 2, m, params.neighbors_to_explore_at_insert(), 10000, true);
    auto dist_ff = make_distance_function_factory(params.distance_metric(), cell_type, vector_size, quant_params);
    if (ivf_params.has_value()) {
        if (!quant_params.has_value()) {
            return std::make_unique<IvfIndex>(vectors, std::move(dist_ff), cell_type, vector_size,
                                              params.distance_metric(), IvfIndexConfig::from_params(*ivf_params),
                                              IvfIndex::posting_path_for(base_file_name));
        }
        LOG(warning, "IVF index does not support quantized tensors, using HNSW");
    }
    if (multi_vector_index) {
        return std::make_unique<HnswIndex<HnswIndexType::MULTI>>(vectors, std::move(dist_ff),
                                                                 make_random_level_generator(m), cfg);
    } else {
        return std::make_unique<HnswIndex<HnswIndexType::SINGLE>>(vectors, std::move(dist_ff),
                                                                  make_random_level_generator(m), cfg);
    }
}

} // namespace search::tensor
