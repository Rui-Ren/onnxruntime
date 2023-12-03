// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"


namespace onnxruntime {

/**
@class GatherSliceToSplitFusion

Fuse 2 Gather nodes and 1 Slice to one split node.
*/

class GatherSliceToSplitFusion : public GraphTransformer {
	public:
		GatherSliceToSplitFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
			: GraphTransformer("GatherSliceToSplitFusion", compatible_execution_providers) {}

		Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

	private:
		bool IsSupportedGather(const Graph& graph, const Node& node, int64_t& index, int64_t& axis, int64_t& indices_n_dims) const;
		bool IsSupportedSlice(const Graph& graph, const Node& node, InlinedVector<int64_t>& starts, InlinedVector<int64_t>& ends, InlinedVector<int64_t>& axes, InlinedVector<int64_t>& steps) const;

};
}
