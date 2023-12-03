// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gather_slice_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

bool GatherSliceToSplitFusion::IsSupportedGather(const Graph& graph, const Node& node, int64_t& index, int64_t& axis, int64_t& indices_n_dims) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gather", {1, 11, 13}) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
    return false;
  }

  const NodeArg& input_arg = *(node.InputDefs()[1]);
  if (!optimizer_utils::IsScalar(input_arg)) return false;

  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());

  if (!tensor_proto) return false;

  if (tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) return false;

  Initializer init_const{*tensor_proto, graph.ModelPath()};

  index = *(init_const.data<int64_t>());
  axis = 0;
  auto& attrs = node.GetAttributes();

  if (attrs.find("axis") != attrs.end()) {
    auto& axis_attr = attrs.at("axis");
    if (utils::HasInt(axis_attr)) axis = axis_attr.i();
  }

  indices_n_dims = tensor_proto->dims_size();
  return true;
}


bool GatherSliceToSplitFusion::IsSupportedSlice(const Graph& graph, const Node& node,
                                                InlinedVector<int64_t>& starts,
                                                InlinedVector<int64_t>& ends,
                                                InlinedVector<int64_t>& axes,
                                                InlinedVector<int64_t>& steps) const {
    // check the version to support Slice
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {1, 10, 11, 13}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
            return false;
    }

    // get the opset version
    int onnx_opset_version = -1;
    if (graph.DomainToVersionMap().find(kOnnxDomain) != graph.DomainToVersionMap().end()) {
        onnx_opset_version = graph.DomainToVersionMap().at(kOnnxDomain);
    };

    // If it is a Slice ops of opset version 1
    if (onnx_opset_version == 1) {
        if (!graph_utils::GetRepeatedNodeAttributeValues(node, "starts", starts) ||
            !graph_utils::GetRepeatedNodeAttributeValues(node, "ends", ends) ||
            starts.size() != ends.size()) {
                return false;
            }

        if (graph_utils::GetRepeatedNodeAttributeValues(node, "axes", axes) && (axes.size() != starts.size())) {
            return false;
        }


    } else if (onnx_opset_version > 10) {
        // starts/ends/axes/steps

        // return a pointer to the corresponding NodeArg if input of the node at this index exists.
        auto get_input_if_exists = [&node](size_t input_idx) -> const NodeArg* {
            const auto& input_defs = node.InputDefs();
            const NodeArg* input = (input_defs.size() > input_idx) ? input_defs[input_idx] : nullptr;
            return (input == nullptr || !input->Exists()) ? nullptr : input;
        };

        // return a pointer to initializer if it is constant;
        auto get_initializer_if_constant =
            [&graph, get_input_if_exists](size_t input_idx) -> const ONNX_NAMESPACE::TensorProto* {
                const NodeArg* input = get_input_if_exists(input_idx);
                return input ? graph_utils::GetConstantInitializer(graph, input->Name()) : nullptr;
        };

        // return initializer data
        auto get_initializer_data =
            [&graph](const ONNX_NAMESPACE::TensorProto* initializer) -> InlinedVector<int64_t> {
                Initializer init(*initializer, graph.ModelPath());
                if (initializer->data_type() == ONNX_NAMESPACE::TensorProto::INT32) {
                    int32_t init_data = init.data<int32_t>();
                    return InlinedVector<int64_t>(init_data, init_data + init.size());
                } else if (initializer->data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
                    int64_t init_data = init.data<int64_t>();
                    return InlinedVector<int64_t>(init_data, init_data + init.size());
                };
                return {};
        };

        // starts and ends inputs have to exist, be constant, and be of the same size.
        const ONNX_NAMESPACE::TensorProto* starts_init = get_initializer_if_constant(1);
        const ONNX_NAMESPACE::TensorProto* ends_init   = get_initializer_if_constant(2);

        if (starts_init && ends_init) {
            starts = get_initializer_data(starts_init);
            ends   = get_initializer_data(ends_init);

            if (starts.size() == 0 || ends.size() == 0 || starts.size() != ends.size()) {
                return false;
            }

            // If axes input exists, it should be constant and of the same size as starts/ends.
            if (get_input_if_exists(3)) {
                const ONNX_NAMESPACE::TensorProto* axes_init = get_initializer_if_constant(3);

                if (!axes_init || axes_init->dims_size() != 1) ||
                    static_cast<size_t>(axes_init->dims().Get(0) != starts.size()) {
                        return false;
                };

                axes = get_initializer_data(axes_init);
            }

            // If steps input exists
            if (get_input_if_exists(4)) {
                const ONNX_NAMESPACE::TensorProto* steps_init = get_initializer_if_constant(4);

                if (!steps_init) return false;
                steps = get_initializer_data(steps_init);
                if (steps.size() != starts.size()) return false;

                for (int64_t step : steps) {
                    if (step != 1) return false;
                }
            }
        } else {
        return false;
        }
    }
}

/*
GatherToSplitFusion is to fuse:
Node -> Gather(index=0, axis=axis)
    |-> Gather(index=1, axis=axis)
    |-> Slice(index=2, axis=axis)
    |...

To

Node -> Split(index=0)

So that we can use one kernel to finish the job.
*/
Status GatherSliceToSplitFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
    GraphViewer graph_viewer(graph);

    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    InlinedVector<const NodeArg*> node_args;

    // Iterate the topological order
    for (auto node_index : node_topology_list) {
        auto* p_node = graph.GetNode(node_index);

        if (p_node == nullptr) continue;

        Node& node = *p_node;

        ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

        // Currently only catch after Reshape ops, optimize in the future
        if (node.OpType() != "Reshape") continue;

        size_t output_count = node.GetOutputEdgesCount();

        if (output_count <= 1) continue;

        // Get the output into node args
        for (auto arg_arg : node.MutableOutputDefs()) {
            node_args.push_back(arg_arg);
        }
    };

    for (const NodeArg* node_arg : node_args) {
        auto shape = node_arg->Shape();
        if (!shape) continue;

        int64_t rank = static_cast<int64_t>(shape->dim_size());

        bool can_fuse = true;
        bool first_edge = true;
        int64_t split_axis = 0;
        int64_t indices_n_dims = -1;

        auto consumers = graph.GetConsumerNodes(node_arg->Name());
        size_t consumer_count = consumers.size();

        if (consumer_count != 3) continue;

        // TODO: how to catch up the Slice output value
        InlinedVector<NodeArg*> gather_outputs;
        InlinedVector<NodeArg*> slice_outputs;

        InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse;

        // find the nodes to be merged
        for (auto consumer : consumers) {
            int64_t index, axis, dims;
            InlinedVector<int64_t> starts, ends, axes, steps;

            bool IsSupportedGatherOps = IsSupportedGather(graph, *consumer, index, axis, dims);
            bool IsSupportedSliceOps = IsSupportedSlice(graph, *consumer, starts, ends, axes, steps);

            if (!consumer || consumer->InputDefs()[0] != node_arg ||
                !IsSupportedGatherOps || !IsSupportedSliceOps) {
                    can_fuse = false;
                    break;
            }

            // Check the Gather Ops
            if (IsSupportedGatherOps) {
                if (indices_n_dims == -1) {
                    indices_n_dims = dims;
                } else if (indices_n_dims != dims) {
                    // Not the same number of dimensions (0 or 1) for all scalar indices.
                    can_fuse = false;
                    break;
                }
                if (axis < 0) axis += rank;
                if (first_edge) {
                    auto dim = shape->dim(static_cast<int>(axis));

                    if (!utils::HasDimValue(dim) || dim.dim_value() != static_cast<int64_t>(consumer_count)) {
                        can_fuse = false;
                        break;
                    }
                    split_axis = axis;
                    first_edge = false;
                } else if (axis != split_axis) {
                        can_fuse = false;
                        break;
                }

                if (index < 0) index += static_cast<int64_t>(consumer_count);
                if (index < 0 || index >= static_cast<int64_t>(consumer_count) || gather_outputs[static_cast<size_t>(index)]) {
                    can_fuse = false;
                    break;
                }

                Node& gather_node = *graph.GetNode(consumer->Index());
                NodeArg* gather_arg = gather_node.MutableOutputDefs()[0];
                nodes_to_fuse.push_back(gather_node);
                gather_outputs.push_back(gather_arg);
            };

            // Check the Slice Ops
            if (IsSupportedSliceOps) {
                if (axes[0] != axis) {
                    can_fuse = false;
                    break;
                }

                Node& slice_node = *graph.GetNode(consumer->Index());
                NodeArg* slice_arg = slice_node.MutableOutputDefs()[0];
                nodes_to_fuse.push_back(slice_node);
                slice_outputs.push_back(slice_arg);
            };
        }

        // Check the Slice direction to be merged

        if (!can_fuse) continue;

        ONNX_NAMESPACE::TypeProto split_output_type;
        const ONNX_NAMESPACE::TensorProto_DataType element_type =
            static_cast<ONNX_NAMESPACE::TensorProto_DataType>(node_arg->TypeAsProto()->tensor_type().elem_type());

        split_output_type.mutable_tensor_type()->set_elem_type(element_type);

        for (int64_t i = 0; i < rank; ++i) {
            if (i == split_axis) {
                split_output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1LL);
            } else {
                *(split_output_type.mutable_tensor_type()->mutable_shape()->add_dim()) = shape->dim(static_cast<int>(i));
            }
        }

        InlinedVector<NodeArg*> split_outputs;

        // TODO: Need to combine <gather_output, slice_output> to get newer node.
        Node& split_node =
            graph.AddNode(
                graph.GenerateNodeArgName("Split"), "Split", "Split for Gather and Slice nodes",
                    {graph.GetNodeArg(node_arg->Name())}, gather_outputs, slice_outputs
            );

        split_node.AddAttribute("axis", split_axis);

        split_node.SetExecutionProviderType(nodes_to_fuse[0].get().GetExecutionProviderType());


        for (Node& n : nodes_to_fuse) {
            graph_utils::RemoveNodeOutputEdges(graph, n);
            graph.RemoveNode(n.Index());
        }

        modified = true;

    }

    return Status::OK();

}


}  // namespace onnxruntime
