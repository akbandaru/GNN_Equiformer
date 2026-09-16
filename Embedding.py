import torch
import e3nn
import math
from e3nn import o3
from torch import nn 
from torch.nn import init
from typing import Optional
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import collections
from e3nn.math import perm


class TensorProductRescale(torch.nn.Module):
    def __init__(self,
        irreps_in1, irreps_in2, irreps_out,
        instructions,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):

        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2, irreps_out=self.irreps_out, 
            instructions=instructions, normalization=normalization,
            internal_weights=internal_weights, shared_weights=shared_weights, 
            path_normalization='none')

        self.init_rescale_bias()


    def calculate_fan_in(self, ins):
        return {
            'uvw': (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            'uvu': self.irreps_in2[ins.i_in2].mul,
            'uvv': self.irreps_in1[ins.i_in1].mul,
            'uuw': self.irreps_in1[ins.i_in1].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]


    def init_rescale_bias(self) -> None:

        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_parity = [irrep_str[-1] for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(self.irreps_bias).split('+')]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if self.irreps_bias_orders[slice_idx] == 0 and self.irreps_bias_parity[slice_idx] == 'e':
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype))
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    #else:
                    #    sqrt_k = 1.
                    #
                    #if self.rescale:
                        #weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    #self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            #for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)


    def forward_tp_rescale_bias(self, x, y, weight=None):

        out = self.tp(x, y, weight)

        #if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for (_, slice, bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                #out[:, slice] += bias
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out


    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(self,
        irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):

        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out,
            instructions=instructions,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)


class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        super().__init__(irreps_in, o3.Irreps('1x0e'), irreps_out, 
            bias=bias, rescale=rescale, internal_weights=True, 
            shared_weights=True, normalization=None)

    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out

class NodeEmbedding(nn.Module):
    def __init__(self, irreps_node_embed_dim, num_features=5, bias=True):
        super().__init__()
        self.num_features = num_features
        self.irreps_node_embedding = o3.Irreps(irreps_node_embed_dim)
        self.input_irreps = o3.Irreps('{}x0e'.format(self.num_features))
        self.feature_lin = LinearRS(self.input_irreps,
                                    self.irreps_node_embedding, bias=bias)

    def forward(self, node_feature):
        node_embedding = self.feature_lin(node_feature)
        return node_embedding

class DepthWiseTensorProduct(nn.Module):
    def __init__(self, irreps_1, irreps_2, irreps_out, internal_weights=False, bias=True):
        super().__init__()
        irreps_in1 = o3.Irreps(irreps_1)
        irreps_in2 = o3.Irreps(irreps_2)
        irreps_output = o3.Irreps(irreps_out)

        instructions = []
        irreps_out_list = []
        for i, (mul, ir_in) in enumerate(irreps_in1):
            for j, (_, ir_edge) in enumerate(irreps_in2):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_output:
                        k = len(irreps_out_list)
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        self.irreps_output = o3.Irreps(irreps_out_list)

        self.tp = TensorProductRescale(irreps_in1, irreps_in2, self.irreps_output,
                                      instructions,
                                      internal_weights=internal_weights,
                                      shared_weights=internal_weights)

    def forward(self, x, edge_attr, weight=None):
        return self.tp(x, edge_attr, weight)

class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            if (i == len(ch_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_biases))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            #modules.append(nn.ReLU())
            #modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])), 
            #    acts=[torch.nn.functional.silu]))
            #modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])), 
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.offset, -bound, bound)


    def forward(self, f_in):
        f_out = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1) 
        return f_out

def boardcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor]=None,
                dim_size: Optional[int]=None):
    index = boardcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num

    def forward(self, x, index, dim_size):
        out = scatter_sum(x, index, dim=0, dim_size=dim_size)
        return out.div(self.avg_aggregate_num ** 0.5)

class GaussianRBF(nn.Module):
    def __init__(self, r_cut=5.0, num_basis=128):
        super().__init__()
        centers = torch.linspace(0, r_cut, num_basis)
        self.centers = centers
        self.width = (centers[1] - centers[0]).item()

    def forward(self, distances):
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-(diff ** 2) / (2 * self.width ** 2))

def build_edge_inputs(data, irreps_edge_attr, rbf):
    pos = data.pos
    batch = data.batch

    edge_src, edge_dst = data.edge_index[0], data.edge_index[1]

    edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)

    if hasattr(data, "cell_offsets") and data.cell_offsets is not None:
        edge_batch = batch.index_select(0, edge_src)
        cell_per_edge = data.cell.index_select(0, edge_batch)
        shift = torch.einsum("ei, eij->ej", data.cell_offsets, cell_per_edge)
        edge_vec = edge_vec + shift

    edge_attr = o3.spherical_harmonics(
        l=o3.Irreps(irreps_edge_attr),
        x=edge_vec,
        normalize=True,
        normalization="component"
    )

    edge_length = edge_vec.norm(dim=1)
    edge_scalars = rbf(edge_length)

    return edge_src, edge_dst, edge_attr, edge_scalars, batch

class EdgeDegreeEmbeddingNetwork(nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.exp = LinearRS(o3.Irreps('1x0e'), self.irreps_node_embedding, bias=True, rescale=True)
        self.dw = DepthWiseTensorProduct(self.irreps_node_embedding,
                                         self.irreps_edge_attr,
                                         self.irreps_node_embedding,
                                         internal_weights=False,
                                         bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.tp.weight_numel])

        for (sl, sl_sqrt_k) in self.dw.tp.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[sl, :] *= sl_sqrt_k
            self.rad.offset.data[sl] *= sl_sqrt_k

        self.proj = LinearRS(self.dw.irreps_output.simplify(), self.irreps_node_embedding)

        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(self, node_input, edge_attr, edge_scalar, edge_src, edge_dst, batch, dim_size):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalar)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim_size)
        return node_features

class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 irreps_node_embed_dim='128x0e+64x1e+32x2e',
                 num_features=5,
                 bias=True,
                 irreps_edge_attr="1x0e+1x1e+1x2e",
                 fc_neurons=[64, 64],
                 avg_aggregate_num=10,
                 r_cut=5.0,
                 num_basis=128):
        super().__init__()
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.node_embedder = NodeEmbedding(irreps_node_embed_dim=irreps_node_embed_dim,
                                           num_features=num_features,
                                           bias=bias)
        self.edge_embedder = EdgeDegreeEmbeddingNetwork(irreps_node_embedding=irreps_node_embed_dim,
                                                        irreps_edge_attr=irreps_edge_attr,
                                                        fc_neurons=fc_neurons,
                                                        avg_aggregate_num=avg_aggregate_num)

    def forward(self, x, edge_src, edge_dst, edge_attr, edge_scalars, batch):
        atom_embedding = self.node_embedder(x)
        num_nodes = atom_embedding.size(0)
        edge_embedding = self.edge_embedder(atom_embedding, edge_attr, edge_scalars, edge_src, edge_dst, batch, dim_size=num_nodes)
        # print(f"atom_embedding: {atom_embedding.shape[0]} | edge_embedding: {edge_embedding.shape[0]} | num nodes (x): {x.shape[0]} | edge_dst max: {int(edge_dst.max())} | num edges: {edge_dst.numel()}")
        embedding = atom_embedding + edge_embedding
        return embedding, x
