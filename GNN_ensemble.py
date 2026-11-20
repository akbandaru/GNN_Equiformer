# ============================================================
# END-TO-END BAYESIAN OPTIMIZATION — EQUIFORMER + OMAT24 POOL
# Surrogate   : Equiformer deep ensemble (invariant uncertainty)
# Pool        : OMAT24Graphs (labelled -> benchmark/active-learning mode)
# Acquisition : UCB (default) or Thompson Sampling
# Objective   : minimize potential energy (scalar, invariant)
# Pre-training: backbone-only warm start (head re-initialized per member)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase import Atoms
from ase.neighborlist import neighbor_list
from e3nn import o3
from Equiformer_energy import Equiformer
from Equiformer_energy import feature_generator_electronegativity, feature_generator_group_num, feature_generator_period_num
from fairchem.core.datasets import AseDBDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm 

period_num_array = feature_generator_period_num()
group_num_array = feature_generator_group_num()
electronegativity_array = feature_generator_electronegativity()

DATAPATH = "/home/bandarua/scratch/projects/data/rattled-relax"


def step_index(sid: str) -> int:
    last = sid.rsplit("_", 1)[-1]
    return int(last) if last.isdigit() else None

class Node:
    def __init__(self,
                 atomic_number: int,
                 atomic_mass: float):
        self.atomic_number = int(atomic_number) / 118
        self.period_number = (period_num_array[atomic_number - 1] - period_num_array.min()) / \
                             (period_num_array.max() - period_num_array.min())
        self.group_number = (group_num_array[atomic_number - 1] - group_num_array.min()) / \
                            (group_num_array.max() - group_num_array.min()).item()
        self.mass = atomic_mass.item()
        self.electronegativity = (electronegativity_array[atomic_number - 1] - electronegativity_array.min()) / \
                                 (electronegativity_array.max() - electronegativity_array.min())


class Graph:
    def __init__(self, ionic_compound: Atoms):
        self.compound = ionic_compound
        self.nodes = self.get_nodes()
        self.edge_index, self.cell_offsets, self.edge_vec, self.edge_len, self.edges = self.get_edges_new()

    def get_nodes(self):
        masses = self.compound.get_masses()
        mass_sum = masses.sum()
        return [
            Node(atomic_number=z, atomic_mass=m / mass_sum)
            for z, m in zip(self.compound.numbers, masses)
        ]

    def get_edges_new(self, cutoff=5.0):
        i, j, S, D, d = neighbor_list("ijSDd", self.compound, cutoff=cutoff)
        edges = np.column_stack((i, j))
        edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
        cell_offsets = torch.tensor(S, dtype=torch.float)
        edge_vec = torch.tensor(D, dtype=torch.float)
        edge_len = torch.tensor(d, dtype=torch.float)
        return edge_index, cell_offsets, edge_vec, edge_len, edges

def average_degree_per_graph(edge_index: torch.Tensor):
    dict_node_degrees = {}
    for node in edge_index[0]:
        if node.item() in dict_node_degrees.keys():
            dict_node_degrees[node.item()] += 1
        else:
            dict_node_degrees[node.item()] = 1
    length_dict = len(dict_node_degrees)
    avg_degree = sum(dict_node_degrees.values()) / length_dict

    return avg_degree


from torch import nn
from e3nn import o3
class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps, 
            o3.Irreps('{}x0e'.format(self.num_irreps)))


    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out

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
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True, irreps_2=o3.Irreps('1x0e')):
        super().__init__(irreps_in, irreps_2, irreps_out, 
            bias=bias, rescale=rescale, internal_weights=True, 
            shared_weights=True, normalization=None)

    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out

from torch import nn
from Embedding import EmbeddingLayer
from LayerNorm import EquivariantLayerNormV2
from GraphAttention import GraphAttention
from FeedForwardNetwork import FeedForwardNetwork
from GraphAttention import ScaledScatter
from GraphAttention import Activation
from e3nn import o3

# _AVG_NODES = 8.375834289843983
# _AVG_DEGREE = 23.40640538220446

class Transformer(nn.Module):
    def __init__(self, 
                 irreps_node="128x0e+64x1e+32x2e",
                 irreps_node_attr="5x0e",
                 irreps_edge_attr="1x0e+1x1e+1x2e",
                 fc_neurons=[128, 64],
                 irreps_head="32x0e+16x1e+8x2e",
                 num_heads=4,
                 alpha_drop=0.2, proj_drop=0.2):
        super().__init__()

        self.layer_norm_1 = EquivariantLayerNormV2(irreps=irreps_node)
        self.graph_attn = GraphAttention(irreps_node_input=irreps_node,
                                         irreps_edge_attr=irreps_edge_attr,
                                         irreps_node_output=irreps_node,
                                         fc_neurons=fc_neurons,
                                         num_heads=num_heads,
                                         irreps_head=irreps_head,
                                         alpha_drop=alpha_drop, proj_drop=proj_drop)
        self.layer_norm_2 = EquivariantLayerNormV2(irreps=irreps_node)
        self.ffn = FeedForwardNetwork(irreps_node_input=irreps_node,
                                      irreps_node_attr=irreps_node_attr,
                                      irreps_node_output=irreps_node,
                                      irreps_mlp_mid=irreps_node)

    def forward(self, x, edge_src, edge_dst, edge_attr, edge_scalars, batch, node_attr):
        result = self.layer_norm_1(x)
        result_1 = self.graph_attn(result, edge_src, edge_dst, edge_attr, edge_scalars, batch)
        result_2 = result + result_1
        result_3 = self.layer_norm_2(result_2)
        result_4 = self.ffn(result_3, node_attr)
        result_5 = result_2 + result_4
        return result_5

class Equiformer(nn.Module):
    def __init__(self, irreps_node="128x0e+64x1e+32x2e", num_features=5,
                 bias=True, irreps_edge_attr="1x0e+1x1e+1x2e", fc_neurons=[128, 64],
                 avg_aggregate_num_1=None, irreps_head="32x0e+16x1e+8x2e",
                 num_heads=4, alpha_drop=0.2, proj_drop=0.2, irreps_node_attr="5x0e",
                 irreps_mlp_mid="384x0e+192x1e+96x2e", irreps_final="1440x0e", output="1x1e",
                 avg_aggregate_num_2=None):
        super().__init__()
        self.embedder = EmbeddingLayer(irreps_node_embed_dim=irreps_node,
                                       num_features=num_features,
                                       bias=bias,
                                       irreps_edge_attr=irreps_edge_attr, 
                                       avg_aggregate_num=avg_aggregate_num_1, 
                                       fc_neurons=fc_neurons)
        self.transformer_1 = Transformer(irreps_node=irreps_node,
                                         irreps_node_attr=irreps_node_attr,
                                         irreps_edge_attr=irreps_edge_attr,
                                         fc_neurons=fc_neurons,
                                         irreps_head=irreps_head,
                                         num_heads=num_heads,
                                         alpha_drop=alpha_drop, proj_drop=proj_drop)

        self.transformer_2 = Transformer(irreps_node=irreps_node,
                                         irreps_node_attr=irreps_node_attr,
                                         irreps_edge_attr=irreps_edge_attr,
                                         fc_neurons=fc_neurons,
                                         irreps_head=irreps_head,
                                         num_heads=num_heads,
                                         alpha_drop=alpha_drop, proj_drop=proj_drop)

        self.transformer_3 = Transformer(irreps_node=irreps_node,
                                         irreps_node_attr=irreps_node_attr,
                                         irreps_edge_attr=irreps_edge_attr,
                                         fc_neurons=fc_neurons,
                                         irreps_head=irreps_head,
                                         num_heads=num_heads,
                                         alpha_drop=alpha_drop, proj_drop=proj_drop)

        self.last_layer_norm = EquivariantLayerNormV2(irreps=irreps_node)
        self.last_ffn = FeedForwardNetwork(irreps_node_input=irreps_node,
                                        irreps_node_attr=irreps_node_attr,
                                        irreps_node_output=irreps_mlp_mid,
                                        irreps_mlp_mid=irreps_mlp_mid)
        self.last_scatter = ScaledScatter(avg_aggregate_num=avg_aggregate_num_2)

        self.head = nn.Sequential(LinearRS(irreps_in=o3.Irreps(irreps_final), irreps_out=o3.Irreps(irreps_final)),
                                Activation(o3.Irreps(irreps_final), acts=[nn.SiLU()]),
                                LinearRS(o3.Irreps(irreps_final), o3.Irreps(output))
                                )

        print("irreps_final:", o3.Irreps(irreps_final))
        print("output:", o3.Irreps(output))

        lin2 = self.head[2]
        print("instructions:", lin2.tp.instructions)   # ← expect []  (the bug)
        print("weight_numel:", lin2.tp.weight_numel)    # ← expect 0

    def forward(self, input, edge_src, edge_dst, edge_attr, edge_scalars, batch):
        result, node_attr_x = self.embedder(input, edge_src, edge_dst, edge_attr, edge_scalars,batch)
        result_1 = self.transformer_1(result, edge_src, edge_dst, edge_attr, edge_scalars, batch, node_attr_x)
        result_2 = self.transformer_2(result_1, edge_src, edge_dst, edge_attr, edge_scalars, batch, node_attr_x)
        result_3 = self.transformer_3(result_2, edge_src, edge_dst, edge_attr, edge_scalars, batch, node_attr_x)
        result_4 = self.last_layer_norm(result_3)
        result_5 = self.last_ffn(result_4, node_attr_x)
        # print(f"Result 5 shape: {result_5.shape}")
        # result_6 = self.head(result_5)
        x = result_5
        # print("input", x.requires_grad, x.grad_fn)

        x = self.head[0](x)
        # print("after LinearRS #1", x.requires_grad, x.grad_fn)

        x = self.head[1](x)
        # print("after Activation", x.requires_grad, x.grad_fn)

        x = self.head[2](x)
        # print("after LinearRS #2", x.requires_grad, x.grad_fn)

        # print(f"Result 6 shape: {x.shape}")
        # print("result", result.requires_grad, result.grad_fn)
        # print("result_1", result_1.requires_grad, result_1.grad_fn)
        # print("result_2", result_2.requires_grad, result_2.grad_fn)
        # print("result_3", result_3.requires_grad, result_3.grad_fn)
        # print("result_4", result_4.requires_grad, result_4.grad_fn)
        # print("result_5", result_5.requires_grad, result_5.grad_fn)
        # print("result_6", x.requires_grad, x.grad_fn)

        result_7 = self.last_scatter(x, batch)
        return result_7



# ============================================================
# 0. CONFIG
# ============================================================
class Config:
    # BO loop
    N_INIT, N_ITERATIONS, N_CANDIDATES = 10, 50, 20
    ENSEMBLE_SIZE, EPOCHS, LR = 5, 100, 1e-4      # LR lowered for fine-tuning

    # acquisition
    BETA, ACQ_MODE, MINIMIZE = 2.0, "UCB", True   # minimize energy

    # uncertainty
    SIGMA_FLOOR, WARM_START, USE_NLL = 1e-3, True, True

    # objective scalarization
    OBJECTIVE_MODE, REF_DIRECTION = "scalar", None

    # edge features (MUST match pre-training)
    R_CUT, NUM_BASIS = 5.0, 128
    IRREPS_EDGE_ATTR = "1x0e+1x1e+1x2e"

    # pre-trained weights
    PRETRAINED_PATH, PRETRAIN_MODE, PERTURB_STD = "/home/bandarua/scratch/projects/gnn_mlip/modelWeights1.pth", "backbone", 0.01

    # pool scoring
    POOL_MODE = "sample"                          # "sample" | "full"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    AVG_AGG_1 = None
    AVG_AGG_2 = None

cfg = Config()


# ============================================================
# 1. RADIAL BASIS (your definition, device-safe)
# ============================================================
class GaussianRBF(nn.Module):
    def __init__(self, r_cut=5.0, num_basis=128):
        super().__init__()
        centers = torch.linspace(0, r_cut, num_basis)
        self.register_buffer("centers", centers)          # follows .to(device)
        self.width = (centers[1] - centers[0]).item()

    def forward(self, distances):
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-(diff ** 2) / (2 * self.width ** 2))


RBF = GaussianRBF(cfg.R_CUT, cfg.NUM_BASIS).to(cfg.DEVICE)


# ============================================================
# 2. EDGE-INPUT BUILDER (your function, verbatim)
# ============================================================
def build_edge_inputs(data, irreps_edge_attr, rbf):
    pos, batch = data.pos, data.batch
    edge_src, edge_dst = data.edge_index[0], data.edge_index[1]
    edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)

    if hasattr(data, "cell_offsets") and data.cell_offsets is not None:
        edge_batch = batch.index_select(0, edge_src)
        cell_per_edge = data.cell.index_select(0, edge_batch)
        shift = torch.einsum("ei, eij->ej", data.cell_offsets, cell_per_edge)
        edge_vec = edge_vec + shift

    edge_attr = o3.spherical_harmonics(
        l=o3.Irreps(irreps_edge_attr), x=edge_vec,
        normalize=True, normalization="component")

    edge_scalars = rbf(edge_vec.norm(dim=1))
    return edge_src, edge_dst, edge_attr, edge_scalars, batch

class OMAT24Graphs(Dataset):
    def __init__(self, graph_list):
        super().__init__(root=None, transform=None)
        self.graph_list = graph_list

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        selected_graph = self.graph_list[idx]
        node_features = []
        for node in selected_graph.nodes:
            node_features.append([node.atomic_number, node.period_number,
                                  node.group_number, node.mass, node.electronegativity])
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            pos=torch.tensor(selected_graph.compound.positions, dtype=torch.float32),
            edge_index=selected_graph.edge_index,
            edge_vec=selected_graph.edge_vec,
            edge_attr=selected_graph.edge_len,
            cell_offsets=selected_graph.cell_offsets,
            cell=torch.tensor(selected_graph.compound.cell).unsqueeze(0).float(),
            y=selected_graph.compound.get_potential_energy(),
        )
        return data

    def best(self, minimize=True):
        best_graph = None
        best_value = None

        for idx in range(self.len()):
            graph = self.graph_list[idx]
            value = graph.compound.get_potential_energy()

            if best_value is None:
                best_graph, best_value = graph, value
                continue

            if (minimize and value < best_value) or (not minimize and value > best_value):
                best_graph, best_value = graph, value

        return best_graph, best_value

# ============================================================
# 3. MODEL ADAPTERS
# ============================================================
def make_equiformer():
    """Fresh Equiformer with the phase-correct output head."""
    if cfg.OBJECTIVE_MODE == "scalar":
        output = "2x0e" if cfg.USE_NLL else "1x0e"
    else:
        output = "1x1e + 1x0e"
    return Equiformer(output=output,
                      avg_aggregate_num_1=cfg.AVG_AGG_1,
                      avg_aggregate_num_2=cfg.AVG_AGG_2)


def run_model(model, data):
    """data is a BATCHED PyG object (.batch populated)."""
    edge_src, edge_dst, edge_attr, edge_scalars, batch = \
        build_edge_inputs(data, cfg.IRREPS_EDGE_ATTR, RBF)
    return model(data.x, edge_src, edge_dst, edge_attr, edge_scalars, batch)


def as_batch(data_or_list):
    items = data_or_list if isinstance(data_or_list, list) else [data_or_list]
    return next(iter(DataLoader(items, batch_size=len(items)))).to(cfg.DEVICE)


def split_mean_var(out):
    if cfg.OBJECTIVE_MODE == "scalar":
        return (out[..., 0], out[..., 1]) if cfg.USE_NLL else (out[..., 0], None)
    vec_mean = out[..., :3]
    raw_var = out[..., 3] if out.shape[-1] > 3 else None
    return vec_mean, raw_var


def to_scalar(mean_c):
    if cfg.OBJECTIVE_MODE == "scalar":
        return mean_c
    if cfg.OBJECTIVE_MODE == "magnitude":
        return mean_c.norm(dim=-1)
    return (mean_c * cfg.REF_DIRECTION).sum(dim=-1)   # projection


def raw_to_variance(raw_var):
    return F.softplus(raw_var) + 1e-6


# ============================================================
# 4. PRE-TRAINED WEIGHT LOADING (backbone-only by default)
# ============================================================
def _load_matching(model, state):
    ms = model.state_dict()
    ok = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    ms.update(ok); model.load_state_dict(ms)
    print(f"    loaded {len(ok)}/{len(ms)} tensors "
          f"({len(ms) - len(ok)} re-initialized)")


def _reinit_head(model, seed):
    torch.manual_seed(seed)
    for module in model.head:
        for p in module.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.zeros_(p)


def load_pretrained_into(model, member_seed):
    if cfg.PRETRAINED_PATH is None:
        return model
    ckpt = torch.load(cfg.PRETRAINED_PATH, map_location=cfg.DEVICE)
    state = ckpt.get("model_state_dict", ckpt)

    if cfg.PRETRAIN_MODE == "all":
        _load_matching(model, state)
    elif cfg.PRETRAIN_MODE == "backbone":
        _load_matching(model, {k: v for k, v in state.items()
                               if not k.startswith("head.")})
        _reinit_head(model, seed=member_seed)
    elif cfg.PRETRAIN_MODE == "noisy":
        _load_matching(model, state)
        with torch.no_grad():
            torch.manual_seed(member_seed)
            for p in model.parameters():
                p.add_(torch.randn_like(p) * cfg.PERTURB_STD)
    return model


# ============================================================
# 5. EQUIFORMER ENSEMBLE
# ============================================================
class EquiformerEnsemble:
    def __init__(self):
        self.members = []

    def _build_members(self):
        self.members = []
        for m in range(cfg.ENSEMBLE_SIZE):
            torch.manual_seed(m)
            model = make_equiformer().to(cfg.DEVICE)
            model = load_pretrained_into(model, member_seed=1000 + m)
            self.members.append(model)

    def _loss(self, out, y_scalar):
        mean_c, raw_var = split_mean_var(out)
        pred = to_scalar(mean_c)
        if cfg.USE_NLL and raw_var is not None:
            var = raw_to_variance(raw_var)
            return (0.5 * torch.log(var) + (y_scalar - pred) ** 2 / (2 * var)).mean()
        return F.mse_loss(pred, y_scalar)

    def fit(self, train_dataset):
        if not self.members or not cfg.WARM_START:
            self._build_members()
        for m, model in enumerate(self.members):
            opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
            torch.manual_seed(m)                                   # per-member shuffle
            loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            model.train()
            for _ in range(cfg.EPOCHS):
                for batch in loader:
                    batch = batch.to(cfg.DEVICE)
                    opt.zero_grad()
                    loss = self._loss(run_model(model, batch),
                                      batch.y.view(-1).float())
                    loss.backward(); opt.step()

    @torch.no_grad()
    def predict(self, batched):
        for model in self.members: model.eval()
        mus, aleo = [], []
        for model in self.members:
            mean_c, raw_var = split_mean_var(run_model(model, batched))
            mus.append(to_scalar(mean_c))
            if cfg.USE_NLL and raw_var is not None:
                aleo.append(raw_to_variance(raw_var))
        mus = torch.stack(mus); mu = mus.mean(dim=0)
        if cfg.USE_NLL and aleo:
            var = (torch.stack(aleo) + mus ** 2).mean(dim=0) - mu ** 2
            sigma = var.clamp(min=0).sqrt()
        else:
            sigma = mus.std(dim=0)
        return mu, torch.clamp(sigma, min=cfg.SIGMA_FLOOR)

    @torch.no_grad()
    def sample_member_scalar(self, batched):
        model = self.members[torch.randint(len(self.members), (1,)).item()]
        model.eval()
        mean_c, _ = split_mean_var(run_model(model, batched))
        return to_scalar(mean_c)


# ============================================================
# 6. ACQUISITION  (batched over all candidates at once)
# ============================================================
def ucb_score(mu, sigma):
    signed = (-mu) if cfg.MINIMIZE else mu
    return signed + cfg.BETA * sigma


def score_candidates(ensemble, cand_data_list):
    batched = as_batch(cand_data_list)               # one forward pass per member
    if cfg.ACQ_MODE == "UCB":
        mu, sigma = ensemble.predict(batched)
        return ucb_score(mu, sigma)
    else:  # Thompson Sampling
        s = ensemble.sample_member_scalar(batched)
        return (-s if cfg.MINIMIZE else s)


# ============================================================
# 7. POOL (OMAT24Graphs, index-based bookkeeping)
# ============================================================
class IndexedPool:
    def __init__(self, ds):
        self.ds = ds
        self.evaluated = set()

    def remaining_indices(self):
        return [i for i in range(self.ds.len()) if i not in self.evaluated]

    def mark_evaluated(self, idx):
        self.evaluated.add(idx)

    def __len__(self):
        return self.ds.len() - len(self.evaluated)


def make_pool_generator(pool, mode="sample"):
    def candidate_generator(n, history=None):
        avail = pool.remaining_indices()
        if not avail:
            raise RuntimeError("Pool exhausted.")
        if mode == "full":
            chosen = avail
        else:
            k = min(n, len(avail))
            chosen = [avail[i] for i in torch.randperm(len(avail))[:k].tolist()]
        return [(i, pool.ds.get(i)) for i in chosen]   # (index, Data) pairs
    return candidate_generator


# ============================================================
# 8. TRUE OBJECTIVE  (benchmark mode: known labels)
# ============================================================
def true_evaluate(data):
    return float(data.y)          # OMAT24 ground-truth potential energy


# ============================================================
# 9. SIMPLE IN-MEMORY TRAINING SET
# ============================================================
class EvaluatedSet(torch.utils.data.Dataset):
    """Holds the (Data, y) graphs the BO loop has 'evaluated' so far."""
    def __init__(self): self.items = []
    def add(self, data): self.items.append(data)
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ============================================================
# 10. MAIN LOOP  (with regret vs. known optimum)
# ============================================================
def bayesian_optimization(pool, holdout=None):
    gen = make_pool_generator(pool, mode=cfg.POOL_MODE)

    # known optimum for regret tracking (benchmark mode only)
    _, global_best = pool.ds.best(minimize=cfg.MINIMIZE)

    train = EvaluatedSet()
    best_value = None

    # seed from the pool
    for idx, data in gen(n=cfg.N_INIT):
        y = true_evaluate(data)
        train.add(data); pool.mark_evaluated(idx)
        if best_value is None or (y < best_value) == cfg.MINIMIZE:
            best_value = y

    ensemble = EquiformerEnsemble()

    for it in range(cfg.N_ITERATIONS):
        ensemble.fit(train)
        # check_calibration(ensemble, holdout)   # optional, if you keep a labelled holdout

        cands = gen(n=cfg.N_CANDIDATES)                 # [(idx, Data), ...]
        indices = [c[0] for c in cands]
        data_list = [c[1] for c in cands]

        scores = score_candidates(ensemble, data_list)  # tensor [num_candidates]
        pick = int(scores.argmax())
        chosen_idx, chosen_data = indices[pick], data_list[pick]

        y = true_evaluate(chosen_data)
        train.add(chosen_data); pool.mark_evaluated(chosen_idx)
        if (y < best_value) == cfg.MINIMIZE:
            best_value = y

        regret = abs(best_value - global_best)
        print(f"[iter {it:3d}] best={best_value:.4f} "
              f"regret={regret:.4f} picked_idx={chosen_idx}")

        if len(pool) == 0:
            print(f"Pool exhausted after {it + 1} rounds."); break

    return best_value, global_best


# ============================================================
# 11. ENTRY POINT
# ============================================================
if __name__ == "__main__":
    cfg.OBJECTIVE_MODE, cfg.USE_NLL, cfg.MINIMIZE = "scalar", True, True

    ase_dataset = AseDBDataset(config=dict(src=DATAPATH))

    keep_indices = []
    seen_parents = set()

    for i in tqdm(range(len(ase_dataset)), desc="Getting Parents"):
        info = ase_dataset.get_atoms(i).info
        parent = info["parent_id"]
        step = step_index(info["sid"])

        is_seed = step in (0, None)

        if is_seed and parent not in seen_parents:
            seen_parents.add(parent)
            keep_indices.append(i)

    print(f"Keeping {len(keep_indices)} of {len(ase_dataset)} structures")

    split = int(len(keep_indices))
    possible_idx = np.arange(start=0, stop=len(keep_indices))
    idx = np.random.choice(len(possible_idx), size=split, replace=False)

    ionic_compounds = []
    for i, index in tqdm(enumerate(idx), desc="Get Compounds"):
        ionic_compounds.append(ase_dataset.get_atoms(index))

    ionic_compounds_PE = []
    for compound in tqdm(ionic_compounds, desc="Check compounds"):
        try:
            compound.get_potential_energy()
            ionic_compounds_PE.append(compound)
        except NotImplementedError:
            print("Does not have the PE value!")
    print(len(ionic_compounds_PE))

    graph_list = []
    for compound in tqdm(ionic_compounds_PE, desc="Building graphs"):
        graph_obj = Graph(ionic_compound=compound)
        if graph_obj.edges.shape[0] > 0:
            graph_list.append(graph_obj)

    pool_ds = OMAT24Graphs(graph_list)        # your loaded graph list
    average_degrees_all_graphs = [average_degree_per_graph(edge_index=data.edge_index)
                                      for data in tqdm(pool_ds, desc="Graph Degrees")]
    _AVG_DEGREE = sum(average_degrees_all_graphs) / len(average_degrees_all_graphs)
    
    nodes_num_all_graph = [data.x.shape[0] for data in tqdm(pool_ds, desc="Graph Nodes")]
    _AVG_NODES = sum(nodes_num_all_graph) / len(nodes_num_all_graph)

    cfg.AVG_AGG_1 = _AVG_DEGREE
    cfg.AVG_AGG_2 = _AVG_NODES

    pool = IndexedPool(pool_ds)

    best, optimum = bayesian_optimization(pool)
    print(f"\nBO best = {best:.4f} | true optimum = {optimum:.4f} "
          f"| final regret = {abs(best - optimum):.4f}")
