"""
Equiformer training script.
Converted from Equiformer_train.ipynb.

Requires local modules: Embedding.py, LayerNorm.py, GraphAttention.py,
FeedForwardNetwork.py  (same ones imported in the original notebook).

Run:
    python equiformer_train.py
"""

import math
import numpy as np

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from e3nn import o3
from tqdm import tqdm
from collections.abc import Callable

from ase import Atoms
from ase.neighborlist import neighbor_list
from mendeleev import element
from fairchem.core.datasets import AseDBDataset

import matplotlib.pyplot as plt

# Local project modules
from Embedding import build_edge_inputs, GaussianRBF, EmbeddingLayer
from LayerNorm import EquivariantLayerNormV2
from GraphAttention import GraphAttention, ScaledScatter, Activation
from FeedForwardNetwork import FeedForwardNetwork


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATAPATH = "/home/bandarua/Data/train/rattled-relax/"
NUM_EPOCHS = 100
WARMUP_STEPS = 5
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Periodic-table feature generators
# ---------------------------------------------------------------------------
def feature_generator_period_num():
    atomic_nums = np.arange(1, 119)
    period_num = np.zeros(shape=len(atomic_nums))
    for num in tqdm(atomic_nums, desc="Period Num"):
        period_num[num - 1] = element(int(num)).period
    return period_num


def get_group_number(atomic_number):
    if 57 <= atomic_number <= 71:
        return 19
    elif 89 <= atomic_number <= 103:
        return 20
    else:
        ele = element(atomic_number)
        return ele.group.group_id


def feature_generator_group_num():
    atomic_nums = np.arange(1, 119)
    group_num = np.zeros(shape=len(atomic_nums))
    for num in tqdm(atomic_nums, desc="Group Number"):
        group_num[num - 1] = get_group_number(int(num))
    return group_num


def get_electronegativity(atomic_number: int):
    ele = element(atomic_number)
    return ele.electronegativity('allred-rochow')


def feature_generator_electronegativity():
    atomic_nums = np.arange(1, 119)
    electronegativity = np.zeros(shape=len(atomic_nums))
    for num in tqdm(atomic_nums, desc="Electronegativity"):
        electronegativity[num - 1] = get_electronegativity(int(num))
    return electronegativity


# These arrays are used by the Node class below; populated in main().
period_num_array = None
group_num_array = None
electronegativity_array = None


# ---------------------------------------------------------------------------
# Graph / dataset construction
# ---------------------------------------------------------------------------
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


def average_degree_per_graph(edge_index: torch.Tensor):
    dict_node_degrees = {}
    for node in edge_index[0]:
        dict_node_degrees[node.item()] = dict_node_degrees.get(node.item(), 0) + 1
    return sum(dict_node_degrees.values()) / len(dict_node_degrees)


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super().__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps, o3.Irreps('{}x0e'.format(self.num_irreps)))

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        return self.mul(x, mask)


class TensorProductRescale(torch.nn.Module):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, instructions,
                 bias=True, rescale=True, internal_weights=None,
                 shared_weights=None, normalization=None):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        self.tp = o3.TensorProduct(
            irreps_in1=self.irreps_in1, irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out, instructions=instructions,
            normalization=normalization, internal_weights=internal_weights,
            shared_weights=shared_weights, path_normalization='none')

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
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()

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
            slices_fan_in = {}
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] + fan_in
                                            if slice_idx in slices_fan_in else fan_in)
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):
        out = self.tp(x, y, weight)
        if self.use_bias:
            for (_, slice, bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        return self.forward_tp_rescale_bias(x, y, weight)


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
                 bias=True, rescale=True, internal_weights=None,
                 shared_weights=None, normalization=None):
        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out,
                         instructions=instructions, bias=bias, rescale=rescale,
                         internal_weights=internal_weights,
                         shared_weights=shared_weights, normalization=normalization)


class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True, irreps_2=o3.Irreps('1x0e')):
        super().__init__(irreps_in, irreps_2, irreps_out,
                         bias=bias, rescale=rescale, internal_weights=True,
                         shared_weights=True, normalization=None)

    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        return self.forward_tp_rescale_bias(x, y)


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
        return result_2 + result_4


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
        self.transformer_1 = Transformer(irreps_node=irreps_node, irreps_node_attr=irreps_node_attr,
                                         irreps_edge_attr=irreps_edge_attr, fc_neurons=fc_neurons,
                                         irreps_head=irreps_head, num_heads=num_heads,
                                         alpha_drop=alpha_drop, proj_drop=proj_drop)
        self.transformer_2 = Transformer(irreps_node=irreps_node, irreps_node_attr=irreps_node_attr,
                                         irreps_edge_attr=irreps_edge_attr, fc_neurons=fc_neurons,
                                         irreps_head=irreps_head, num_heads=num_heads,
                                         alpha_drop=alpha_drop, proj_drop=proj_drop)
        self.transformer_3 = Transformer(irreps_node=irreps_node, irreps_node_attr=irreps_node_attr,
                                         irreps_edge_attr=irreps_edge_attr, fc_neurons=fc_neurons,
                                         irreps_head=irreps_head, num_heads=num_heads,
                                         alpha_drop=alpha_drop, proj_drop=proj_drop)
        self.last_layer_norm = EquivariantLayerNormV2(irreps=irreps_node)
        self.last_ffn = FeedForwardNetwork(irreps_node_input=irreps_node,
                                           irreps_node_attr=irreps_node_attr,
                                           irreps_node_output=irreps_mlp_mid,
                                           irreps_mlp_mid=irreps_mlp_mid)
        self.last_scatter = ScaledScatter(avg_aggregate_num=avg_aggregate_num_2)

        self.head = nn.Sequential(
            LinearRS(irreps_in=o3.Irreps(irreps_final), irreps_out=o3.Irreps(irreps_final)),
            Activation(o3.Irreps(irreps_final), acts=[nn.SiLU(), None, None, None, None, None]),
            LinearRS(o3.Irreps(irreps_final), o3.Irreps(output))
        )

        # print("irreps_final:", o3.Irreps(irreps_final))
        # print("output:", o3.Irreps(output))
        lin2 = self.head[2]
        # print("instructions:", lin2.tp.instructions)   # expect []
        # print("weight_numel:", lin2.tp.weight_numel)    # expect 0

    def forward(self, input, edge_src, edge_dst, edge_attr, edge_scalars, batch):
        result, node_attr_x = self.embedder(input, edge_src, edge_dst, edge_attr, edge_scalars, batch)
        result_1 = self.transformer_1(result, edge_src, edge_dst, edge_attr, edge_scalars, batch, node_attr_x)
        result_2 = self.transformer_2(result_1, edge_src, edge_dst, edge_attr, edge_scalars, batch, node_attr_x)
        result_3 = self.transformer_3(result_2, edge_src, edge_dst, edge_attr, edge_scalars, batch, node_attr_x)
        result_4 = self.last_layer_norm(result_3)
        result_5 = self.last_ffn(result_4, node_attr_x)
        x = result_5
        x = self.head[0](x)
        x = self.head[1](x)
        x = self.head[2](x)
        return self.last_scatter(x, batch)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def make_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ---------------------------------------------------------------------------
# Train / test steps
# ---------------------------------------------------------------------------
def train_step(epoch, model, rbf, dataloader, loss_fn, optimizer, device,
               scheduler_func, disable_progress_bar=False,
               irreps_edge_attr=o3.Irreps("1x0e+1x1e+1x2e")):
    model.train()
    train_loss = 0
    progress_bar = tqdm(enumerate(dataloader), desc=f"Training Epoch {epoch}",
                        total=len(dataloader), disable=disable_progress_bar)
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=scheduler_func)

    for batch_num, batch_data in progress_bar:
        X, y = batch_data.x.to(device), batch_data.y.unsqueeze(-1).to(device)
        edge_src, edge_dst, edge_attr, edge_scalars, batch = build_edge_inputs(
            data=batch_data, irreps_edge_attr=irreps_edge_attr, rbf=rbf)
        y_pred = model(X, edge_src.to(device), edge_dst.to(device), edge_attr.to(device),
                       edge_scalars.to(device), batch.to(device))
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({"train_loss": train_loss / (batch_num + 1)})

    return train_loss / len(dataloader)


def test_step(epoch, model, rbf, dataloader, loss_fn, device,
              disable_progress_bar=False,
              irreps_edge_attr=o3.Irreps("1x0e+1x1e+1x2e")):
    model.eval()
    test_loss = 0
    progress_bar = tqdm(enumerate(dataloader), desc=f"Test Epoch {epoch}",
                        total=len(dataloader), disable=disable_progress_bar)
    with torch.inference_mode():
        for batch_num, batch_data in progress_bar:
            X, y = batch_data.x.to(device), batch_data.y.unsqueeze(-1).to(device)
            edge_src, edge_dst, edge_attr, edge_scalars, batch = build_edge_inputs(
                data=batch_data, irreps_edge_attr=irreps_edge_attr, rbf=rbf)
            y_pred = model(X, edge_src.to(device), edge_dst.to(device), edge_attr.to(device),
                           edge_scalars.to(device), batch.to(device))
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            progress_bar.set_postfix({"test_loss": test_loss / (batch_num + 1)})
    test_loss /= len(dataloader)
    # print(f"Test Loss: {test_loss:.4f}")
    return test_loss


def train(model, rbf, train_dataloader, test_dataloader, optimizer, loss_fn,
          epochs, device, scheduler_func, disable_progress_bar=False):
    results = {"train_loss": [], "test_loss": []}
    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        tr = train_step(epoch, model, rbf, train_dataloader, loss_fn, optimizer,
                        device, scheduler_func, disable_progress_bar)
        te = test_step(epoch, model, rbf, test_dataloader, loss_fn, device)
        print(f"Epoch: {epoch+1} | train_loss: {tr:.4f} | test_loss: {te:.4f}")
        results["train_loss"].append(tr)
        results["test_loss"].append(te)
    return results


def plot_loss_curves(loss: dict):
    epochs = range(len(loss['train_loss']))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss['train_loss'], label="Training Loss", marker="o")
    plt.plot(epochs, loss['test_loss'], label="Testing Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curves.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global period_num_array, group_num_array, electronegativity_array

    # --- Load dataset ---
    ase_dataset = AseDBDataset(config=dict(src=DATAPATH))

    # --- Random 10% split for train pool ---
    split = int(len(ase_dataset) * 0.10)
    possible_idx = np.arange(0, len(ase_dataset))
    idx = np.random.choice(len(possible_idx), size=split, replace=False)

    # --- Gather train compounds ---
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

    # --- Build periodic-table feature arrays ---
    period_num_array = feature_generator_period_num()
    group_num_array = feature_generator_group_num()
    electronegativity_array = feature_generator_electronegativity()

    # --- Build training graphs ---
    graph_list = []
    for compound in tqdm(ionic_compounds_PE, desc="Building graphs"):
        graph_obj = Graph(ionic_compound=compound)
        if graph_obj.edges.shape[0] > 0:
            graph_list.append(graph_obj)

    train_dataset = OMAT24Graphs(graph_list=graph_list)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, persistent_workers=False)

    # --- Aggregation statistics ---
    average_degrees_all_graphs = [average_degree_per_graph(edge_index=data.edge_index)
                                  for data in tqdm(train_dataset, desc="Graph Degrees")]
    _AVG_DEGREE = sum(average_degrees_all_graphs) / len(average_degrees_all_graphs)

    nodes_num_all_graph = [data.x.shape[0] for data in tqdm(train_dataset, desc="Graph Nodes")]
    _AVG_NODES = sum(nodes_num_all_graph) / len(nodes_num_all_graph)

    # --- Test split (from remaining indices) ---
    mask = np.ones(len(possible_idx), dtype=bool)
    mask[idx] = False
    new_possible_idx = possible_idx[mask, ...]
    two_percent_split = int(len(new_possible_idx) * 0.02)
    idx_test = np.random.choice(len(new_possible_idx), size=two_percent_split, replace=False)

    ionic_compounds_test = []
    for i, index in tqdm(enumerate(idx_test), desc="Test Compounds"):
        ionic_compounds_test.append(ase_dataset.get_atoms(index))

    ionic_compounds_PE_test = []
    for compound in tqdm(ionic_compounds_test, desc="Test Compounds PE"):
        try:
            compound.get_potential_energy()
            ionic_compounds_PE_test.append(compound)
        except NotImplementedError:
            print("Does not have the PE value!")
    print(len(ionic_compounds_PE_test))

    graph_list_test = []
    for compound in tqdm(ionic_compounds_PE_test, desc="Building Test Graphs"):
        graph_obj = Graph(ionic_compound=compound)
        if graph_obj.edges.shape[0] > 0:
            graph_list_test.append(graph_obj)

    test_dataset = OMAT24Graphs(graph_list=graph_list_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=0, persistent_workers=False)

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)

    # --- Model ---
    model = Equiformer(
        irreps_node="128x0e+32x0o+32x1e+32x1o+16x2e+16x2o",
        num_features=5,
        bias=True,
        irreps_edge_attr="1x0e+1x1o+1x2e",
        fc_neurons=[128, 64],
        avg_aggregate_num_1=_AVG_DEGREE,
        irreps_head="32x0e+8x0o+8x1e+8x1o+4x2e+4x2o",
        num_heads=4,
        alpha_drop=0.2,
        proj_drop=0.2,
        irreps_node_attr="5x0e",
        irreps_mlp_mid="384x0e+96x0o+96x1e+96x1o+48x2e+48x2o",
        irreps_final="384x0e+96x0o+96x1e+96x1o+48x2e+48x2o",
        output="1x0e",
        avg_aggregate_num_2=_AVG_NODES,
    ).to(device)

    # --- Loss / optimizer (weight-decay split) ---
    loss_fn = torch.nn.L1Loss()
    decay, no_decay = [], []
    for name, param in tqdm(model.named_parameters(), desc="Optimization Params"):
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 5e-3},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=1e-4)

    # --- RBF ---
    rbf = GaussianRBF().to(device)

    # --- LR schedule ---
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler_func = make_lr_lambda(WARMUP_STEPS, total_steps)

    # --- Train ---
    result = train(model=model, rbf=rbf,
                   train_dataloader=train_dataloader,
                   test_dataloader=test_dataloader,
                   optimizer=optimizer, loss_fn=loss_fn,
                   epochs=NUM_EPOCHS, device=device,
                   scheduler_func=scheduler_func)

    # print("Final:", result['train_loss'][-1], result['test_loss'][-1])
    plot_loss_curves(loss=result)

    # --- Save model ---
    torch.save(model.state_dict(), "model_500_epochs_energy_1.pth")


if __name__ == "__main__":
    main()
