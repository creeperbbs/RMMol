# -*- coding: utf-8 -*-
# @Author: Hongzhi Yao
# @Date:   2025-06-21 09:05:55
# @Last Modified by:   Hongzhi Yao
# @Last Modified time: 2025-7-5 16:33:14
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
num_atom_type = 119  
num_chirality_tag = 4
num_degree = 11
num_hs = 9
num_formal_charge = 11
num_implicit_valence = 7
num_hybridization_tag = 7
num_bond_type = 6
num_bond_direction = 3
from torch_geometric.utils import add_self_loops

class GraphDegreeConv(nn.Module):
    def __init__(self, node_size, edge_size, output_size, degree_list, device, batch_normalize=True):
        super(GraphDegreeConv, self).__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.output_size = output_size
        self.batch_normalize = batch_normalize
        self.device = device
        if self.batch_normalize:
            self.normalize = nn.BatchNorm1d(output_size, affine=False)
        self.bias = nn.Parameter(torch.zeros(1, output_size))
        self.linear = nn.Linear(node_size, output_size, bias=False)
        self.degree_list = degree_list
        self.degree_layer_list = nn.ModuleList()
        for degree in degree_list:
            self.degree_layer_list.append(nn.Linear(node_size + edge_size, output_size, bias=False).float())

    def forward(self, node_repr, edge_repr, neighbor_by_degree):
        degree_activation_list = []
        for d_idx, degree_layer in enumerate(self.degree_layer_list):
            degree = self.degree_list[d_idx]
            node_neighbor_list = neighbor_by_degree[degree]['node']
            edge_neighbor_list = neighbor_by_degree[degree]['edge']
            if degree == 0 and node_neighbor_list:
                
                zero = torch.zeros(len(node_neighbor_list), self.output_size).to(self.device).float()
                degree_activation_list.append(zero)
            else:
                if node_neighbor_list:
                    
                    
                    # (#nodes, #degree, node_size)
                    node_neighbor_repr = node_repr[node_neighbor_list, ...]
                    # (#nodes, #degree, edge_size)
                    edge_neighbor_repr = edge_repr[edge_neighbor_list, ...]
                    # (#nodes, #degree, node_size + edge_size)
                    stacked = torch.cat([node_neighbor_repr, edge_neighbor_repr], dim=2)
                    summed = torch.sum(stacked, dim=1, keepdim=False)
                    degree_activation = degree_layer(summed)
                    degree_activation_list.append(degree_activation)
        neighbor_repr = torch.cat(degree_activation_list, dim=0)
        self_repr = self.linear.float()(node_repr)
        # size = (#nodes, #output_size)
        activations = self_repr + neighbor_repr + self.bias.expand_as(self_repr)
        if self.batch_normalize:
            activations = self.normalize(activations)
        return F.relu(activations)
class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        self.aggr = aggr
        super(GINConv, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(3, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(3, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:, 0] = 5  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1]) + self.edge_embedding3(edge_attr[:, 2]) + self.edge_embedding4(edge_attr[:, 3])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

    
class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", feat_dim=512,
                 batch_size=32, degree_list=None, device=None,atom_emb_dim = 256,bond_emb_dim = 64):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.dec_mask_token = nn.Parameter(torch.zeros(1, emb_dim))
        self.gnn_type = gnn_type
        self.degree_list = degree_list or []
        self.device = device or torch.device('cpu')
        self.atom_emb_dim = atom_emb_dim
        self.bond_emb_dim = bond_emb_dim
        
        self.x_embedding1 = nn.Embedding(num_atom_type, atom_emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, atom_emb_dim)
        self.x_embedding3 = nn.Embedding(num_hybridization_tag, atom_emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree+1, atom_emb_dim)
        self.x_embedding5 = nn.Embedding(num_hs+1, atom_emb_dim)
        self.x_embedding6 = nn.Embedding(num_formal_charge+1, atom_emb_dim)
        self.x_embedding7 = nn.Embedding(num_implicit_valence+1, atom_emb_dim)

        
        self.edge_embedding1 = nn.Embedding(num_bond_type, bond_emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, bond_emb_dim)
        self.edge_embedding3 = nn.Embedding(3, bond_emb_dim)
        self.edge_embedding4 = nn.Embedding(3, bond_emb_dim)
        self.out_layers = nn.ModuleList()
        layers_sizes = [atom_emb_dim] + [emb_dim for layer in range(num_layer-1)] + [emb_dim]
        for input_size in layers_sizes:
            self.out_layers.append(nn.Linear(input_size, emb_dim))
            
        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer-1):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim))
            elif gnn_type == "degree":
                if _ == 0:
                    self.gnns.append(
                    GraphDegreeConv(
                        node_size=self.atom_emb_dim,
                        edge_size=self.bond_emb_dim,
                        output_size=emb_dim,
                        degree_list=degree_list,
                        device=self.device
                    )
                )
                else:
                    self.gnns.append(
                        GraphDegreeConv(
                            node_size=emb_dim,
                            edge_size=self.bond_emb_dim,
                            output_size=emb_dim,
                            degree_list=degree_list,
                            device=self.device
                        )
                    )
        self.gnns.append( GraphDegreeConv(
                            node_size=emb_dim,
                            edge_size=self.bond_emb_dim,
                            output_size=emb_dim,
                            degree_list=degree_list,
                            device=self.device))
        self.feat_lin_1 = nn.Linear(emb_dim, self.feat_dim)
        self.feat_lin_2 = nn.Linear(emb_dim, self.feat_dim)
        self.out_lin = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, self.feat_dim)
        )

    def forward(self, *argv):
        if len(argv) == 3:
            data = None
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        device = x.device
        deg = torch.bincount(edge_index.view(-1), minlength=x.size(0)).long()
        deg = deg // 2
        # x[:, 3] = deg.float()
        # deg = x[:,3].long()
        
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1]) + self.x_embedding3(x[:, 2])+ self.x_embedding4(x[:, 3]) + self.x_embedding5(x[:, 4])+ self.x_embedding6(x[:, 5]) + self.x_embedding7(x[:, 6])
        num_nodes = x.size(0)

        edge_emb = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1]) + self.edge_embedding3(edge_attr[:, 2]) + self.edge_embedding4(edge_attr[:, 3])
        sorted_deg, sorted_nodes = torch.sort(deg, descending=False)
        x_reconstructed = x[sorted_nodes].clone()
        to_map = sorted_nodes
        inv_perm = torch.zeros_like(to_map, device=device)
        inv_perm[to_map] = torch.arange(num_nodes, device=device)
        edge_index = inv_perm[edge_index]
        row, col = edge_index
        deg = deg[to_map]
        unique_deg, counts = torch.unique(deg, return_counts=True)
        neighbor_by_degree = []
        if data is not None and hasattr(data, 'batch'):
            graph = data.batch[to_map]
            num_graphs = int(data.batch.max().item()) + 1
        else:
            graph = torch.zeros(num_nodes, dtype=torch.long, device=device)
            num_graphs = 1
        batch_idx = [[] for i in range(num_graphs)]
        begin_index = 0
        index = 0
        
        for d in self.degree_list:
            node_groups = []
            edge_groups = []
            # node_mask = (deg == d)
            # node_indices = torch.where(node_mask)[0].tolist()
            if d in unique_deg:
                node_indices = counts[index]
                for node in range(begin_index,begin_index+node_indices):
                    batch_idx[int(graph[node].item())].append(node)
                    edge_mask = (row == node)
                    edge_ids = torch.where(edge_mask)[0].tolist()
                    
                    
                    neighbors = [col[e].item() for e in edge_ids]
                
                    edge_groups.append(edge_ids)
                    
                    node_groups.append(neighbors)
                begin_index+=node_indices
                index+=1
            neighbor_by_degree.append({
                'node': node_groups,
                'edge': edge_groups
            })
        
        def fingerprint_update(linear, node_repr):
            atom_activations =torch.nn.functional.softmax(linear(node_repr), dim=-1)
            return atom_activations
        # h_list = [x_reconstructed]
        
        # max_num_nodes = max([data.ptr.tolist()[i] - data.ptr.tolist()[i-1] for i in range(len(data.ptr.tolist())) if i > 0])
        if data is not None and hasattr(data, 'ptr'):
            max_num_nodes = max([data.ptr.tolist()[i] - data.ptr.tolist()[i-1] for i in range(len(data.ptr.tolist())) if i > 0])
        else:
            max_num_nodes = x.size(0)
        if data is not None and hasattr(data, 'num_graphs'):
            fingerprint_atom = torch.zeros(data.num_graphs,max_num_nodes, self.emb_dim, device=device).float()
        else:
            fingerprint_atom = torch.zeros(1,max_num_nodes, self.emb_dim, device=device).float()
        # fingerprint_atom = torch.zeros(data.num_graphs,max_num_nodes, self.emb_dim).to(self.device).float()
        atom_activations = torch.zeros( x_reconstructed.size(0), self.emb_dim, device=device).float()
        
        for layer_idx in range(self.num_layer):
            # (#nodes, #output_size)
            
            atom_activations += fingerprint_update(self.out_layers[layer_idx].float(), x_reconstructed.float()).to(device)
            
            
            x_reconstructed = self.gnns[layer_idx](x_reconstructed.float().to(device), edge_emb.to(device),
                                                    neighbor_by_degree)
        atom_activations += fingerprint_update(self.out_layers[-1],  x_reconstructed).to(device)

        for idx, atom_idx in enumerate(batch_idx):
            fingerprint_atom[idx][:len(atom_idx)] = atom_activations[atom_idx, ...]
        graph_rep = torch.sum(fingerprint_atom, dim=1)
        atom_activations_original = torch.zeros_like(atom_activations)
        atom_activations_original[to_map] = atom_activations
        return atom_activations_original, self.out_lin(graph_rep)
    
class GNNDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super().__init__()
        self._dec_type = gnn_type
        if gnn_type == "gin":
            self.conv = GINConv(hidden_dim, out_dim, aggr = "add")
        elif gnn_type == "linear":
            self.dec = torch.nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = torch.nn.PReLU()
        self.temp = 0.2


    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[mask_node_indices] = 0
            # x[mask_node_indices] = self.dec_token
            out = self.conv(x, edge_index, edge_attr)
            # out = F.softmax(out, dim=-1) / self.temp
        return out
