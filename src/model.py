'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''

from torch import nn, Tensor
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_geometric.utils import degree


class MCCF(MessagePassing):    
    def __init__(self, 
                 option,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 num_layers=3,
                 add_self_loops = False,
                 mu = 0,
                 drop= 0,
                 device = 'cpu',
                 verbose = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.edge_index_norm = None
        self.verbose = verbose
        self.mu = mu
        self.option = option
        self.dropout = drop
        self.device = device
        
        print("Model: MCCF | ", "Option:", option, " | Layers:", num_layers, " | emb dimension:", embedding_dim, " | dropout:", drop)
        
        self.user_baseline = False
        self.item_baseline = False
        self.u_abs_drift = False
        self.u_rel_drift = False
        
        self.users_emb_final = None
        self.items_emb_final = None
        
        if option == 'lgcn_b':  # lgcn + baseline
            self.item_baseline = True
            self.user_baseline = True
        elif option == 'lgcn_b_a':  # lgcn + baseline + absolute
            self.u_abs_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_b_r':  # lgcn + baseline + relative
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_b_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
        else: # pure lightGCN model only
            option = 'lgcn'
            self.mu = 0
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim).to(self.device)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim).to(self.device)
        
        self.users_emb.weight.requires_grad = True
        self.items_emb.weight.requires_grad = True
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        if self.user_baseline:
            self._u_base_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._u_base_emb.weight)
            self._u_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The user baseline embedding is ON.")
        
        if self.item_baseline:
            self._i_base_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._i_base_emb.weight)
            self._i_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The item baseline embedding is ON.")

        if self.u_abs_drift:
            self._u_abs_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)  
            nn.init.zeros_(self._u_abs_drift_emb.weight)
            self._u_abs_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The absolute user drift temporal embedding is ON.")

        if self.u_rel_drift:
            self._u_rel_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)   
            nn.init.zeros_(self._u_rel_drift_emb.weight)
            self._u_rel_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The relative user drift temporal embedding is ON.")
                
        self.f = nn.ReLU()
        #self.f = nn.SiLU()
              
    def forward(self, edge_index: Tensor, src: Tensor, dest: Tensor, u_abs_decay: Tensor, u_rel_decay: Tensor, i_rel_decay: Tensor):
        
        if(self.edge_index_norm is None):
            self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
                  
        u_emb_0 = self.users_emb.weight
        i_emb_0 = self.items_emb.weight 
        
        emb_0 = torch.cat([u_emb_0, i_emb_0])
        embs = [emb_0]
        emb_k = emb_0
        
        #if(self.edge_index_norm is None):
        #    # Compute normalization
        #    from_, to_ = edge_index
        #    deg = degree(to_, self.num_users + self.num_items, dtype=emb_k.dtype)
        #    deg_inv_sqrt = deg.pow(-0.5)
        #    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #    self.edge_index_norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
    
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=self.edge_index_norm[0], x=emb_k, norm=self.edge_index_norm[1])
            embs.append(emb_k)
             
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)          
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        self.users_emb_final = users_emb_final
        self.items_emb_final = items_emb_final
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
        
        _inner_pro = torch.mul(user_embeds, item_embeds)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[src]
            _inner_pro = _inner_pro + _u_base_emb
            
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[dest]
            _inner_pro = _inner_pro + _i_base_emb
            
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[src]
            _u_abs_drift_emb = _u_abs_drift_emb * u_abs_decay.unsqueeze(1)
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[src]
            _u_rel_drift_emb = _u_rel_drift_emb * u_rel_decay.unsqueeze(1) 
            _inner_pro = _inner_pro + _u_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.option != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
            ratings = self.f(_inner_pro)
        else:
            ratings = _inner_pro
        
        #ratings = self.f(_inner_pro)       
        return ratings
    
    def message(self, x_j, norm):
        out =  x_j * norm.view(-1, 1)
                
        return out
    
    #def aggregate(self, edge_index, x, norm, edge_weight=None):
    #    row, col = edge_index
    #    out = self.message(x[col], norm, edge_weight)
    #    out = torch_scatter.scatter_add(out, row, dim=0, dim_size=x.size(0))
    #    #  out = torch_scatter.scatter(inputs, index, dim=0, reduce='sum')
    #    return out
    
    def predict(self, u_id, i_id):
        
        user_embed = self.users_emb_final[u_id]
        item_embed = self.items_emb_final[i_id]
        
        _inner_pro = torch.mul(user_embed, item_embed)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[u_id]
            _inner_pro = _inner_pro + _u_base_emb
        
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[i_id]
            _inner_pro = _inner_pro + _i_base_emb
        
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[u_id]
            #_u_abs_drift_emb = _u_abs_drift_emb * u_abs_t_decay.unsqueeze(1)
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[u_id]
            #_u_rel_drift_emb = _u_rel_drift_emb * u_rel_t_decay.unsqueeze(1) 
            _inner_pro = _inner_pro + _u_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.option != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
        
        rating = self.f(_inner_pro)
              
        return rating

    def save_embeddings(self, path):
        torch.save({
            'users_emb_final': self.users_emb_final,
            'items_emb_final': self.items_emb_final
        }, path)

    def load_embeddings(self, path):
        checkpoint = torch.load(path)
        self.users_emb_final = checkpoint['users_emb_final']
        self.items_emb_final = checkpoint['items_emb_final']
        
class NGCF(MessagePassing):    
    def __init__(self, 
                 option,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 num_layers=3,
                 add_self_loops = False,
                 mu = 0,
                 drop= 0.1,
                 device = 'cpu',
                 verbose = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.edge_index_norm = None
        self.verbose = verbose
        self.mu = mu
        self.option = option
        self.dropout = drop
        self.device = device
        self.bias = True
        
        print("Model: NCGF, | Option:", option, " | Layers:", num_layers, " | emb dimension:", embedding_dim, " | dropout:", drop)
        
        self.users_emb_final = None
        self.items_emb_final = None
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim).to(self.device)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim).to(self.device)
        
        self.users_emb.weight.requires_grad = True
        self.items_emb.weight.requires_grad = True
        
        self.lin_1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=self.bias)
        self.lin_2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=self.bias)
        
        self.lin_1.weight.requires_grad = True
        self.lin_2.weight.requires_grad = True
        
        self.init_parameters()
        
        torch.autograd.set_detect_anomaly(True)
                
        self.f = nn.ReLU()
    
    def init_parameters(self):
        nn.init.xavier_uniform_(self.items_emb.weight, gain=1)
        nn.init.xavier_uniform_(self.users_emb.weight, gain=1)
        
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)
              
    def forward(self, edge_index: Tensor, src: Tensor, dest: Tensor, u_abs_decay: Tensor, u_rel_decay: Tensor, i_rel_decay: Tensor):
        
        if(self.edge_index_norm is None):
            self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
                  
        #u_emb_0 = self.users_emb.weight + self._u_abs_drift_emb.weight + self._u_rel_drift_emb.weight + self._u_base_emb.weight
        u_emb_0 = self.users_emb.weight
        i_emb_0 = self.items_emb.weight
        
        emb_0 = torch.cat([u_emb_0, i_emb_0])
        embs = [emb_0]
        emb_k = emb_0
        
        #if(self.edge_index_norm is None):
        #    # Compute normalization
        #    from_, to_ = edge_index
        #    deg = degree(to_, self.num_users + self.num_items, dtype=emb_k.dtype)
        #    deg_inv_sqrt = deg.pow(-0.5)
        #    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #    self.edge_index_norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
    
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=self.edge_index_norm[0], x=(emb_k, emb_k) , norm=self.edge_index_norm[1])
            emb_k = emb_k + self.lin_1(emb_k)
            emb_k = F.dropout(emb_k, self.dropout, self.training)
            emb_k = F.leaky_relu(emb_k)
            embs.append(emb_k)
             
        #embs = torch.stack(embs, dim=1)
        emb_final = torch.cat(embs, dim=-1)          
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        self.users_emb_final = users_emb_final
        self.items_emb_final = items_emb_final
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
        
        _inner_pro = torch.mul(user_embeds, item_embeds)
                 
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        ratings = _inner_pro
        
        #ratings = self.f(_inner_pro)       
        return ratings
    
    def message(self, x_j, x_i, norm):
        out =  norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))
                
        return out
        
class tempLGCN(MessagePassing):    
    def __init__(self, 
                 option,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 num_layers=3,
                 add_self_loops = False,
                 mu = 0,
                 drop= 0,
                 device = 'cpu',
                 verbose = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.edge_index_norm = None
        self.verbose = verbose
        self.mu = mu
        self.option = option
        self.dropout = drop
        self.device = device
        
        print("Model: tempLGCN, | Option:", option, " | Layers:", num_layers, " | emb dimension:", embedding_dim, " | dropout:", drop)
        
        self.user_baseline = False
        self.item_baseline = False
        self.u_abs_drift = False
        self.u_rel_drift = False
        
        self.users_emb_final = None
        self.items_emb_final = None
        
        if option == 'lgcn_b':  # lgcn + baseline
            self.item_baseline = True
            self.user_baseline = True
        elif option == 'lgcn_b_a':  # lgcn + baseline + absolute
            self.u_abs_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_b_r':  # lgcn + baseline + relative
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_b_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
        else: # pure lightGCN model only
            option = 'lgcn'
            self.mu = 0
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim).to(self.device)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim).to(self.device)
        
        self.users_emb.weight.requires_grad = True
        self.items_emb.weight.requires_grad = True
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        if self.user_baseline:
            self._u_base_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._u_base_emb.weight)
            self._u_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The user baseline embedding is ON.")
        
        if self.item_baseline:
            self._i_base_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._i_base_emb.weight)
            self._i_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The item baseline embedding is ON.")

        if self.u_abs_drift:
            self._u_abs_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)  
            nn.init.zeros_(self._u_abs_drift_emb.weight)
            self._u_abs_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The absolute user drift temporal embedding is ON.")

        if self.u_rel_drift:
            self._u_rel_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)   
            nn.init.zeros_(self._u_rel_drift_emb.weight)
            self._u_rel_drift_emb.weight.requires_grad = True
            if self.verbose:
                print("The relative user drift temporal embedding is ON.")
                
        self.f = nn.ReLU()
              
    def forward(self, edge_index: Tensor, src: Tensor, dest: Tensor, u_abs_decay: Tensor, u_rel_decay: Tensor, i_rel_decay: Tensor):
        
        if(self.edge_index_norm is None):
            self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
                  
        u_emb_0 = self.users_emb.weight
        
        if self.u_abs_drift:
            u_emb_0 = u_emb_0 + self._u_abs_drift_emb.weight
            
        if self.u_rel_drift:
            u_emb_0 = u_emb_0 + self._u_rel_drift_emb.weight
        
        if self.user_baseline:
            u_emb_0 = u_emb_0 + self._u_base_emb.weight
            
        i_emb_0 = self.items_emb.weight
        
        emb_0 = torch.cat([u_emb_0, i_emb_0])
        embs = [emb_0]
        emb_k = emb_0
        
        #if(self.edge_index_norm is None):
        #    # Compute normalization
        #    from_, to_ = edge_index
        #    deg = degree(to_, self.num_users + self.num_items, dtype=emb_k.dtype)
        #    deg_inv_sqrt = deg.pow(-0.5)
        #    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #    self.edge_index_norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
    
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=self.edge_index_norm[0], x=emb_k, norm=self.edge_index_norm[1])
            embs.append(emb_k)
             
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)          
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        self.users_emb_final = users_emb_final
        self.items_emb_final = items_emb_final
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
        
        _inner_pro = torch.mul(user_embeds, item_embeds)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[src]
            _inner_pro = _inner_pro + _u_base_emb
            
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[dest]
            _inner_pro = _inner_pro + _i_base_emb
            
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[src]
            _u_abs_drift_emb = _u_abs_drift_emb * u_abs_decay.unsqueeze(1)
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[src]
            _u_rel_drift_emb = _u_rel_drift_emb * u_rel_decay.unsqueeze(1) 
            _inner_pro = _inner_pro + _u_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.option != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
            ratings = self.f(_inner_pro) 
        else:
            ratings = _inner_pro
        
        return ratings
    
    def message(self, x_j, norm):
        out =  x_j * norm.view(-1, 1)
                
        return out
    

class tempLGCN_attn(MessagePassing):    
    def __init__(self, 
                 option,
                 num_users,
                 num_items,
                 embedding_dim=64,
                 num_layers=3,
                 add_self_loops = False,
                 mu = 0,
                 drop= 0,
                 device = 'cpu',
                 verbose = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops
        self.edge_index_norm = None
        self.verbose = verbose
        self.mu = mu
        self.option = option
        self.dropout = drop
        self.device = device
        
        print("Model: tempLGCN_attn, | Option:", option, " | Layers:", num_layers, " | emb dimension:", embedding_dim, " | dropout:", drop)
        
        self.user_baseline = False
        self.item_baseline = False
        self.u_abs_drift = False
        self.u_rel_drift = False
        
        self.users_emb_final = None
        self.items_emb_final = None
        
        if option == 'lgcn_b':  # lgcn + baseline
            self.item_baseline = True
            self.user_baseline = True
        elif option == 'lgcn_b_a':  # lgcn + baseline + absolute
            self.u_abs_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_b_r':  # lgcn + baseline + relative
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_b_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
            self.user_baseline = True
            self.item_baseline = True
        elif option == 'lgcn_ar': # lgcn + baseline + absolute + relative 
            self.u_abs_drift = True
            self.u_rel_drift = True
        else: # pure lightGCN model only
            option = 'lgcn'
            self.mu = 0
        
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim).to(self.device)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim).to(self.device)
        
        self.users_emb.weight.requires_grad = True
        self.items_emb.weight.requires_grad = True
        
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        if self.user_baseline:
            self._u_base_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._u_base_emb.weight)
            self._u_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The user baseline embedding is ON.")
        
        if self.item_baseline:
            self._i_base_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim).to(self.device)
            nn.init.zeros_(self._i_base_emb.weight)
            self._i_base_emb.weight.requires_grad = True
            if self.verbose:
                print("The item baseline embedding is ON.")

        if self.u_abs_drift:
            self._u_abs_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)  
            nn.init.zeros_(self._u_abs_drift_emb.weight)
            self._u_abs_drift_emb.weight.requires_grad = True
            
            # Initialize the period as a trainable parameter (starting with some default value)
            self.period = nn.Parameter(torch.tensor([1.0]))  # Starting with 24 (for daily cycles, for example)
            self.phase_shift = nn.Parameter(torch.tensor([0.0]))  # Optional, can also be trainable if needed
            
            self._u_abs_beta_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.embedding_dim).to(self.device)  
            nn.init.zeros_(self._u_abs_beta_emb.weight)
            self._u_abs_beta_emb.weight.requires_grad = True
            
            if self.verbose:
                print("The absolute user drift temporal embedding is ON.")

        if self.u_rel_drift:
            self._u_rel_drift_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim).to(self.device)   
            nn.init.zeros_(self._u_rel_drift_emb.weight)
            self._u_rel_drift_emb.weight.requires_grad = True
            
            self._u_rel_beta_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.embedding_dim).to(self.device)  
            nn.init.zeros_(self._u_rel_beta_emb.weight)
            self._u_rel_beta_emb.weight.requires_grad = True
            
            if self.verbose:
                print("The relative user drift temporal embedding is ON.")
                
        self.f = nn.ReLU()
              
    def forward(self, edge_index: Tensor, src: Tensor, dest: Tensor, u_abs_decay: Tensor, u_rel_decay: Tensor, i_rel_decay: Tensor):
        
        if(self.edge_index_norm is None):
            self.edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
                  
        u_emb_0 = self.users_emb.weight
        
        if self.u_abs_drift:
            u_emb_0 = u_emb_0 + self._u_abs_drift_emb.weight
            
        if self.u_rel_drift:
            u_emb_0 = u_emb_0 + self._u_rel_drift_emb.weight
        
        if self.user_baseline:
            u_emb_0 = u_emb_0 + self._u_base_emb.weight
            
        i_emb_0 = self.items_emb.weight
        
        emb_0 = torch.cat([u_emb_0, i_emb_0])
        embs = [emb_0]
        emb_k = emb_0
        
        #if(self.edge_index_norm is None):
        #    # Compute normalization
        #    from_, to_ = edge_index
        #    deg = degree(to_, self.num_users + self.num_items, dtype=emb_k.dtype)
        #    deg_inv_sqrt = deg.pow(-0.5)
        #    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #    self.edge_index_norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
    
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index=self.edge_index_norm[0], x=emb_k, norm=self.edge_index_norm[1])
            embs.append(emb_k)
             
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)          
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        
        self.users_emb_final = users_emb_final
        self.items_emb_final = items_emb_final
        
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]
        
        _inner_pro = torch.mul(user_embeds, item_embeds)
          
        if self.user_baseline:
            _u_base_emb = self._u_base_emb.weight[src]
            _inner_pro = _inner_pro + _u_base_emb
            
        if self.item_baseline:
            _i_base_emb = self._i_base_emb.weight[dest]
            _inner_pro = _inner_pro + _i_base_emb
            
        if self.u_abs_drift:
            _u_abs_drift_emb = self._u_abs_drift_emb.weight[src]
            
            #abs_decay = torch.sigmoid(u_abs_decay.unsqueeze(1) * self._u_abs_beta_emb.weight) # much less memory
            
            # Use the learnable period and phase shift in the cosine function
            abs_decay = torch.cos((2 * torch.pi * u_abs_decay.unsqueeze(1) / self.period) + self.phase_shift)

            _u_abs_drift_emb = _u_abs_drift_emb * abs_decay
            _inner_pro = _inner_pro + _u_abs_drift_emb
            
        if self.u_rel_drift:
            _u_rel_drift_emb = self._u_rel_drift_emb.weight[src]
            rel_decay = torch.sigmoid(u_rel_decay.unsqueeze(1) * self._u_rel_beta_emb.weight)
            #rel_decay = torch.tanh_(u_rel_decay.unsqueeze(1) * self._u_rel_beta_emb.weight)
            #rel_decay = torch.tanh(self._u_rel_beta_emb.weight)
            _u_rel_drift_emb = _u_rel_drift_emb * rel_decay
            _inner_pro = _inner_pro + _u_rel_drift_emb
             
        _inner_pro = torch.sum(_inner_pro, dim=-1)
        
        if self.option != 'lgcn': 
            _inner_pro = _inner_pro + self.mu
            ratings = self.f(_inner_pro) 
        else:
            ratings = _inner_pro
        
        return ratings
    
    def message(self, x_j, norm):
        
        out =  x_j * norm.view(-1, 1)        
        return out
    
