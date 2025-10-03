import torch
import torch.nn as nn
from torch_scatter import scatter

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:,0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))  # attention
        # alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr)))) #ablation study remove h_qr
        # replace to softmax
        # alpha = torch.softmax(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))), dim=1)

        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new

class MRD_GNN(torch.nn.Module):
    def __init__(self, params, loader, node_embeddings=None, use_layer_attention=True):
        super(MRD_GNN, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        self.use_layer_attention = use_layer_attention

        # pretrain embedding
        if node_embeddings is not None:
            self.node_embeddings = torch.tensor(node_embeddings, dtype=torch.float32).to('cuda', non_blocking=True)
            #self.node_embeddings = torch.FloatTensor(node_embeddings).cuda()
            input_dim = 48
        else:
            self.node_embeddings = None
            input_dim = self.hidden_dim

        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        self.gnn_layers.append(GNNLayer(input_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        for i in range(1, self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        # layer Attention: simple learnable weights
        # Only used when use_layer_attention=True
        if use_layer_attention:
            self.layer_attn_weights = nn.Parameter(torch.ones(self.n_layer))


        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, mode='train'):
        n = len(subs)
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()

        all_hidden_states = []

        for j in range(1, self.n_layer + 1):  # 1到n_layer
            if self.node_embeddings is not None:
                hidden = self.node_embeddings[q_sub].cuda()
            else:
                hidden = torch.zeros(n, self.hidden_dim).cuda()

            h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
            nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)

            for i in range(j):  # 递归到第j跳
                nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
                hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
                h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
                hidden = self.dropout(hidden)
                hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
                hidden = hidden.squeeze(0)

            all_hidden_states.append(hidden)


        max_nodes = max([hidden.size(0) for hidden in all_hidden_states])

        padded_hidden_states = []
        for hidden in all_hidden_states:
            padding_size = max_nodes - hidden.size(0)
            if padding_size > 0:
                padding = torch.zeros(padding_size, hidden.size(1)).cuda()
                hidden = torch.cat([hidden, padding], dim=0)
            padded_hidden_states.append(hidden)

        all_hidden_states_stacked = torch.stack(padded_hidden_states, dim=0)
        # (n_layer, max_nodes, hidden_dim)

        if self.use_layer_attention:
            layer_attn_scores = torch.softmax(self.layer_attn_weights, dim=0)  # (n_layer,)
            layer_attn_scores = layer_attn_scores.view(-1, 1, 1)  # (n_layer, 1, 1)
            weighted_hidden = all_hidden_states_stacked * layer_attn_scores  # (n_layer, max_nodes, hidden_dim)
            final_representation = weighted_hidden.sum(dim=0)  # (max_nodes, hidden_dim)
        else:
            final_representation = all_hidden_states_stacked[-1]  # (max_nodes, hidden_dim)

        scores = self.W_final(final_representation).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()         # non_visited entities have 0 scores
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        return scores_all



