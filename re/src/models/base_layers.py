# -*- encoding: utf-8 -*-
"""
@File    :   base_layers.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/5/8 8:36   
@Description :   None 

"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertLayer
from ipdb import set_trace
from einops import rearrange

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class lstm_FCLayer(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim,dropout_rate):
        super(lstm_FCLayer,self).__init__()
        self.lstm_1 = nn.LSTM(input_dim,hidden_dim,num_layers=2,bidirectional=True,dropout= dropout_rate,batch_first=True)
        self.linear_fc = nn.Linear(hidden_dim*2,output_dim)
        for name, param in self.lstm_1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.linear_fc.weight)
        nn.init.constant_(self.linear_fc.bias, 0.)
    def forward(self,x,y):
        x,(y1,y2) = self.lstm_1(x,y)
        return self.linear_fc(x),(y1,y2)

class Simple1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_rate, seq_len , kernel_size = 3, stride = 1, padding = 1):
        super(Simple1DCNN, self).__init__()

        # 1D卷积层
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(out_channels)
        # self.pool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(out_channels * (seq_len//2),  out_channels)
        # self.fc1 = nn.Linear(out_channels * (seq_len - kernel_size +1),  out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear( out_channels,  out_channels)

    def forward(self, x):
        # 输入大小：(batch_size, seq_len, hidden_size)

        # 1D卷积层
        # set_trace()
        x = x.permute(0, 2, 1)  # 将输入的维度顺序调整为 (batch_size, hidden_size, seq_len)
        x = self.conv1d(x)
        # set_trace()
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1d(x)

        # 展平操作
        x = x.view(x.size(0), -1)


        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class Simple1DCNN_1(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_rate, seq_len , kernel_size = 1):
        super(Simple1DCNN_1, self).__init__()

        # 1D卷积层
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.pool1d = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.pool1d = nn.MaxPool1d(seq_len,seq_len)
        # 全连接层
        self.fc1 = nn.Linear(out_channels ,  out_channels)
        # self.fc1 = nn.Linear(out_channels * (seq_len - kernel_size +1),  out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear( out_channels,  out_channels)

    def forward(self, x):
        # 输入大小：(batch_size, seq_len, hidden_size)

        # 1D卷积层
        x = x.permute(0, 2, 1)  # 将输入的维度顺序调整为 (batch_size, hidden_size, seq_len)
        x = self.conv1d(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1d(x)

        # 展平操作
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc_transform = nn.Linear(input_size, input_size)
        self.fc_gate = nn.Linear(input_size, input_size)

    def forward(self, x):
        transform = F.relu(self.fc_transform(x))
        gate = torch.sigmoid(self.fc_gate(x))
        carry = 1 - gate

        transformed = transform * gate + x * carry

        return transformed

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # 定义权重矩阵
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        # 输出层权重矩阵
        self.W_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, query,key,value ,mask=None):
        # 线性变换，将输入分为多个头

        query = self.W_q(query).view(query.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.W_k(key).view(key.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.W_v( value).view(value.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))#batch,head,seq_len,seq_len
        
        if mask is not None:
            batch_size,seq_len = mask.shape
            mask = mask.view(batch_size,1,seq_len,1).expand(-1,self.num_heads,-1,-1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights[torch.isnan(attention_weights)] = 0.0
        # 加权和
        weighted_sum = torch.matmul(attention_weights, value)

        # 合并多个头的输出
        output = weighted_sum.transpose(1, 2).contiguous().view(query.size(0), -1, self.hidden_size)

        # 输出层变换
        output = self.W_o(output)
        # nan_mask = torch.isnan(output)

        # output = torch.where(nan_mask, torch.zeros_like(output), output)
        
        return output

class EncoderLayer(nn.Module):
    def __init__(self, config, dropout_rate=0.2):
        super(EncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        config.hidden_act='swish'
        self.bert_layer = BertLayer(config)


    def forward(self, x):

        x = self.dropout(x)
        x = self.bert_layer(x)[0]
        x = self.dropout(x)

        return x
#下面的GAT是针对于batch*hidden
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        W = self.W.to(h.device)
        Wh = torch.mm(h, W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        a = self.a.to(Wh.device)
        Wh1 = torch.matmul(Wh, a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    
#下面的GAT是针对于batch*seq_len*hidden
class SPACY_GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SPACY_GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.W2 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, j,adj):
        W1 = self.W1.to(h.device)
        Wh = torch.matmul(h, W1)  # h.shape: (batch, seq_len, in_features), Wh.shape: (batch, seq_len, out_features)
        W2 = self.W2.to(h.device)
        Wj = torch.matmul(j,W2)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # Softmax along the seq_len dimension
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        j_prime = torch.matmul(attention,Wj)
        output = torch.add(h_prime,j_prime)
        if self.concat:
            return F.elu(output)
        else:
            return output

    def _prepare_attentional_mechanism_input(self, Wh):
        a = self.a.to(Wh.device)
        Wh1 = torch.matmul(Wh, a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)  # Transpose to match dimensions
        return self.leakyrelu(e)
    
class Spacy_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(Spacy_GAT, self).__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.empty(size=(nfeat,nhid * nheads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attentions = [SPACY_GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SPACY_GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, y,adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,y,adj) for att in self.attentions], dim=2)  # Concatenate along the seq_len dimension
        x = F.dropout(x, self.dropout, training=self.training)
        W = self.W.to(x.device)
        y = torch.matmul(y,W)
        x = F.elu(self.out_att(x,y, adj))
        return F.log_softmax(x, dim=2)  # Softmax along the seq_len dimension

class QGAAttention(nn.Module):
    def __init__(self, hidden_dim, num_kv_heads=2, num_heads=8):
        super(QGAAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_head_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** 0.5
        
        self.query_proj = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.key_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.value_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_dim)

    def forward(self, x):
        # Project inputs
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape inputs to (batch_size, seq_len, num_heads, head_dim)
        batch_size, seq_len, _ = query.size()
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Rearrange for attention computation
        query = rearrange(query, "b n h d -> b h n d")
        key = rearrange(key, "b s h d -> b h s d")
        value = rearrange(value, "b s h d -> b h s d")
        query = rearrange(query, "b (h g) n d -> b g h n d", g=self.num_head_groups)
        
        # Calculate attention scores
        scores = torch.einsum("b g h n d, b h s d -> b g h n s", query, key)
        attention = F.softmax(scores / self.scale, dim=-1)
        
        # Apply attention to value
        out = torch.einsum("b g h n s, b h s d -> b g h n d", attention, value)
        
        # Reshape output back to original dimensions
        out = rearrange(out, "b g h n d -> b n (g h) d")
        
        # Final linear projection
        out = self.out_proj(out.view(batch_size, seq_len, -1))
        
        return out