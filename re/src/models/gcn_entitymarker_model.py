# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这里的模型是single entity marker
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""

from ipdb import set_trace

from config import BertConfig
from src.models.base_layers import FCLayer
from src.models.bert_model import EntityMarkerBaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import math
import torch
from torch.nn.parameter import Parameter




class gcn_SingleEntityMarkersREModel(EntityMarkerBaseModel):
    def __init__(self, config: BertConfig):
        super(gcn_SingleEntityMarkersREModel, self).__init__(config)

        self.num_labels = config.num_labels
        
        self.config = config
        self.scheme = config.scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)
        self.sent_fc_layer = FCLayer(self.bert_config.hidden_size*3, self.entity_dim, self.config.dropout_prob)
        self.mid_size = 128
        self.output_size = 64
        self.mid_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_prob)
        )
        # self.GCN_layer_1 = GCNConv(self.bert_config.hidden_size*2,self.mid_size)
        # self.GCN_layer_2 = GCNConv(self.mid_size, self.mid_size)
        # self.GCN_layer_3 = GCNConv(self.mid_size, self.output_size)
        self.GCN_layer_1 = GraphConvolution(self.bert_config.hidden_size*2,self.mid_size)
        self.GCN_layer_2 = GraphConvolution(self.mid_size, self.mid_size)
        self.GCN_layer_3 = GraphConvolution(self.mid_size, self.output_size)

        self.classifier = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        if self.config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

        self.init_layer()
        self.GCN_layer_1.reset_parameters()
        self.GCN_layer_2.reset_parameters()
        self.GCN_layer_3.reset_parameters()

        if self.num_labels == 1:
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()



    def init_layer(self):
        # 模型层数的初始化初始化
        nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
        nn.init.constant_(self.entity_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.sent_fc_layer.linear.weight)
        nn.init.constant_(self.sent_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier.linear.weight)
        nn.init.constant_(self.classifier.linear.bias, 0.)

    
    def make_graph(self,batch_ent):
        # 默认思考gpu卡的个数为2
        graph = [[],[]]
        for j in range(len(batch_ent)):
            e1 = batch_ent[j]['e1']
            e2 = batch_ent[j]['e2']
            #这里是否添加取决于使用的是哪种GCN
            for k in range(j+1,len(batch_ent)):
                if e1 == batch_ent[k]['e1'] or e2 == batch_ent[k]['e2']:
                    graph[0].append(j)
                    graph[0].append(k)
                    graph[1].append(k)
                    graph[1].append(j)

        return graph
    def make_gcn_graph(self,batch_ent):
        graph = [[0 for i in range(len(batch_ent))] for j in range(len(batch_ent))]
        for j in range(len(batch_ent)):
            e1 = batch_ent[j]['e1']
            e2 = batch_ent[j]['e2']
            graph[j][j] = 1
            #这里是否添加取决于使用的是哪种GCN
            for k in range(j+1,len(batch_ent)):
                if e1 == batch_ent[k]['e1'] or e2 == batch_ent[k]['e2']:
                    graph[j][k] = 1 
                    graph[k][j] = 1
        return graph

    def forward(self, input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask ,ent_pair):
        """
        但是这里将实体对放在一个[CLS]sent<SEP>中，而不是两个sent之中

        :param input_ids:
        :param token_type_ids:
        :param attention_masks:
        :param e1_mask:  这里e1_mask和e2_mask覆盖了special tag([s1][e1],[s2][e2])，所以这里需要需要切片以下
        :param e2_mask:
        :param labels:
        :return:
        """
        device_info = str(input_ids.device)
        if "cuda:0" in device_info:
            ent_pair = ent_pair[:input_ids.shape[0]]
        # 判断张量是否在 GPU 1 上
        elif "cuda:1" in device_info:
            ent_pair = ent_pair[-input_ids.shape[0]:]
        # edge_index = torch.tensor(self.make_graph(ent_pair),dtype=torch.long).to(input_ids.device)
        edge_index = torch.tensor(self.make_gcn_graph(ent_pair),dtype=torch.float,requires_grad=False).to(input_ids.device)
        outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)

        concat_h = self.get_pool_output(sequence_output, pooled_output, input_ids, e1_mask, e2_mask,attention_mask=attention_masks,front_next_outputs = edge_index,neg_edge_index=edge_index)
        logits = self.classifier(concat_h)

        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        return F.relu(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
