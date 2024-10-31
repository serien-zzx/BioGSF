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
from src.models.base_layers import FCLayer,lstm_FCLayer,Simple1DCNN,MultiHeadAttention,Spacy_GAT,GAT,Highway,Simple1DCNN_1,QGAAttention
from src.models.bert_model import EntityMarkerBaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch




class gat_NEGSingleEntityMarkersREModel(EntityMarkerBaseModel):
    def __init__(self, config: BertConfig):
        super(gat_NEGSingleEntityMarkersREModel, self).__init__(config)

        self.num_labels = config.num_labels
        
        self.config = config
        self.scheme = config.scheme
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        self.double_fc_layer = FCLayer(self.bert_config.hidden_size*2, self.bert_config.hidden_size ,self.config.dropout_prob)
        self.highway = Highway(self.bert_config.hidden_size)
        # if config.scheme in  [-35,36,-36,37,-37]:
        #     self.sent_fc_layer = FCLayer(self.bert_config.hidden_size*4, self.entity_dim, self.config.dropout_prob)
        # else:    
        #     self.sent_fc_layer = FCLayer(self.bert_config.hidden_size*3, self.entity_dim, self.config.dropout_prob)
        self.sent_fc_layer = FCLayer(self.bert_config.hidden_size*4, self.entity_dim, self.config.dropout_prob)
       
        self.CNN_layer = Simple1DCNN(self.bert_config.hidden_size,self.bert_config.hidden_size,self.config.dropout_prob,self.config.neg_len)
        # self.CNN_layer_1 = Simple1DCNN_1(self.bert_config.hidden_size,self.bert_config.hidden_size,self.config.dropout_prob,self.config.neg_len)
        self.attention_layer = MultiHeadAttention(self.bert_config.hidden_size,8)
        self.mid_size = 128
        self.output_size = 64
        self.alpha = 0.01
        self.num_head = 4
        self.GAT_layer = GAT(self.bert_config.hidden_size*3,self.mid_size,self.output_size,self.config.dropout_prob,self.alpha,self.num_head)
        # self.spacy_GAT_layer = Spacy_GAT(self.bert_config.hidden_size,self.bert_config.hidden_size,self.bert_config.hidden_size,self.config.dropout_prob,self.alpha,self.num_head)
        # self.GAT_layer_1 = GAT(self.bert_config.hidden_size*3,self.mid_size,self.mid_size,self.config.dropout_prob,self.alpha,self.num_head)
        # self.GAT_layer_2 = GAT(self.mid_size,self.mid_size,self.output_size,self.config.dropout_prob,self.alpha,self.num_head)
        self.attention_layer_1= QGAAttention(self.bert_config.hidden_size,2,8)
        # self.bn1 = nn.BatchNorm1d(self.mid_size)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(self.output_size)
        self.graph_mid_layer = FCLayer(self.bert_config.hidden_size*3, self.output_size,self.config.dropout_prob)
        # self.GAT_layer_3 = GAT(self.mid_size,self.mid_size,self.output_size,self.config.dropout_prob,self.alpha,self.num_head)

        self.classifier = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        if self.config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

        self.init_layer()

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

        nn.init.xavier_normal_(self.double_fc_layer.linear.weight)
        nn.init.constant_(self.double_fc_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.graph_mid_layer.linear.weight)
        nn.init.constant_(self.graph_mid_layer.linear.bias, 0.)

        nn.init.xavier_normal_(self.classifier.linear.weight)
        nn.init.constant_(self.classifier.linear.bias, 0.)

    
    def make_graph(self,batch_ent):
        # 默认思考gpu卡的个数为2
        graph = [[0 for i in range(len(batch_ent))] for i in range(len(batch_ent))]

        for j in range(len(batch_ent)):
            e1 = batch_ent[j]['e1']
            e2 = batch_ent[j]['e2']
            for k in range(j+1,len(batch_ent)):
                if e1 == batch_ent[k]['e1'] or e2 == batch_ent[k]['e2']:
                    graph[j][k] = 1
                    graph[k][j] = 1

        return graph


    def forward(self, input_ids, token_type_ids, attention_masks, e1_mask, e2_mask ,labels,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask,ent_pair,ent_type=None,neg_edge_index=None):
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
        edge_index = torch.tensor(self.make_graph(ent_pair),dtype=torch.long).to(input_ids.device)
        outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)

        e1_neg_outputs = self.bert_model(
            e1_neg_input_ids, attention_mask=e1_neg_attention_masks, token_type_ids=e1_neg_token_type_ids
        )


        e1_neg_sequence_output = e1_neg_outputs[0]
        # e2_neg_sequence_output = e2_neg_outputs[0]
        concat_h = self.get_pool_output(sequence_output, pooled_output, input_ids, e1_mask, e2_mask,attention_mask=attention_masks,front_next_outputs = edge_index,e1_front_outputs=e1_neg_sequence_output,e1_next_outputs=e1_neg_mask,neg_edge_index=neg_edge_index)

        logits = self.classifier(concat_h)

        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits


