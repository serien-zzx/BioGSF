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
from src.models.base_layers import FCLayer,Simple1DCNN,MultiHeadAttention
from src.models.bert_model import EntityMarkerBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F




class NEGSingleEntityMarkersREModel(EntityMarkerBaseModel):
    def __init__(self, config: BertConfig):
        super(NEGSingleEntityMarkersREModel, self).__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        self.scheme = config.scheme
        self.sent_num = 2
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.attention_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size

        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim, self.config.dropout_prob)
        if self.scheme == -113:
            self.entity_fc_layer = FCLayer(self.bert_config.hidden_size*3, self.entity_dim, self.config.dropout_prob)
        else:
            self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)
        self.CNN_layer = Simple1DCNN(self.bert_config.hidden_size,self.bert_config.hidden_size,self.config.dropout_prob,self.config.neg_len)
        self.attention_layer = MultiHeadAttention(self.bert_config.hidden_size,8)
        # self.head_num=8
        # self.neg_attention_layer = MultiHeadAttention1(self.attention_dim,self.head_num)
        # self.neg_attention_layer_1 = TransformerLayer(self.attention_dim,self.head_num,self.attention_dim,self.config.dropout_prob)


        # self.max_pooling = nn.MaxPool1d(kernel_size=config.max_len)
        self.classifier = FCLayer(
            self.classifier_dim,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )

        self.neg_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)

        self.ent_neg_fc_layer = nn.Linear(self.bert_config.hidden_size*2, self.entity_dim, self.config.dropout_prob)

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

        nn.init.xavier_normal_(self.classifier.linear.weight)
        nn.init.constant_(self.classifier.linear.bias, 0.)

        nn.init.xavier_normal_(self.neg_fc_layer.linear.weight)
        nn.init.constant_(self.neg_fc_layer.linear.bias,0.)

        nn.init.xavier_normal_(self.ent_neg_fc_layer.weight)
        nn.init.constant_(self.ent_neg_fc_layer.bias,0.)





    def forward(self, input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask):
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

        outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )


        sequence_output = outputs[0]  # shape=(batch_size,seq_len,hidden_size)
        pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)

        e1_neg_outputs = self.bert_model(
            e1_neg_input_ids, attention_mask=e1_neg_attention_masks, token_type_ids=e1_neg_token_type_ids
        )

        e2_neg_outputs = self.bert_model(
            e2_neg_input_ids, attention_mask=e2_neg_attention_masks, token_type_ids=e2_neg_token_type_ids
        )
        # e1_neg_sequence_output = self.neg_fc_layer(e1_neg_outputs[0])
        # e2_neg_sequence_output = self.neg_fc_layer(e2_neg_outputs[0])
        e1_neg_sequence_output = e1_neg_outputs[0]
        e2_neg_sequence_output = e2_neg_outputs[0]


        concat_h = self.get_pool_output(sequence_output, pooled_output, input_ids, e1_mask, e2_mask,e1_front_outputs=e1_neg_sequence_output,e2_front_outputs=e2_neg_sequence_output,e1_next_outputs=e1_neg_mask,e2_next_outputs=e2_neg_mask)

        logits = self.classifier(concat_h)


        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss, logits

        return logits

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # Q, K, 和 V 矩阵的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.neg_k = nn.Linear(d_model, d_model)

        # 合并多头注意力输出的权重矩阵
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, mask=None):
        batch_size = query.shape[0]

        # 使用线性变换获得Q、K、V
        Q = self.W_q(query)
        K = self.W_k(query)
        V = self.W_v(query)

        # 分割成多个头
        Q = self.split_heads(Q, batch_size)#[16,8,512,96]
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        neg_kk = self.neg_k(key)

        neg = self.split_heads(neg_kk, batch_size)#[16,8,1,96]
        # 缩放点积注意力
        scaled_attention_logits = torch.matmul(Q, K.permute(0, 1, 3, 2)) /  torch.matmul(K,(K+neg).permute(0, 1, 3, 2))#[16, 8, 512, 96]

        # 掩码处理
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, V)

        # 合并多头注意力输出
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output
    

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        # 第一个线性变换和激活函数
        x = self.linear1(x)
        x = self.relu(x)

        # 第二个线性变换
        x = self.linear2(x)

        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout_prob):
        super(TransformerLayer, self).__init__()

        # 多头自注意力层
        self.multihead_self_attention = MultiHeadSelfAttention(d_model, num_heads)
        
        # 第一个 "Add & Norm" 层
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed Forward 层
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        
        # 第二个 "Add & Norm" 层
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(self, x, y,mask=None):
        # 多头自注意力
        attention_output = self.multihead_self_attention(x, y, mask)
        
        # 第一个 "Add & Norm" 层
        x = self.norm1(x + attention_output)
        x = self.dropout_layer(x)
        # Feed Forward 层
        feed_forward_output = self.feed_forward(x)
        
        # 第二个 "Add & Norm" 层
        x = self.norm2(x + feed_forward_output)
        x = self.dropout_layer(x)
        return x


    
class MultiHeadAttention1(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention1, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = FCLayer(d_model, d_model)
        # self.kk = nn.Linear(d_model,d_model)
        self.init_layer()

    def init_layer(self):
        # 模型层数的初始化初始化
        nn.init.xavier_normal_(self.wq.weight)
        nn.init.constant_(self.wq.bias,0.)

        nn.init.xavier_normal_(self.wk.weight)
        nn.init.constant_(self.wk.bias,0.)

        nn.init.xavier_normal_(self.wv.weight)
        nn.init.constant_(self.wv.bias,0.)


        nn.init.xavier_normal_(self.dense.linear.weight)
        nn.init.constant_(self.dense.linear.bias,0.) 
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key):
        batch_size = query.size(0)
        
        q = self.split_heads(self.wq(query), batch_size)
        k = self.split_heads(self.wk(key), batch_size)
        v = self.split_heads(self.wv(key), batch_size)#(batch,head,seq_len=1,hidden/head)
        
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.depth ** 0.5)
        
        attn_scores = torch.softmax(qk, dim=-1)
        weighted_sum = torch.matmul(attn_scores, v)


        output = weighted_sum.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output).squeeze(dim=1)
        
        return output