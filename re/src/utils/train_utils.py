# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description : 这是模型训练过程中需要的各种trick
                例如 学习率调整器...
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08:
-------------------------------------------------
"""

import numpy as np
from ipdb import set_trace
from torch.utils.data import DataLoader,WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
import torch


from src.dataset_utils.entity_type_marker import NormalDataset,NEGNormalDataset
from src.models.entitymarker_model import SingleEntityMarkersREModel
from src.models.neg_entitymarker_model import NEGSingleEntityMarkersREModel
from src.models.gcn_entitymarker_model import gcn_SingleEntityMarkersREModel
from src.models.gcn_neg_entitymarker_model import gcn_NEGSingleEntityMarkersREModel
from src.models.gat_entitymarker_model import gat_SingleEntityMarkersREModel
from src.models.gat_neg_entitymarker_model import gat_NEGSingleEntityMarkersREModel

def batch_to_device(batch_data,device):
    for i in range(len(batch_data)):
        batch_data[i] = batch_data[i].to(device)
    return batch_data

def relation_classification_decode(logits):
    '''
    这里是解码，将关系分类预测的结果进行解码
    :param logits: shape=(batch_size,num_labels )
    :return:
    '''
    output = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return output


def build_optimizer_and_scheduler(config,model,t_toal,l1_lambda=0):
    '''
    使用warmup学习器,这个是用于基于BERT模型的学习器和优化器
    :param config:
    :param model:
    :param t_total:
    :return:
    '''
    # 这里是存储bert的参数
    bert_param_optimizer = []
    # 这里存储其他网络层的参数
    other_param_optimizer = []

    # 差分学习率
    no_decay = ['bias', 'LayerNorm.weight']
    model_pram = list(model.named_parameters())
    l1_lambda =l1_lambda

    for name, param in model_pram:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        # bert module
        {
            "params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            # 这个表示不包含在no_decay的param进行weight decay
            "weight_decay": config.weight_decay,
            'lr': config.bert_lr
        },

        {
            "params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            # 这个表示只要是包括no_decay的就不进行weight_decay
            "weight_decay": 0.0,
            'lr': config.bert_lr
        },
        # 除了bert的其他模块，差分学习率
        {
            "params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            'lr': config.other_lr
        },
        {
            "params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            'lr': config.other_lr
        },
    ]
    if config.model_name == 'bioelectra_gpnn_entitymarker_model':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.bert_lr, eps=config.adam_epsilon,weight_decay = l1_lambda)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_lr, eps=config.adam_epsilon,weight_decay = l1_lambda)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(config.warmup_proportion * t_toal), num_training_steps=t_toal
    )

    return optimizer,scheduler

def set_tokenize_special_tag(config,tokenizer):

    tokenizer.add_tokens(config.ent1_start_tag)
    tokenizer.add_tokens(config.ent1_end_tag)
    tokenizer.add_tokens(config.ent2_start_tag)
    tokenizer.add_tokens(config.ent2_end_tag)
    tokenizer.add_tokens(config.dep_tag)
    #文章中出现的UNK字符
    tokenizer.add_tokens('↓')
    tokenizer.add_tokens('∼')
    tokenizer.add_tokens('⩽')
    tokenizer.add_tokens('∶')
    tokenizer.add_tokens('"')

    config.ent1_start_tag_id = tokenizer.convert_tokens_to_ids(config.ent1_start_tag)
    config.ent1_end_tag_id = tokenizer.convert_tokens_to_ids(config.ent1_end_tag)
    config.ent2_start_tag_id = tokenizer.convert_tokens_to_ids(config.ent2_start_tag)
    config.ent2_end_tag_id = tokenizer.convert_tokens_to_ids(config.ent2_end_tag)
    config.dep_tag_id = tokenizer.convert_tokens_to_ids(config.dep_tag)

    
def build_optimizer(config,model):
    '''
        创建optimizer
        这里采用差分学习率的方法，对不同层采用不同的学习率
    '''
    # 这里是存储bert的参数
    bert_param_optimizer = []
    # 这里存储其他网络层的参数
    other_param_optimizer = []

    # 差分学习率
    no_decay = ['bias', 'LayerNorm.weight']
    model_pram = list(model.named_parameters())

    for name, param in model_pram:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        # bert module
        {
            "params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            # 这个表示不包含在no_decay的param进行weight decay
            "weight_decay": config.weight_decay,
            'lr': config.bert_lr
        },

        {
            "params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            # 这个表示只要是包括no_decay的就不进行weight_decay
            "weight_decay": 0.0,
            'lr': config.bert_lr
        },
        # 除了bert的其他模块，差分学习率
        {
            "params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            'lr': config.other_lr
        },
        {
            "params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            'lr': config.other_lr
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.bert_lr, eps=config.adam_epsilon)


    return optimizer

def choose_dataloader(config,examples,label2id,tokenizer,device,type_='train'):

    if config.data_format == 'single':  # 这个针对sentence-level的关系分类
        train_dataset = NormalDataset(examples, config=config, label2id=label2id, tokenizer=tokenizer, device=device)
    elif config.data_format == 'neg_single':
        train_dataset = NEGNormalDataset(examples, config=config, label2id=label2id, tokenizer=tokenizer, device=device)



    else:
        raise ValueError


    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=train_dataset.collate_fn,
                                num_workers=0, batch_size=config.batch_size)
    return train_dataloader
def set_sample(examples):
    class_weights = defaultdict(int)
    for feature in examples:
        class_weights[feature.label] +=1
    weights = [len(examples)/class_weights[feature.label] for feature in examples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def save_metric_writer(metric_writer, loss, global_step, p, r, f1, type='Training'):
    """
    这个主要是记录模型的performance
    """
    metric_writer.add_scalar("{}/loss".format(type), loss, global_step)
    metric_writer.add_scalar("{}/precision".format(type), p, global_step)
    metric_writer.add_scalar("{}/recall".format(type), r, global_step)
    metric_writer.add_scalar("{}/f1".format(type), f1, global_step)


def save_parameter_writer(model_writer, model, step):
    """
    这个是记录模型的参数、梯度等信息
    """
    for name, param in model.named_parameters():
        model_writer.add_histogram('model_param_' + name, param.clone().cpu().data.numpy(), step)
        if param.grad is not None:
            model_writer.add_histogram('model_grad_' + name, param.grad.clone().cpu().numpy(), step)


def choose_model(config):
    if config.model_name == 'single_entity_marker':
        model = SingleEntityMarkersREModel(config,scheme=config.scheme)
    elif config.model_name =='neg_single_entity_marker':
        model = NEGSingleEntityMarkersREModel(config)
    elif config.model_name == 'gcn_single_entity_marker':
        model = gcn_SingleEntityMarkersREModel(config)
    elif config.model_name == 'gcn_neg_single_entity_marker':
        model = gcn_NEGSingleEntityMarkersREModel(config) 
    elif config.model_name == 'gat_single_entity_marker':
        model = gat_SingleEntityMarkersREModel(config)
    elif config.model_name =='gat_neg_single_entity_marker':
        model = gat_NEGSingleEntityMarkersREModel(config)

    else:
        raise ValueError
    return model


def error_analysis(all_dev_labels, all_predicate_tokens,dev_raw_text):
    """
    这是分析关系分类错误的情况
    收集分类错误的对应句子，以及预测的类别标签()
    """
    error_idx_li = []
    for idx in range(len(all_dev_labels)):
        if all_dev_labels[idx]!=all_predicate_tokens[idx]:
            error_idx_li.append([dev_raw_text[idx],all_predicate_tokens[idx],all_dev_labels[idx]])

    set_trace()