# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   这个使用已经训练完成的模型来预测无标签的数据
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""


import os
import datetime
import pickle
import time
from collections import Counter

from ipdb import set_trace
from tqdm import tqdm


from sklearn.metrics import precision_recall_fscore_support,accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix
from transformers import BertTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from src.dataset_utils.data_process_utils import read_raw_data,get_label2id

from src.dataset_utils.entity_type_marker import NormalDataset,NEGNormalDataset

from src.utils.function_utils import correct_datetime, set_seed, save_model, load_model_and_parallel
from src.utils.tips_utils import get_bert_config, get_logger, print_hyperparameters, wandb_log, show_log
from src.utils.train_utils import relation_classification_decode, batch_to_device, choose_model, \
    set_tokenize_special_tag
from collections import defaultdict


def predicate(config,ckpt_path=None,logger=None):

    # 读取所有的相关数据
    examples = read_raw_data(config)
    
    label2id, id2label = get_label2id(config.relation_labels)
    config.num_labels = len(label2id)

    device = torch.device('cuda') if config.use_gpu else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
    set_tokenize_special_tag(config, tokenizer)
    # 这个针对sentence-level的关系分类
    if config.data_format == 'single':
        dev_dataset = NormalDataset(examples, config=config,label2id=None, tokenizer=tokenizer,device=device)
    elif config.data_format == 'neg_single':
        dev_dataset = NEGNormalDataset(examples, config=config,label2id=None, tokenizer=tokenizer,device=device)
    # MTB的方法，这个至少可以解决一些cross-sentence 关系分类

    else:
        raise ValueError


    dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=False, collate_fn=dev_dataset.collate_fn_predicate,
                                num_workers=0, batch_size=config.batch_size)
    # 选择模型
    model = choose_model(config)

    model.bert_model.resize_token_embeddings(len(tokenizer))
    # 注意这里load_type
    # 根据训练完成的模型来选择合适的load_type

    model, device = load_model_and_parallel(model, '0', ckpt_path=ckpt_path, strict=True,
                                            load_type='one2one')
    model.to(device)
    all_predicate_tokens = []
    relation_predicate_res = []
    relation_counter = defaultdict(int)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for step, batch_data in tqdm(enumerate(dev_dataloader),desc='正在预测数据...',total=len(dev_dataloader)):
            if config.data_format == 'single':
                labels = None
                if 'gcn' in config.model_name or 'gat' in config.model_name:
                    input_ids, token_type_ids, attention_masks, e1_mask, e2_mask,ent_pair = batch_data
                    logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask,ent_pair)
                else:
                    input_ids, token_type_ids, attention_masks, e1_mask, e2_mask = batch_data
                    logits = model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)
                predicate_token = relation_classification_decode(logits)
            elif config.data_format =='neg_single':
                labels = None
                input_ids, token_type_ids, attention_masks, e1_mask,e2_mask,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask,ent_pair= batch_data
                logits = model(input_ids, token_type_ids, attention_masks, e1_mask,e2_mask, labels,e1_neg_input_ids, e1_neg_token_type_ids, e1_neg_attention_masks ,e2_neg_input_ids, e2_neg_token_type_ids, e2_neg_attention_masks,e1_neg_mask,e2_neg_mask,ent_pair)
                predicate_token = relation_classification_decode(logits)
            else:
                raise ValueError
            # set_trace()
            # 保证他们实体之间存在interaction
            if config.dataset_name == 'Case_study':
                id2label = {
                    0:'PDI',
                    1:'PPI',
                    2:'CPI',
                    3:'DVI',
                    4:'CDI',
                    5:'CVI',
                    6:'DDI'
                }
            else:
                id2label = {
                    0:'None',
                    1:'PPI',
                    2:'DDI',
                    3:'CPI',
                    4:'CDI',
                }
            
            logger.info("当前数据个数有:{}/{}".format(len(relation_predicate_res),step*config.batch_size))
            # set_trace()
            for idx in range(len(predicate_token)):
                flag = predicate_token[idx]
                try:
                    flag = predicate_token[idx]
                except:
                    print("发生意外....")
                    break

                # if flag:
                relation_counter[id2label[flag]]+=1
                if config.num_labels > 2:
                    if config.dataset_name == 'Case_study':
                            relation_predicate_res.append({
                            'id': 'r' + str(step * config.batch_size + idx),
                            'abstract_id': examples[step * config.batch_size + idx].abstract_id,
                            'text': examples[step * config.batch_size + idx].text,
                            'e1_id': examples[step * config.batch_size+idx].ent1_id,
                            'e1_name':examples[step * config.batch_size+idx].ent1_name,
                            'e2_id': examples[step* config.batch_size+idx].ent2_id,
                            'e2_name':examples[step * config.batch_size+idx].ent2_name,
                            'relation_type': id2label[flag],
                        })
                    else:
                        relation_predicate_res.append({
                            'id': 'r' + str(step * config.batch_size + idx),
                            'abstract_id': examples[step * config.batch_size + idx].abstract_id,
                            'e1_id': examples[step * config.batch_size+idx].ent1_id,
                            'e2_id': examples[step* config.batch_size+idx].ent2_id,
                            'relation_type': id2label[flag],
                        })


                elif config.num_labels == 2:

                    relation_predicate_res.append({
                        'id': 'r' + str(step * config.batch_size + idx),
                        'abstract_id': examples[step * config.batch_size + idx].abstract_id,
                        'e1_id': examples[step * config.batch_size + idx].ent1_id,
                        'e2_id': examples[step * config.batch_size + idx].ent2_id,
                        'relation_type': 1,
                    })

        print("花费时间",time.time()-start_time)
        # set_trace()
    with open("./outputs/predicate_outputs/case_study/{}_scheme{}re_results.txt".format(config.model_name,config.scheme),'wb') as f:
        pickle.dump(relation_predicate_res,f)

    logger.info('-------预测类别结果----------------')
    logger.info(relation_counter)







if __name__ == '__main__':
    config = get_bert_config()
    # 设置时间
    now = datetime.datetime.now()
    diff = datetime.timedelta(hours=8)
    now = now + diff

    set_seed(config.seed)

    # 测试wandb

    # project表示这次项目，entity:表示提交人，config为超参数

    logger = get_logger(config)

    logger.info('---------------使用模型预测本次数据---------------------')
    print_hyperparameters(config, logger)
    #use your best model
    ckpt_path='/outputs/save_models/your_train_model/best_model'

    predicate(config,ckpt_path,logger) 