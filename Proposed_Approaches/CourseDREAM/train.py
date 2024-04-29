import os
import math
import random
import time
import logging
import pickle
from turtle import shapesize
import torch
import numpy as np
from math import ceil
#from utils import data_helpers as dh
import data_helpers as dh
from config import Config
from rnn_model import DRModel
import tensorflow as tf
from dataprocess import *
#from utils import *
import utils
#from offered_courses import *
import pandas as pd
import offered_courses
logging.info("✔︎ DREAM Model Training...")
logger = dh.logger_fn("torch-log", "logs/training-{0}.log".format(time.asctime()))

dilim = '-' * 120
logger.info(dilim)
for attr in sorted(Config().__dict__):
    logger.info('{:>50}|{:<50}'.format(attr.upper(), Config().__dict__[attr]))
logger.info(dilim)

def train(offered_courses, train_set_without_target, target_set, item_dict):
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data(Config().TRAININGSET_DIR)

    logger.info("✔︎ Validation data processing...")
    validation_data = dh.load_data(Config().VALIDATIONSET_DIR)

    logger.info("✔︎ Test data processing...")
    #target_data = dh.load_data(Config().TESTSET_DIR)
    target_data = target_set

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Model config
    model = DRModel(Config())

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config().learning_rate)

    def bpr_loss(uids, baskets, dynamic_user, item_embedding):
        """
        Bayesian personalized ranking loss for implicit feedback.

        Args:
            uids: batch of users' ID
            baskets: batch of users' baskets
            dynamic_user: batch of users' dynamic representations
            item_embedding: item_embedding matrix
        """
        loss = 0
        for uid, bks, du in zip(uids, baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item]
            loss_u = []  # loss for user
            for t, basket_t in enumerate(bks):
                if basket_t[0] != 0 and t != 0:
                    pos_idx = torch.LongTensor(basket_t)
                    #pos_idx = torch.LongTensor(int_labels2)

                    # Sample negative products
                    neg = random.sample(list(neg_samples[uid]), len(basket_t))
                    #neg = random.sample(list(neg_samples[uid]), len(int_labels2))

                    neg_idx = torch.LongTensor(neg)

                    # Score p(u, t, v > v')
                    score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]

                    # Average Negative log likelihood for basket_t
                    loss_u.append(torch.mean(-torch.nn.LogSigmoid()(score)))
            for i in loss_u:
                loss = loss + i / len(loss_u)
        avg_loss = torch.div(loss, len(baskets))
        return avg_loss

    def train_model():
        model.train()  # turn on training mode for dropout
        dr_hidden = model.init_hidden(Config().batch_size)
        train_loss = 0
        #train_recall = 0.0
        start_time= time.perf_counter()
       #start_time = time.clock()

        num_batches = ceil(len(train_data) / Config().batch_size)
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, shuffle=True)):
            uids, baskets, lens, prev_idx = x
            model.zero_grad() 
            dynamic_user, _ = model(baskets, lens, dr_hidden)

            loss = bpr_loss(uids, baskets, dynamic_user, model.encode.weight)
            loss.backward()

            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config().clip)

            # Parameter updating
            optimizer.step()
            train_loss += loss.data

            # Logging
            #if i % Config().log_interval == 0 and i > 0:
            elapsed = (time.process_time() - start_time) / Config().log_interval
            cur_loss = train_loss.item() / Config().log_interval  # turn tensor into float
            train_loss = 0
            start_time = time.process_time()
            logger.info('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.4f} |'
                        .format(epoch, i+1, num_batches, elapsed, cur_loss))

    def validate_model():
        model.eval()
        dr_hidden = model.init_hidden(Config().batch_size)
        val_loss = 0
        start_time= time.perf_counter()
        #start_time = time.clock()

        num_batches = ceil(len(validation_data) / Config().batch_size)
        for i, x in enumerate(dh.batch_iter(validation_data, Config().batch_size, Config().seq_len, shuffle=False)):
            uids, baskets, lens, prev_idx = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            loss = bpr_loss(uids, baskets, dynamic_user, model.encode.weight)
            val_loss += loss.data

        # Logging
        elapsed = (time.perf_counter() - start_time) * 1000 / num_batches
        val_loss = val_loss.item() / num_batches
        logger.info('[Validation]| Epochs {:3d} | Elapsed {:02.2f} | Loss {:05.4f} | '
                    .format(epoch, elapsed, val_loss))
        return val_loss
    
    # calculate recall 
    def recall_cal(positives, pred_items, count_at_least_one_cor_pred):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        if(correct_preds>=1): count_at_least_one_cor_pred += 1
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))
        return float(correct_preds/actual_bsize), count_at_least_one_cor_pred
    def test_model(offered_courses):
        model.eval()
        item_embedding = model.encode.weight
        dr_hidden = model.init_hidden(Config().batch_size)

        hitratio_numer = 0
        hitratio_denom = 0
        #ndcg = 0.0
        recall = 0.0
        recall_2= 0.0
        recall_temp = 0.0
        count=0
        count_at_least_one_cor_pred = 0
        #print(target_data)
        for i, x in enumerate(dh.batch_iter(train_set_without_target, Config().batch_size, Config().seq_len, shuffle=False)):
            uids, baskets, lens, prev_idx = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            for uid, l, du in zip(uids, lens, dynamic_user):
                scores = []
                du_latest = du[l - 1].unsqueeze(0)
                user_baskets = train_set_without_target[train_set_without_target['userID'] == uid].baskets.values[0]
                target_semester = target_data[target_data['userID'] == uid].last_semester.values[0]
                

                item_list1= []
                # calculating <u,p> score for all test items <u,p> pair
                positives = target_data[target_data['userID'] == uid].baskets.values[0]  # list dim 1
                target_semester = target_data[target_data['userID'] == uid].last_semester.values[0]

                for x1 in positives:
                    item_list1.append(x1)
                #print(positives)

                p_length = len(positives)
                positives2 = torch.LongTensor(positives)
                #print(positives)
                # Deal with positives samples
                scores_pos = list(torch.mm(du_latest, item_embedding[positives2].t()).data.numpy()[0])
                for s in scores_pos:
                    scores.append(s)

                # Deal with negative samples
                negtives = list(neg_samples[uid]) #taking all the available items 
                #negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
                for x2 in negtives:
                    item_list1.append(x2)
                negtives2 = torch.LongTensor(negtives)
                scores_neg = list(torch.mm(du_latest, item_embedding[negtives2].t()).data.numpy()[0])
                for s in scores_neg:
                    scores.append(s)
                #print(item_list1)
                #print(scores)
                # Calculate hit-ratio
                index_k = []
                #top_k1= Config().top_k
                top_k1 = len(positives)
                #print(offered_courses[l+1])
                k=0
                pred_items= []
                count1= 0
                #print(offered_courses)
                while(k<top_k1):
                    index = scores.index(max(scores))
                    item1 = item_list1[index]
                    if not utils.filtering(item1, user_baskets, offered_courses[target_semester], item_dict):
                        #if index not in index_k:
                        if item1 not in pred_items:
                            #index_k.append(index)
                            pred_items.append(item1)
                            k+=1
                    scores[index] = -9999
                    count1+= 1
                    if(count1==len(scores)): break
                #print(index_k)
                #print(pred_items)
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer += len((set(positives) & set(pred_items)))
                hitratio_denom += p_length
                #print(index_k)

                # Calculate NDCG
                # u_dcg = 0
                # u_idcg = 0
                # for k1 in range(Config().top_k):
                #     if index_k[k1] < p_length:  
                #         u_dcg += 1 / math.log(k1 + 1 + 1, 2)
                #     u_idcg += 1 / math.log(k1 + 1 + 1, 2)
                # ndcg += u_dcg / u_idcg
                #calculate recall
                #recall_2+= recall_cal(positives, index_k)
                recall_temp, count_at_least_one_cor_pred= recall_cal(positives, pred_items, count_at_least_one_cor_pred)
                recall_2+= recall_temp
                count=count+1
                
        hit_ratio = hitratio_numer / hitratio_denom
        #ndcg = ndcg / len(train_data)
        recall = recall_2/ count
        logger.info('[Test]| Epochs {:3d} | Hit ratio {:02.4f} | recall {:05.4f} |'
                    .format(epoch, hit_ratio, recall))
        print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
        percentage_of_at_least_one_cor_pred = count_at_least_one_cor_pred/ len(target_data)
        print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)

        return hit_ratio, recall

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{hitratio:.4f}-{recall:.4f}.model'

    best_hit_ratio = None

    try:
        # Training
        for epoch in range(Config().epochs):
            train_model()
            logger.info('-' * 89)

            val_loss = validate_model()
            logger.info('-' * 89)

            hit_ratio, recall = test_model(offered_courses)
            logger.info('-' * 89)

            # Checkpoint
            if not best_hit_ratio or hit_ratio > best_hit_ratio:
                with open(checkpoint_dir.format(epoch=epoch, hitratio=hit_ratio, recall=recall), 'wb') as f:
                    torch.save(model, f)
                best_hit_ratio = hit_ratio
    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')
    print("model directory: ", timestamp)
    print("config for train: 64, 2, 0.6")


if __name__ == '__main__':
    #train()
    # train_data = pd.read_json('./train_data_all.json', orient='records', lines= True)
    # train_all, train_set_without_target, train_target,  item_dict, user_dict, reversed_item_dict, reversed_user_dict, max_len = preprocess_train_data(train_data)
    start = time.time()
    train_data = pd.read_json('./Filtered_data/train_sample_augmented_CR.json', orient='records', lines= True)
    train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data) 
    train_all = pd.read_json('./Others/DREAM/train_sample_all.json', orient='records', lines=True)
    train_set_without_target = pd.read_json('./Others/DREAM/train_set_without_target.json', orient='records', lines=True)
    target_set = pd.read_json('./Others/DREAM/target_set.json', orient='records', lines=True)

    #offered_courses = calculate_offered_courses(train_all)
    offered_courses = offered_courses.offered_course_cal('./all_data_CR.json')
    #train(offered_courses, train_set_without_target, reversed_item_dict, reversed_user_dict)
    train(offered_courses, train_set_without_target, target_set, item_dict)
    end = time.time()
    total_training_time = end - start
    #print("Total training time:", total_training_time)
    
