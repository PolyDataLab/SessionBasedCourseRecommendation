import time
import random
import math
import pickle
import torch
import numpy as np
import pandas as pd
from config import Config
import data_helpers as dh
import dataprocess as dp
import tensorflow as tf
from dataprocess import *
from utils import *
from offered_courses import *
import utils

logger = dh.logger_fn("torch-log", "logs/test-{0}.log".format(time.asctime()))

#MODEL = input("☛ Please input the model file you want to test: ")
#MODEL = "./Others/DREAM_2/runs/1663182568/model-09-0.8750-0.3290.model"
#MODEL = "1681419484"
#MODEL = "1681749277"
#MODEL = "1683234575"
#MODEL = "1683744082"
#MODEL = "1695333140"
#MODEL = "1695402666"
#MODEL = "1695415347"
#MODEL = "1695418339" 
#MODEL = "1695647021"
#MODEL = "1695651209"
#MODEL = "1697464080"
#MODEL = "1697494782" #best recall score - 64, 3, 0.4
#MODEL = "1700236037" # 64, 1, 0.5
#MODEL = "1700236101" # 64, 2, 0.5
#MODEL = "1700236150" # 64, 1, 0.4
#MODEL = "1700236193" # 64, 2, 04
#MODEL = "1700241635" ## 64, 1, 0.3
#MODEL = "1700241683" # 64, 2, 0.3
#MODEL = "1700241725" # 64, 3, 0.3
#MODEL = "1700241774" # 64, 1, 0.6
#MODEL = "1700241824" # 64, 2, 0.6
MODEL = "1700241865" # 64, 3, 0.6

#MODEL = './Course_Beacon/runs/1674078249'

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")

MODEL_DIR = dh.load_model_file(MODEL)
#MODEL_DIR = "./Others/DREAM/runs/1663182568/model-09-0.8750-0.3290.model"

def recall_cal(positives, pred_items):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        return float(correct_preds/actual_bsize)
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))

def course_CIS_dept_filtering(course):
    list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
    flag = 0
    for term in list_of_terms:
        if course.find(term)!= -1:
            flag = 1
    return flag 
def calculate_term_dict(term_dict, semester, basket, reversed_item_dict):
    for item in basket:
        if semester not in term_dict:
            count_course = {}
        else:
            count_course = term_dict[semester]
        if reversed_item_dict[item] not in count_course:
            count_course[reversed_item_dict[item]] = 1
        else:
            count_course[reversed_item_dict[item]] = count_course[reversed_item_dict[item]]+ 1
        term_dict[semester] = count_course
    return term_dict

def calculate_term_dict_2(term_dict, semester, basket, reversed_item_dict):
    for item in basket:
        if semester not in term_dict:
            count_course = {}
        else:
            count_course = term_dict[semester]
        if item not in count_course:
            count_course[item] = 1
        else:
            count_course[item] = count_course[item]+ 1
        term_dict[semester] = count_course
    return term_dict

def calculate_term_dict_true(term_dict_true, semester, t_basket, pred_basket, reversed_item_dict):
    for item in pred_basket:
        if item in t_basket:
            if semester not in term_dict_true:
                count_course = {}
            else:
                count_course = term_dict_true[semester]
            if reversed_item_dict[item] not in count_course:
                count_course[reversed_item_dict[item]] = 1
            else:
                count_course[reversed_item_dict[item]] = count_course[reversed_item_dict[item]]+ 1
            term_dict_true[semester] = count_course
    return term_dict_true

def calculate_term_dict_false(term_dict_false, semester, t_basket, pred_basket, reversed_item_dict):
    for item in pred_basket:
        if item not in t_basket:
            if semester not in term_dict_false:
                count_course = {}
            else:
                count_course = term_dict_false[semester]
            if reversed_item_dict[item] not in count_course:
                count_course[reversed_item_dict[item]] = 1
            else:
                count_course[reversed_item_dict[item]] = count_course[reversed_item_dict[item]]+ 1
            term_dict_false[semester] = count_course
    return term_dict_false
# def course_allocation(data, reversed_item_dict):
#     term_dict = {}
#     count_course = {}
#     for x in range(len(data)):
#         semester = data['last_semester'][x]
#         #if semester not in term_dict:
#         term_dict, count_course = calculate_term_dict(term_dict, semester, count_course, data['baskets'][x], reversed_item_dict)
#         # else:
#         #     term_dict[semester], count_course = calculate_term_dict(term_dict, semester, count_course, data['baskets'][x], reversed_item_dict)
#     return term_dict
def calculate_avg_n_actual_courses(input_data, reversed_item_dict):
    data = input_data
    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if item not in frequency_of_courses:
                    frequency_of_courses[item] = 1
                else:
                    frequency_of_courses[item] += 1
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict_2(term_dict_all, semester, basket, reversed_item_dict)
    count_course_all = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in count_course_all:
                count_course_all[item] = [cnt, 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = count_course_all[item]
                cnt1 += cnt
                n1 += 1
                #count_course_all[item] = list1
                count_course_all[item] = [cnt1, n1]
    count_course_avg = {}
    for course, n in count_course_all.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        count_course_avg[course] = float(cnt2/n2)
    #calculate standard deviation
    course_sd = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in course_sd:
                course_sd[item] = [pow((cnt-count_course_avg[item]),2), 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = course_sd[item]
                cnt1 = cnt1+ pow((cnt-count_course_avg[item]),2)
                n1 += 1
                #count_course_all[item] = list1
                course_sd[item] = [cnt1, n1]
    course_sd_main = {}
    course_number_terms = {}
    for course, n in course_sd.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        if(n2==1): course_sd_main[course] = float(math.sqrt(cnt2/n2))
        else: course_sd_main[course] = float(math.sqrt(cnt2/(n2-1)))
        course_number_terms[course] = n2
    
    return term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms

def course_CIS_dept_filtering(course):
    list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
    flag = 0
    for term in list_of_terms:
        if course.find(term)!= -1:
            flag = 1
    return flag 

def find_prior_term(course, prior_semester, term_dict_all_prior):
    flag = 0
    count_course_prior_2 = {}
    while(flag!=1):
        #print("prior_semester: ", prior_semester)
        if prior_semester in term_dict_all_prior:
            count_course_prior_2 = term_dict_all_prior[prior_semester]
        if course in count_course_prior_2:
            flag =1
        if prior_semester %5==0:
            prior_semester = prior_semester-4
        else:
            prior_semester = prior_semester-3
    return count_course_prior_2 

def calculate_std_dev(error_list):
    sum_err = 0.0
    for err in error_list:
        sum_err += err
    avg_err = sum_err/ len(error_list)
    sum_diff = 0.0
    for err in error_list:
        sum_diff += pow((err-avg_err), 2)
    std_dev = math.sqrt((sum_diff/len(error_list)))
    return avg_err, std_dev

def find_course_avg_prior_terms(prior_semester, term_dict_all_prior):
    count_course_all = {}
    for keys, values in term_dict_all_prior.items():
        count_course = values
        if keys<= prior_semester:
            for item, cnt in count_course.items():
                if item not in count_course_all:
                    count_course_all[item] = [cnt, 1]
                else:
                    # list1 = count_course_all[item]
                    # list1[0] = list1[0]+ cnt
                    # list1[1] = list1[0]+ 1
                    cnt1, n1 = count_course_all[item]
                    cnt1 += cnt
                    n1 += 1
                    #count_course_all[item] = list1
                    count_course_all[item] = [cnt1, n1]
    count_course_avg = {}
    for course2, n in count_course_all.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        count_course_avg[course2] = float(cnt2/n2)
    return count_course_avg

def find_course_avg_last_4_prior_terms(course, prior_semester, term_dict_all_prior):
    term_dict_all_prior = dict(sorted(term_dict_all_prior.items(), key=lambda item: item[0], reverse= True))
    count_course_all = {}
    cnt_terms = 0
    for keys, values in term_dict_all_prior.items():
        count_course = values
        if keys<= prior_semester:
            for item, cnt in count_course.items():
                if item==course:
                    if item not in count_course_all:
                        count_course_all[item] = [cnt, 1]
                    else:
                        # list1 = count_course_all[item]
                        # list1[0] = list1[0]+ cnt
                        # list1[1] = list1[0]+ 1
                        cnt1, n1 = count_course_all[item]
                        cnt1 += cnt
                        n1 += 1
                        #count_course_all[item] = list1
                        count_course_all[item] = [cnt1, n1]
                    cnt_terms += 1
        if cnt_terms==4: break
    count_course_avg = {}
    for course, n in count_course_all.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        #print("number of terms: ",n2)
        count_course_avg[course] = float(cnt2/n2)
    return count_course_avg

def valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, frequency_of_courses_train, count_course_avg_train, output_path):
    f = open(output_path, "w") #generating text file with recommendation using filtering function
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    #test_data = dh.load_data(Config().TRAININGSET_DIR)
    valid_data = dh.load_data('./Others/DREAM/valid_sample_without_target.json')

    logger.info("✔︎ Test data processing...")
    #test_target = dh.load_data(Config().TESTSET_DIR)
    valid_target = dh.load_data('./Others/DREAM/validation_target_set.json')

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    dr_model = torch.load(MODEL_DIR)

    dr_model.eval()

    item_embedding = dr_model.encode.weight
    hidden = dr_model.init_hidden(Config().batch_size)

    hitratio_numer = 0
    hitratio_denom = 0
    #ndcg = 0.0
    recall = 0.0
    recall_2= 0.0
    #recall_3= 0.0
    count=0
    recall_bsize = {}
    missed_bsize = {}
    retake_bsize = {}
    non_CIS_bsize = {}
    CIS_missed_bsize = {}
    #test_recall = 0.0
    #last_batch_actual_size = len(valid_data) % Config().batch_size
    for i, x in enumerate(dh.batch_iter(valid_data, Config().batch_size, Config().seq_len, shuffle=False)):
        uids, baskets, lens, prev_idx = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            scores = []
            du_latest = du[l - 1].unsqueeze(0)
            user_baskets = valid_data[valid_data['userID'] == uid].baskets.values[0]
            prior_bsize = len(user_baskets)
            #print("user_baskets: ", user_baskets)
            item_list1= []
            # calculating <u,p> score for all test items <u,p> pair
            positives = valid_target[valid_target['userID'] == uid].baskets.values[0]  # list dim 1
            target_semester = valid_target[valid_target['userID'] == uid].last_semester.values[0]
            #print("uid: ", uid, " ",positives)
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
            #negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
            negtives = list(neg_samples[uid])
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
            recall_temp =0.0
            #print(offered_courses[l+1])
            if t_idx==1: # we are not considering randomly selected instances for last batch
                k=0
                pred_items= []
                count1= 0
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
                f.write("UserID: ")
                # f.write(str(reversed_user_dict[reversed_user_dict2[uid]])+ "| ")
                f.write(str(reversed_user_dict2[uid])+ "| ")
                #f.write(str(uid)+ "| ")
                f.write("target basket: ")
                target_basket2 = []
                for item2 in positives:
                    f.write(str(reversed_item_dict[item2])+ " ")
                    target_basket2.append(reversed_item_dict[item2])
                
                rec_basket2 = []
                f.write(", Recommended basket: ")
                for item3 in pred_items:
                    f.write(str(reversed_item_dict[item3])+ " ")
                    rec_basket2.append(reversed_item_dict[item3])
                f.write("\n") 
                    
                prior_courses = []
                for basket3 in user_baskets:
                    for item4 in basket3:
                        if reversed_item_dict[item4] not in prior_courses:
                            prior_courses.append(reversed_item_dict[item4])
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer += len((set(positives) & set(pred_items)))
                hitratio_denom += p_length
                #print(index_k)

                #calculate recall
                #recall_2+= recall_cal(positives, index_k)
                recall_temp = recall_cal(positives, pred_items)
                recall_2+= recall_temp
                if prior_bsize not in recall_bsize:
                    recall_bsize[prior_bsize]= [recall_temp]
                else:
                    recall_bsize[prior_bsize] += [recall_temp]
                # number of non-CIS and retake courses out of missed courses    
                n_missed = 0
                n_retake = 0
                n_non_CIS =0
                n_CIS = 0
                n_non_CIS_all = 0
                unique_courses = []
                freq =0
                for course2 in target_basket2:
                    if course2 not in rec_basket2:
                        n_missed += 1
                        if course_CIS_dept_filtering(course2)==0:
                            n_non_CIS +=1
                        else:
                            n_CIS +=1
                        if course2 in prior_courses:
                            n_retake += 1
                    if course_CIS_dept_filtering(course2)==0:
                        n_non_CIS_all += 1
                        if course2 not in unique_courses:
                            unique_courses.append(course2)
                        # freq += count_course_avg_train[course2]
                        freq += count_course_avg_train[course2]
                if prior_bsize not in non_CIS_bsize:
                    non_CIS_bsize[prior_bsize]= [n_non_CIS_all, unique_courses, 1, freq]
                else:
                    n3, uq, cnt3, fq = non_CIS_bsize[prior_bsize]
                    for c4 in unique_courses:
                        if c4 not in uq:
                            uq.append(c4)
                    n3 += n_non_CIS_all
                    cnt3+= 1
                    fq += freq
                    non_CIS_bsize[prior_bsize] = [n3, uq, cnt3, fq]
                if n_missed>0:
                    if prior_bsize not in missed_bsize:
                        missed_bsize[prior_bsize]= [n_non_CIS, n_missed]
                    else:
                        x3, y3 = missed_bsize[prior_bsize]
                        x3+= n_non_CIS
                        y3 += n_missed
                        missed_bsize[prior_bsize] = [x3, y3]
                    
                    if prior_bsize not in retake_bsize:
                        retake_bsize[prior_bsize]= [n_retake, n_missed]
                    else:
                        x4, y4 = retake_bsize[prior_bsize]
                        x4+= n_retake
                        y4 += n_missed
                        retake_bsize[prior_bsize] = [x4, y4]
                    if prior_bsize not in CIS_missed_bsize:
                        CIS_missed_bsize[prior_bsize]= [n_CIS, n_missed]
                    else:
                        x5, y5 = CIS_missed_bsize[prior_bsize]
                        x5+= n_CIS
                        y5 += n_missed
                        CIS_missed_bsize[prior_bsize] = [x5, y5]
                # if n_missed>0:
                #     v3= n_non_CIS/n_missed
                #     if prior_bsize not in missed_bsize:
                #         missed_bsize[prior_bsize]= [v3]
                #     else:
                #         # x3, y3 = missed_bsize[prior_bsize]
                #         # x3+= n_non_CIS
                #         # y3 += n_missed
                #         missed_bsize[prior_bsize] += [v3]
                #     v4= n_retake/n_missed
                #     if prior_bsize not in retake_bsize:
                #         retake_bsize[prior_bsize]= [v4]
                #     else:
                #         # x4, y4 = retake_bsize[prior_bsize]
                #         # x4+= n_retake
                #         # y4 += n_missed
                #         retake_bsize[prior_bsize] += [v4]
                count=count+1

    hitratio = hitratio_numer / hitratio_denom
    #ndcg = ndcg / len(test_data)
    recall = recall_2/ count
    print(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write("\n")
    #print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    print('Recall[@n]: {0}'.format(recall))
    f.write(str('Recall[@n]: {0}'.format(recall)))
    f.write("\n")
     # recall scores for different basket sizes
    recall_bsize = dict(sorted(recall_bsize.items(), key=lambda item: item[0], reverse= False))
    for k, v in recall_bsize.items():
        bsize = k
        sum = 0
        for r in v:
            sum += r
        recall = sum/len(v)
        print("prior basket size: ", bsize)
        print("number of instances: ", len(v))
        print("recall score for validation data: ", recall)
     # number of non_CIS courses out of missed courses for different number of prior semesters
    missed_bsize = dict(sorted(missed_bsize.items(), key=lambda item: item[0], reverse= False))
    for k, v in missed_bsize.items():
        bsize = k
        tot_non_CIS, tot_missed = v
        # per_of_non_CIS = v[0]/ v[1]
        per_of_non_CIS = (tot_non_CIS/ tot_missed) *100
        print("prior basket size: ", bsize)
        # print("number of instances: ", len(v))
        print(" percentage of non CIS courses out of missed courses for validation data: ", per_of_non_CIS)
    # number of retake courses out of missed courses for different number of prior semesters
    retake_bsize = dict(sorted(retake_bsize.items(), key=lambda item: item[0], reverse= False))
    for k, v in retake_bsize.items():
        bsize = k
        tot_retake, tot_missed = v
        per_of_retaken_courses = (tot_retake/ tot_missed) *100
        print("prior basket size: ", bsize)
        # print("number of instances: ", len(v))
        print("percentage of retaken courses out of missed courses for validation data: ", per_of_retaken_courses)

    non_CIS_bsize = dict(sorted(non_CIS_bsize.items(), key=lambda item: item[0], reverse= False))
    for k, v in non_CIS_bsize.items():
        bsize = k
        # sum5 = 0
        # for r in v:
        #     sum5 += r
        sum5, un_c, ct, freq1 = v
        avg_non_CIS = sum5/ct
        avg_pop = freq1/sum5
        print("prior basket size: ", bsize)
        print("number of instances: ", ct)
        print("total non_CIS_courses: ", sum5)
        print("average non_CIS courses for validation data: ",avg_non_CIS)
        print("total unique non_CIS courses for validation data: ",len(un_c))
        print("average popularity of non_CIS courses for validation data: ", avg_pop)
        if bsize ==17: print(un_c)
    
     # number of CIS courses missed out of missed courses for different number of prior semesters
    CIS_missed_bsize = dict(sorted(CIS_missed_bsize.items(), key=lambda item: item[0], reverse= False))
    for k, v in CIS_missed_bsize.items():
        bsize = k
        tot_CIS, tot_missed = v
        # per_of_non_CIS = v[0]/ v[1]
        per_of_CIS_missed = (tot_CIS/ tot_missed) *100
        print("prior basket size: ", bsize)
        # print("number of instances: ", len(v))
        print(" percentage of CIS courses out of missed courses: ", per_of_CIS_missed)
    # number of non_CIS courses out of missed courses for different number of prior semesters
    # missed_bsize = dict(sorted(missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in missed_bsize.items():
    #     bsize = k
    #     #tot_non_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     sum3 = 0
    #     for r in v:
    #         sum3 += r
    #     per_of_non_CIS = (sum3/ len(v)) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of non CIS courses out of missed courses for test data: ", per_of_non_CIS)
    # # number of retake courses out of missed courses for different number of prior semesters
    # retake_bsize = dict(sorted(retake_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in retake_bsize.items():
    #     bsize = k
    #     # tot_retake, tot_missed = v
    #     sum4 = 0
    #     for r in v:
    #         sum4 += r
    #     per_of_retaken_courses = (sum4/ len(v)) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print("percentage of retaken courses out of missed courses for test data: ", per_of_retaken_courses)

    f.close()

def remove_summer_term_from_valid(input_data):
    data = input_data
    users = data.userID.values
    test_all = []
    index =0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['last_semester']!= 1205:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester']]
            test_all.append(row)
        index +=1
    valid_set_without_summer = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester'])
    return valid_set_without_summer

if __name__ == '__main__':
    train_data = pd.read_json('./Filtered_data/train_sample_augmented_CR.json', orient='records', lines= True)
    # train_all, train_set_without_target, target, item_dict, user_dict, reversed_item_dict, reversed_user_dict, max_len = preprocess_train_data(train_data)
    train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data) 
    train_data_unique = pd.read_json('./train_data_all_CR.json', orient='records', lines= True)
    # train_all, train_set_without_target, target, item_dict, user_dict, reversed_item_dict, reversed_user_dict, max_len = preprocess_train_data(train_data) 
    print("number of items:", len(item_dict))
    term_dict_train, frequency_of_courses_train, count_course_avg_train, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(train_data_unique, reversed_item_dict)
    valid_data = pd.read_json('./valid_data_all_CR.json', orient='records', lines= True)
    #valid_data_excluding_summer_term = remove_summer_term_from_valid(valid_data)
    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    #valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data_excluding_summer_term, reversed_user_dict, item_dict)
    valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data)
    # valid_all = pd.read_json('./Others/DREAM/valid_sample_all.json', orient='records', lines=True)
    # valid_set_without_target= pd.read_json('./Others/DREAM/valid_sample_without_target.json', orient='records', lines=True)
    # valid_target = pd.read_json('./Others/DREAM/validation_target_set.json', orient='records', lines=True)
    #offered_courses = calculate_offered_courses(valid_all)
    offered_courses = offered_course_cal('./all_data_CR.json')
    data_dir= './Others/DREAM/'
    output_dir = data_dir + "/output_dir"
    create_folder(output_dir)
    output_path= output_dir+ "/valid_prediction_2.txt"
    valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, frequency_of_courses_train, count_course_avg_train, output_path)
    print("config of validation: 64, 3, 0.6")
