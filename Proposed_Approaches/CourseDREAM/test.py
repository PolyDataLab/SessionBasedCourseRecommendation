import time
import random
import math
import pickle
import torch
import pandas as pd
import numpy as np
from config import Config
import data_helpers as dh
#import dataprocess as dp
import tensorflow as tf
from dataprocess import *
from utils import *
import utils
import dataprocess
import offered_courses
from offered_courses import *

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
MODEL = "1697494782" #best recall score - 64, 3, 0.4
#MODEL = "1700236037" # 64, 1, 0.5
#MODEL = "1700236101" # 64, 2, 0.5
#MODEL = "1700236150" # 64, 1, 0.4
#MODEL = "1700236193" # 64, 2, 04
#MODEL = "1700241635" ## 64, 1, 0.3
#MODEL = "1700241683" # 64, 2, 0.3
#MODEL = "1700241725" # 64, 3, 0.3
#MODEL = "1700241774" # 64, 1, 0.6
#MODEL = "1700241824" # 64, 2, 0.6
#MODEL = "1700241865" # 64, 3, 0.6
#MODEL = './Course_Beacon/runs/1674078249'

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")
MODEL_DIR = dh.load_model_file(MODEL)
#MODEL_DIR = "./Others/DREAM/runs/1663182568/model-09-0.8750-0.3290.model"

def recall_cal(positives, pred_items, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        if(correct_preds>=1): count_at_least_one_cor_pred += 1
        if correct_preds>=2: count_at_least_two_cor_pred+= 1
        if correct_preds>=3: count_at_least_three_cor_pred+= 1
        if correct_preds>=4: count_at_least_four_cor_pred+= 1
        if correct_preds>=5: count_at_least_five_cor_pred+= 1
        if correct_preds==actual_bsize: count_all_cor_pred+= 1

        if (actual_bsize>=6): 
            if(correct_preds==1): count_cor_pred[6,1]+= 1
            if(correct_preds==2): count_cor_pred[6,2]+= 1
            if(correct_preds==3): count_cor_pred[6,3]+= 1
            if(correct_preds==4): count_cor_pred[6,4]+= 1
            if(correct_preds==5): count_cor_pred[6,5]+= 1
            if(correct_preds>=6): count_cor_pred[6,6]+= 1
        else:
            if(correct_preds==1): count_cor_pred[actual_bsize,1]+= 1
            if(correct_preds==2): count_cor_pred[actual_bsize,2]+= 1
            if(correct_preds==3): count_cor_pred[actual_bsize,3]+= 1
            if(correct_preds==4): count_cor_pred[actual_bsize,4]+= 1
            if(correct_preds==5): count_cor_pred[actual_bsize,5]+= 1
        
        return float(correct_preds/actual_bsize), count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))

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

# calculate standard deviation
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

# a baseline approach to find average of number of enrollments in each course in last 4 semesters
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

# calculate mse, rmse, mae using the recommendations from CourseDREAM model
def calculate_mse_for_course_allocation(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg_all, course_sd_main, course_number_terms, term_dict_all_prior, output_dir):
    mse_for_course_allocation = 0.0
    mse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mse_for_course_allocation_considering_not_predicted_courses = 0.0
    mae_for_course_allocation = 0.0
    mae_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mae_for_course_allocation_considering_not_predicted_courses = 0.0
    msse_for_course_allocation = 0.0
    msse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_msse_for_course_allocation_considering_not_predicted_courses = 0.0
    mase_for_course_allocation = 0.0
    mase_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mase_for_course_allocation_considering_not_predicted_courses = 0.0
    #count1= 0
    count2 = 0
    output_path1= output_dir+ "/test_course_allocation_filtered.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    error_list = []
    ab_error_list = []
    st_error_list = []
    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        # number of students in the previous offering
        if semester %5==0:
            prior_semester = semester-4
        else:
            prior_semester = semester-3

        if semester in term_dict_predicted:
            count_course_predicted = term_dict_predicted[semester]
            count_course_predicted_true = term_dict_predicted_true[semester]
            count_course_predicted_false = term_dict_predicted_false[semester]

            count_course_avg_all = find_course_avg_prior_terms(prior_semester, term_dict_all_prior)
            for item1 in count_course.keys():
                f.write("Semester: ")
                f.write(str(semester)+ " ")
                f.write("Course ID: ")
                f.write(str(item1)+ " ")
                count_course_prior = find_prior_term(item1, prior_semester, term_dict_all_prior)
                
                count_course_avg = find_course_avg_last_4_prior_terms(item1, prior_semester, term_dict_all_prior)

                if item1 in count_course_predicted:
                    #mse_for_course_allocation += pow(((count_course[item1]/count_total_course[semester])-(count_course_predicted[item1]/count_total_course[semester])), 2)
                    mse_for_course_allocation += pow((count_course_predicted[item1]-count_course[item1]), 2)
                    mae_for_course_allocation += abs(count_course_predicted[item1]-count_course[item1])
                    msse_for_course_allocation += pow(abs((count_course_predicted[item1]-count_course[item1])/count_course[item1]), 2)
                    mase_for_course_allocation += abs((count_course_predicted[item1]-count_course[item1])/count_course[item1])
                    error_list.append(count_course_predicted[item1]-count_course[item1])
                    ab_error_list.append(abs(count_course_predicted[item1]-count_course[item1]))
                    st_error_list.append(abs((count_course_predicted[item1]-count_course[item1])/count_course[item1]))
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(count_course_predicted[item1]))
                    f.write("\n")
                    if item1 in count_course_predicted_true and item1 in count_course_predicted_false:
                        row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], count_course_predicted_false[item1], count_course_avg_all[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                        #course_allocation.append(row)
                    elif item1 in count_course_predicted_true and item1 not in count_course_predicted_false:
                        row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], 0, count_course_avg_all[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                        #course_allocation.append(row)
                    elif item1 not in count_course_predicted_true and item1 in count_course_predicted_false:
                        row = [semester, item1, count_course[item1], count_course_predicted[item1], 0, count_course_predicted_false[item1], count_course_avg_all[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                        #course_allocation.append(row)
                    #count1 += 1
                    if item1 in count_course_prior:
                        row.append(count_course_prior[item1])
                    else:
                        row.append(0)
                    course_allocation.append(row)
                    count2 += 1
                else:
                    mse_for_course_allocation_2 += pow((count_course[item1]-0), 2)
                    mae_for_course_allocation_2 += abs((count_course[item1]-0))
                    msse_for_course_allocation_2 += pow(((count_course[item1]-0)/count_course[item1]), 2)
                    mase_for_course_allocation_2 += abs((count_course[item1]-0)/count_course[item1])
                    error_list.append(0-count_course[item1])
                    ab_error_list.append(abs(0-count_course[item1]))
                    st_error_list.append(abs((0-count_course[item1])/count_course[item1]))
                    
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(0))
                    f.write("\n")
                    if item1 in count_course_prior:
                        row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg_all[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], count_course_prior[item1]]
                    else:
                        row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg_all[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], 0]
                    course_allocation.append(row)
                    count2 += 1
    #avg_mse_for_course_allocation = mse_for_course_allocation/ count1
    avg_mse_for_course_allocation_considering_not_predicted_courses = (mse_for_course_allocation+ mse_for_course_allocation_2 )/ count2
    avg_mae_for_course_allocation_considering_not_predicted_courses = (mae_for_course_allocation+ mae_for_course_allocation_2 )/ count2
    avg_msse_for_course_allocation_considering_not_predicted_courses = (msse_for_course_allocation+ msse_for_course_allocation_2 )/ count2
    avg_mase_for_course_allocation_considering_not_predicted_courses = (mase_for_course_allocation+ mase_for_course_allocation_2 )/ count2
   
    f.close()
    course_allocation_actual_predicted = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'predicted_n', 'predicted_n_true', 'predicted_n_false', 'avg_n_all_prior', 'avg_n_last_4_prior', 'st_dev_actual', 'number_of_terms', 'n_sts_last_offering'])
    course_allocation_actual_predicted.to_csv(output_dir+'/course_allocation_actual_predicted_updated_new_v7_filtered.csv')
    return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list

    #return avg_mse_for_course_allocation, avg_mse_for_course_allocation_considering_not_predicted_courses

# calculate mse, rmse, mae for a baseline approach to find average of number of enrollments in each course in all prior semesters
def calculate_mse_for_course_allocation_avg(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg_all, course_sd_main, course_number_terms, term_dict_all_prior, output_dir):
    mse_for_course_allocation = 0.0
    mse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mse_for_course_allocation_considering_not_predicted_courses = 0.0
    mae_for_course_allocation = 0.0
    mae_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mae_for_course_allocation_considering_not_predicted_courses = 0.0
    msse_for_course_allocation = 0.0
    msse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_msse_for_course_allocation_considering_not_predicted_courses = 0.0
    mase_for_course_allocation = 0.0
    mase_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mase_for_course_allocation_considering_not_predicted_courses = 0.0
    #count1= 0
    count2 = 0
    output_path1= output_dir+ "/test_course_allocation_v2.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    error_list = []
    ab_error_list = []
    st_error_list = []

    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        # number of students in the previous offering
        if semester%5==0:
            prior_semester = semester-4
        else:
            prior_semester = semester-3

        if semester in term_dict_predicted:
            count_course_predicted = term_dict_predicted[semester]
            count_course_predicted_true = term_dict_predicted_true[semester]
            count_course_predicted_false = term_dict_predicted_false[semester]

            # if prior_semester in term_dict:
            #     count_course_prior = term_dict[prior_semester]
            # elif prior_semester in term_dict_all_prior:
            #     count_course_prior = term_dict_all_prior[prior_semester]
            count_course_avg = find_course_avg_prior_terms(prior_semester, term_dict_all_prior)
            for item1 in count_course.keys():
                f.write("Semester: ")
                f.write(str(semester)+ " ")
                f.write("Course ID: ")
                f.write(str(item1)+ " ")
                count_course_prior = find_prior_term(item1, prior_semester, term_dict_all_prior)

                if item1 in count_course_avg:
                    #mse_for_course_allocation += pow(((count_course[item1]/count_total_course[semester])-(count_course_predicted[item1]/count_total_course[semester])), 2)
                    mse_for_course_allocation += pow((count_course_avg[item1]-count_course[item1]), 2)
                    mae_for_course_allocation += abs(count_course_avg[item1]-count_course[item1])
                    msse_for_course_allocation += pow(((count_course_avg[item1]-count_course[item1])/count_course[item1]), 2)
                    mase_for_course_allocation += abs((count_course_avg[item1]-count_course[item1])/count_course[item1])
                    error_list.append(count_course_avg[item1]-count_course[item1])
                    ab_error_list.append(abs(count_course_avg[item1]-count_course[item1]))
                    st_error_list.append(abs((count_course_avg[item1]-count_course[item1])/count_course[item1]))
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(count_course_avg[item1]))
                    f.write("\n")
                    # if item1 in count_course_predicted_true and item1 in count_course_predicted_false:
                    #     row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                    #     #course_allocation.append(row)
                    # elif item1 in count_course_predicted_true and item1 not in count_course_predicted_false:
                    #     row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                    #     #course_allocation.append(row)
                    # elif item1 not in count_course_predicted_true and item1 in count_course_predicted_false:
                    #     row = [semester, item1, count_course[item1], count_course_predicted[item1], 0, count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                    #     #course_allocation.append(row)
                    # #count1 += 1
                    # if item1 in count_course_prior:
                    #     row.append(count_course_prior[item1])
                    # else:
                    #     row.append(0)
                    # course_allocation.append(row)
                    count2 += 1
                else:
                    mse_for_course_allocation_2 += pow((count_course[item1]-0), 2)
                    mae_for_course_allocation_2 += abs((count_course[item1]-0))
                    msse_for_course_allocation_2 += pow(((count_course[item1]-0)/count_course[item1]), 2)
                    mase_for_course_allocation_2 += abs((count_course[item1]-0)/count_course[item1])
                    error_list.append(0-count_course[item1])
                    ab_error_list.append(abs(0-count_course[item1]))
                    st_error_list.append(abs((0-count_course[item1])/count_course[item1]))
                    
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(0))
                    f.write("\n")
                    # if item1 in count_course_prior:
                    #     row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], count_course_prior[item1]]
                    # else:
                    #     row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], 0]
                    # course_allocation.append(row)
                    count2 += 1
    #avg_mse_for_course_allocation = mse_for_course_allocation/ count1
    avg_mse_for_course_allocation_considering_not_predicted_courses = (mse_for_course_allocation+ mse_for_course_allocation_2 )/ count2
    avg_mae_for_course_allocation_considering_not_predicted_courses = (mae_for_course_allocation+ mae_for_course_allocation_2 )/ count2
    avg_msse_for_course_allocation_considering_not_predicted_courses = (msse_for_course_allocation+ msse_for_course_allocation_2 )/ count2
    avg_mase_for_course_allocation_considering_not_predicted_courses = (mase_for_course_allocation+ mase_for_course_allocation_2 )/ count2
   
    f.close()
    #course_allocation_actual_predicted = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'predicted_n', 'predicted_n_true', 'predicted_n_false', 'avg_n_actual', 'st_dev_actual', 'number_of_terms', 'n_sts_last_offering'])
    #course_allocation_actual_predicted.to_csv(output_dir+'/course_allocation_actual_predicted_avg.csv')
    return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list

    #return avg_mse_for_course_allocation, avg_mse_for_course_allocation_considering_not_predicted_courses
 #calculate mse, rmse, mae for a baseline approach to find average of number of enrollments in each course in 4 prior semesters
def calculate_mse_for_course_allocation_avg_last_4(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg_all, course_sd_main, course_number_terms, term_dict_all_prior, output_dir):
    mse_for_course_allocation = 0.0
    mse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mse_for_course_allocation_considering_not_predicted_courses = 0.0
    mae_for_course_allocation = 0.0
    mae_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mae_for_course_allocation_considering_not_predicted_courses = 0.0
    msse_for_course_allocation = 0.0
    msse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_msse_for_course_allocation_considering_not_predicted_courses = 0.0
    mase_for_course_allocation = 0.0
    mase_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mase_for_course_allocation_considering_not_predicted_courses = 0.0
    #count1= 0
    count2 = 0
    output_path1= output_dir+ "/test_course_allocation_v2.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    error_list = []
    ab_error_list = []
    st_error_list = []

    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        # number of students in the previous offering
        if semester%5==0:
            prior_semester = semester-4
        else:
            prior_semester = semester-3

        if semester in term_dict_predicted:
            count_course_predicted = term_dict_predicted[semester]
            count_course_predicted_true = term_dict_predicted_true[semester]
            count_course_predicted_false = term_dict_predicted_false[semester]

            # if prior_semester in term_dict:
            #     count_course_prior = term_dict[prior_semester]
            # elif prior_semester in term_dict_all_prior:
            #     count_course_prior = term_dict_all_prior[prior_semester]

            for item1 in count_course.keys():
                f.write("Semester: ")
                f.write(str(semester)+ " ")
                f.write("Course ID: ")
                f.write(str(item1)+ " ")
                count_course_prior = find_prior_term(item1, prior_semester, term_dict_all_prior)
                count_course_avg = find_course_avg_last_4_prior_terms(item1, prior_semester, term_dict_all_prior)

                if item1 in count_course_avg:
                    #mse_for_course_allocation += pow(((count_course[item1]/count_total_course[semester])-(count_course_predicted[item1]/count_total_course[semester])), 2)
                    mse_for_course_allocation += pow((count_course_avg[item1]-count_course[item1]), 2)
                    mae_for_course_allocation += abs(count_course_avg[item1]-count_course[item1])
                    msse_for_course_allocation += pow(((count_course_avg[item1]-count_course[item1])/count_course[item1]), 2)
                    mase_for_course_allocation += abs((count_course_avg[item1]-count_course[item1])/count_course[item1])
                    error_list.append(count_course_avg[item1]-count_course[item1])
                    ab_error_list.append(abs(count_course_avg[item1]-count_course[item1]))
                    st_error_list.append(abs((count_course_avg[item1]-count_course[item1])/count_course[item1]))
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(count_course_avg[item1]))
                    f.write("\n")
                    # if item1 in count_course_predicted_true and item1 in count_course_predicted_false:
                    #     row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                    #     #course_allocation.append(row)
                    # elif item1 in count_course_predicted_true and item1 not in count_course_predicted_false:
                    #     row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                    #     #course_allocation.append(row)
                    # elif item1 not in count_course_predicted_true and item1 in count_course_predicted_false:
                    #     row = [semester, item1, count_course[item1], count_course_predicted[item1], 0, count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                    #     #course_allocation.append(row)
                    # #count1 += 1
                    # if item1 in count_course_prior:
                    #     row.append(count_course_prior[item1])
                    # else:
                    #     row.append(0)
                    # course_allocation.append(row)
                    count2 += 1
                else:
                    mse_for_course_allocation_2 += pow((count_course[item1]-0), 2)
                    mae_for_course_allocation_2 += abs((count_course[item1]-0))
                    msse_for_course_allocation_2 += pow(((count_course[item1]-0)/count_course[item1]), 2)
                    mase_for_course_allocation_2 += abs((count_course[item1]-0)/count_course[item1])
                    error_list.append(0-count_course[item1])
                    ab_error_list.append(abs(0-count_course[item1]))
                    st_error_list.append(abs((0-count_course[item1])/count_course[item1]))
                    
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    f.write("predicted: ")
                    f.write(str(0))
                    f.write("\n")
                    # if item1 in count_course_prior:
                    #     row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], count_course_prior[item1]]
                    # else:
                    #     row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], 0]
                    # course_allocation.append(row)
                    count2 += 1
    #avg_mse_for_course_allocation = mse_for_course_allocation/ count1
    avg_mse_for_course_allocation_considering_not_predicted_courses = (mse_for_course_allocation+ mse_for_course_allocation_2 )/ count2
    avg_mae_for_course_allocation_considering_not_predicted_courses = (mae_for_course_allocation+ mae_for_course_allocation_2 )/ count2
    avg_msse_for_course_allocation_considering_not_predicted_courses = (msse_for_course_allocation+ msse_for_course_allocation_2 )/ count2
    avg_mase_for_course_allocation_considering_not_predicted_courses = (mase_for_course_allocation+ mase_for_course_allocation_2 )/ count2
   
    f.close()
    #course_allocation_actual_predicted = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'predicted_n', 'predicted_n_true', 'predicted_n_false', 'avg_n_actual', 'st_dev_actual', 'number_of_terms', 'n_sts_last_offering'])
    #course_allocation_actual_predicted.to_csv(output_dir+'/course_allocation_actual_predicted_avg.csv')
    return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list

#calculate mse, rmse, mae for a baseline approach to find number of enrollments in each course in last semester that it was offered
def calculate_mse_for_course_allocation_last_offering(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg, course_sd_main, course_number_terms, term_dict_all_prior, output_dir):
    mse_for_course_allocation = 0.0
    mse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mse_for_course_allocation_considering_not_predicted_courses = 0.0
    mae_for_course_allocation = 0.0
    mae_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mae_for_course_allocation_considering_not_predicted_courses = 0.0
    msse_for_course_allocation = 0.0
    msse_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_msse_for_course_allocation_considering_not_predicted_courses = 0.0
    mase_for_course_allocation = 0.0
    mase_for_course_allocation_2 = 0.0
    # avg_recall_for_course_allocation = 0.0
    # avg_recall_for_course_allocation_considering_not_predicted_courses = 0.0
    #avg_mse_for_course_allocation = 0.0
    avg_mase_for_course_allocation_considering_not_predicted_courses = 0.0
    #count1= 0
    count2 = 0
    output_path1= output_dir+ "/test_course_allocation_last_v2.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    error_list = []
    ab_error_list = []
    st_error_list = []

    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        # number of students in the previous offering
        if semester %5==0:
            prior_semester = semester-4
        else:
            prior_semester = semester-3

        if semester in term_dict_predicted:
            count_course_predicted = term_dict_predicted[semester]
            count_course_predicted_true = term_dict_predicted_true[semester]
            count_course_predicted_false = term_dict_predicted_false[semester]

            # if prior_semester in term_dict:
            #     count_course_prior = term_dict[prior_semester]
            # elif prior_semester in term_dict_all_prior:
            #     count_course_prior = term_dict_all_prior[prior_semester]

            for item1 in count_course.keys():
                f.write("Semester: ")
                f.write(str(semester)+ " ")
                f.write("Course ID: ")
                f.write(str(item1)+ " ")
                count_course_prior = find_prior_term(item1, prior_semester, term_dict_all_prior)

                if item1 in count_course_prior:
                    #mse_for_course_allocation += pow(((count_course[item1]/count_total_course[semester])-(count_course_predicted[item1]/count_total_course[semester])), 2)
                    mse_for_course_allocation += pow((count_course_prior[item1]-count_course[item1]), 2)
                    mae_for_course_allocation += abs(count_course_prior[item1]-count_course[item1])
                    msse_for_course_allocation += pow(((count_course_prior[item1]-count_course[item1])/count_course[item1]), 2)
                    mase_for_course_allocation += abs((count_course_prior[item1]-count_course[item1])/count_course[item1])
                    error_list.append(count_course_prior[item1]-count_course[item1])
                    ab_error_list.append(abs(count_course_prior[item1]-count_course[item1]))
                    st_error_list.append(abs((count_course_prior[item1]-count_course[item1])/count_course[item1]))
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    if item1 in count_course_predicted:
                        f.write("predicted: ")
                        f.write(str(count_course_predicted[item1]))
                        f.write("\n")
                        if item1 in count_course_predicted_true and item1 in count_course_predicted_false:
                            row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                            #course_allocation.append(row)
                        elif item1 in count_course_predicted_true and item1 not in count_course_predicted_false:
                            row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                            #course_allocation.append(row)
                        elif item1 not in count_course_predicted_true and item1 in count_course_predicted_false:
                            row = [semester, item1, count_course[item1], count_course_predicted[item1], 0, count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                            #course_allocation.append(row)
                        #count1 += 1
                        if item1 in count_course_prior:
                            row.append(count_course_prior[item1])
                        else:
                            row.append(0)
                        course_allocation.append(row)
                        
                    else:
                        f.write("predicted: ")
                        f.write(str(0))
                        f.write("\n")
                        if item1 in count_course_prior:
                            row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], count_course_prior[item1]]
                        else:
                            row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], 0]
                        course_allocation.append(row)
                    
                    count2 += 1
                else:
                    #count_course_prior_prev = find_prior_term(item1, prior_semester, term_dict, term_dict_all_prior, term_dict_valid)
                    mse_for_course_allocation_2 += pow((0-count_course[item1]), 2)
                    mae_for_course_allocation_2 += abs(0-count_course[item1])
                    msse_for_course_allocation_2 += pow(((0-count_course[item1])/count_course[item1]), 2)
                    mase_for_course_allocation_2 += abs((0-count_course[item1])/count_course[item1])
                    error_list.append(0-count_course[item1])
                    ab_error_list.append(abs(0-count_course[item1]))
                    st_error_list.append(abs((0-count_course[item1])/count_course[item1]))
                    f.write("actual: ")
                    f.write(str(count_course[item1])+ " ")
                    if item1 in count_course_predicted:
                        f.write("predicted: ")
                        f.write(str(count_course_predicted[item1]))
                        f.write("\n")
                        if item1 in count_course_predicted_true and item1 in count_course_predicted_false:
                            row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                            #course_allocation.append(row)
                        elif item1 in count_course_predicted_true and item1 not in count_course_predicted_false:
                            row = [semester, item1, count_course[item1], count_course_predicted[item1], count_course_predicted_true[item1], 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                            #course_allocation.append(row)
                        elif item1 not in count_course_predicted_true and item1 in count_course_predicted_false:
                            row = [semester, item1, count_course[item1], count_course_predicted[item1], 0, count_course_predicted_false[item1], count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
                            #course_allocation.append(row)
                        #count1 += 1
                            row.append(0)
                        course_allocation.append(row)
                        
                    else:
                        f.write("predicted: ")
                        f.write(str(0))
                        f.write("\n")
                        row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], 0]
                        course_allocation.append(row)
                    count2 += 1
    #avg_mse_for_course_allocation = mse_for_course_allocation/ count1
    avg_mse_for_course_allocation_considering_not_predicted_courses = (mse_for_course_allocation+ mse_for_course_allocation_2 )/ count2
    avg_mae_for_course_allocation_considering_not_predicted_courses = (mae_for_course_allocation+ mae_for_course_allocation_2 )/ count2
    avg_msse_for_course_allocation_considering_not_predicted_courses = (msse_for_course_allocation+ msse_for_course_allocation_2 )/ count2
    avg_mase_for_course_allocation_considering_not_predicted_courses = (mase_for_course_allocation+ mase_for_course_allocation_2 )/ count2
   
    f.close()
    course_allocation_actual_predicted = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'predicted_n', 'predicted_n_true', 'predicted_n_false', 'avg_n_actual', 'st_dev_actual', 'number_of_terms', 'n_sts_last_offering'])
    course_allocation_actual_predicted.to_csv(output_dir+'/course_allocation_actual_predicted_last_offering_updated_new_v2.csv')
    return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list

    #return avg_mse_for_course_allocation, avg_mse_for_course_allocation_considering_not_predicted_courses

def remove_summer_term_from_test(input_data):
    data = input_data
    users = data.userID.values
    test_all = []
    index =0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['last_semester']!= 1215:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester']]
            test_all.append(row)
        index +=1
    test_set_without_summer = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester'])
    return test_set_without_summer

# testing with CDREAM model
def test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, output_path):
    f = open(output_path, "w") #generating text file with recommendation using filtering function
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    #test_data = dh.load_data(Config().TRAININGSET_DIR)
    # test_data = dh.load_data('./Others/DREAM/test_sample_without_target.json')
    test_data = dh.load_data('./Others/DREAM/test_sample_without_target.json')

    logger.info("✔︎ Test data processing...")
    #test_target = dh.load_data(Config().TESTSET_DIR)
    test_target = dh.load_data('./Others/DREAM/test_target_set.json')

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
    count_at_least_one_cor_pred = 0
    total_correct_preds = 0
    recall_test_for_one_cor_pred = 0.0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
    recall_temp =0.0
    target_basket_size = {}
    target_basket_size[1] = 0
    target_basket_size[2] = 0
    target_basket_size[3] = 0
    target_basket_size[4] = 0
    target_basket_size[5] = 0
    target_basket_size[6] = 0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    
    term_dict = {}
    #count_course = {}
    term_dict_predicted = {}
    term_dict_predicted_true = {}
    term_dict_predicted_false = {}
    recall_bsize = {}
    missed_bsize = {} #non-CIS courses
    retake_bsize = {}
    non_CIS_bsize = {}
    CIS_missed_bsize = {}
    rec_info = []
    #count_course_predicted = {}

    #count_one_cor_pred, count_two_cor_pred, count_three_cor_pred, count_four_cor_pred, count_five_cor_pred, count_six_or_more_cor_pred  = 0, 0, 0, 0, 0, 0
    

    #test_recall = 0.0
    for i, x in enumerate(dh.batch_iter(test_data, Config().batch_size, Config().seq_len, shuffle=False)):
    # for i, x in enumerate(dh.batch_iter(test_data, len(test_data), Config().seq_len, shuffle=False)):
        uids, baskets, lens, prev_idx = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        #count_iter = 0
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            #dealing with last batch
            # count_iter+= 1
            # if i==39:
            #     if count_iter==12: break
            scores = []
            du_latest = du[l - 1].unsqueeze(0)
            user_baskets = test_data[test_data['userID'] == uid].baskets.values[0]
            prior_bsize = len(user_baskets)
            #print("user_baskets: ", user_baskets)
            item_list1= []
            # calculating <u,p> score for all test items <u,p> pair
            positives = test_target[test_target['userID'] == uid].baskets.values[0]  # list dim 1
            target_semester = test_target[test_target['userID'] == uid].last_semester.values[0]

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
            # index_k = []
            #top_k1= Config().top_k
            top_k1 = len(positives)
            #print(offered_courses[l+1])
            if t_idx==1: # we are not cosnidering randomly selected instances for last batch
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
                #f.write("UserID: ")
                #f.write(str(reversed_user_dict[reversed_user_dict3[uid]])+ "| ")
                #f.write(str(reversed_user_dict3[uid])+ "| ")
                #f.write("target basket: ")
                target_basket2 = []
                for item2 in positives:
                    f.write(str(reversed_item_dict[item2])+ " ")
                    target_basket2.append(reversed_item_dict[item2])

                f.write(", Recommended basket: ")
                rec_basket2 = []
                for item3 in pred_items:
                    f.write(str(reversed_item_dict[item3])+ " ")
                    rec_basket2.append(reversed_item_dict[item3])
                prior_courses = []
                for basket3 in user_baskets:
                    for item4 in basket3:
                        if reversed_item_dict[item4] not in prior_courses:
                            prior_courses.append(reversed_item_dict[item4])


                f.write("\n") 
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer += len((set(positives) & set(pred_items)))
                hitratio_denom += p_length
                #print(index_k)
                pred_courses = []
                for item3 in pred_items:
                    pred_courses.append(reversed_item_dict[item3])

                #calculate recall
                #recall_2+= recall_cal(positives, index_k)
                recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal(positives, pred_items, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)  
                if top_k1>=2: count_actual_bsize_at_least_2 += 1
                if top_k1>=3: count_actual_bsize_at_least_3 += 1
                if top_k1>=4: count_actual_bsize_at_least_4 += 1
                if top_k1>=5: count_actual_bsize_at_least_5 += 1
                if top_k1>=6: count_actual_bsize_at_least_6 += 1
                # target_basket2 = []
                # for item2 in positives:
                #     target_basket2.append(reversed_item_dict[item2])
                # pred_courses = []
                # for item3 in pred_items:
                #     pred_courses.append(reversed_item_dict[item3])
                rel_rec = len((set(positives) & set(pred_items)))
                row = [top_k1, target_basket2, pred_courses, rel_rec, recall_temp, target_semester]
                rec_info.append(row)
                # test_rec_info = pd.DataFrame(rec_info, columns=['bsize', 'target_courses', 'rec_courses', 'n_rel_rec', 'recall_score', 'target_semester'])
                # test_rec_info.to_json('./Others/DREAM/test_rec_info.json', orient='records', lines=True)
                # test_rec_info.to_csv('./Others/DREAM/test_rec_info.csv')
                if recall_temp>0:  
                    recall_test_for_one_cor_pred += recall_temp
                correct_preds2= len((set(positives) & set(pred_items)))
                total_correct_preds += correct_preds2
                if prior_bsize not in recall_bsize:
                    recall_bsize[prior_bsize]= [recall_temp]
                else:
                    recall_bsize[prior_bsize] += [recall_temp]
                # # number of non-CIS and retake courses out of missed courses    
                # n_missed = 0
                # n_retake = 0
                # n_non_CIS =0
                # n_CIS =0
                # n_non_CIS_all = 0
                # unique_courses = []
                # freq = 0
                # for course2 in target_basket2:
                #     if course2 not in rec_basket2:
                #         n_missed += 1
                #         if course_CIS_dept_filtering(course2)==0:
                #             n_non_CIS +=1
                #         else:
                #             n_CIS +=1
                #         if course2 in prior_courses:
                #             n_retake += 1
                #     if course_CIS_dept_filtering(course2)==0:
                #         n_non_CIS_all += 1
                #         if course2 not in unique_courses:
                #             unique_courses.append(course2)
                #         # freq += count_course_avg_train[course2]
                #         freq += frequency_of_courses_train[course2]
                # if prior_bsize not in non_CIS_bsize:
                #     non_CIS_bsize[prior_bsize]= [n_non_CIS_all, unique_courses, 1, freq]
                # else:
                #     n3, uq, cnt3, fq = non_CIS_bsize[prior_bsize]
                #     for c4 in unique_courses:
                #         if c4 not in uq:
                #             uq.append(c4)
                #     n3 += n_non_CIS_all
                #     cnt3+= 1
                #     fq += freq
                #     non_CIS_bsize[prior_bsize] = [n3, uq, cnt3, fq]

                # if n_missed>0:
                #     if prior_bsize not in missed_bsize:
                #         missed_bsize[prior_bsize]= [n_non_CIS, n_missed]
                #     else:
                #         x3, y3 = missed_bsize[prior_bsize]
                #         x3+= n_non_CIS
                #         y3 += n_missed
                #         missed_bsize[prior_bsize] = [x3, y3]
                    
                #     if prior_bsize not in retake_bsize:
                #         retake_bsize[prior_bsize]= [n_retake, n_missed]
                #     else:
                #         x4, y4 = retake_bsize[prior_bsize]
                #         x4+= n_retake
                #         y4 += n_missed
                #         retake_bsize[prior_bsize] = [x4, y4]

                #     if prior_bsize not in CIS_missed_bsize:
                #         CIS_missed_bsize[prior_bsize]= [n_CIS, n_missed]
                #     else:
                #         x5, y5 = CIS_missed_bsize[prior_bsize]
                #         x5+= n_CIS
                #         y5 += n_missed
                #         CIS_missed_bsize[prior_bsize] = [x5, y5]
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

                if top_k1>=6: target_basket_size[6] += 1 
                else: target_basket_size[top_k1] += 1 
                recall_2+= recall_temp
                #course allocation for courses in target basket
                term_dict = calculate_term_dict(term_dict, target_semester, positives, reversed_item_dict)

                #course allocation for predicted courses
                term_dict_predicted = calculate_term_dict(term_dict_predicted, target_semester, pred_items, reversed_item_dict)
                term_dict_predicted_true = calculate_term_dict_true(term_dict_predicted_true, target_semester, positives, pred_items, reversed_item_dict)
                term_dict_predicted_false = calculate_term_dict_false(term_dict_predicted_false, target_semester, positives, pred_items, reversed_item_dict)
                count=count+1
            
    hitratio = hitratio_numer / hitratio_denom
    #ndcg = ndcg / len(test_data)
    print("total count: ", count)
    recall = recall_2/ count
    # print('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio))
    # f.write(str('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio)))
    print(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write("\n")
    #print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    print('Recall[@n]: {0}'.format(recall))
    f.write(str('Recall[@n]: {0}'.format(recall)))
    f.write("\n")
    print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
    f.write("count_at_least_one_cor_pred "+ str(count_at_least_one_cor_pred)+ "\n")
    percentage_of_at_least_one_cor_pred = (count_at_least_one_cor_pred/ len(test_target)) *100
    print("percentage_of_at_least_one_cor_pred: " + str(percentage_of_at_least_one_cor_pred)+"\n")
    f.write("percentage_of_at_least_one_cor_pred: " + str(percentage_of_at_least_one_cor_pred)+"\n")

    percentage_of_at_least_two_cor_pred = (count_at_least_two_cor_pred/ count_actual_bsize_at_least_2) *100
    print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
    f.write("percentage_of_at_least_two_cor_pred: "+ str(percentage_of_at_least_two_cor_pred)+ "\n")
    percentage_of_at_least_three_cor_pred = (count_at_least_three_cor_pred/ count_actual_bsize_at_least_3) *100
    print("percentage_of_at_least_three_cor_pred: ", percentage_of_at_least_three_cor_pred)
    f.write("percentage_of_at_least_three_cor_pred: "+ str(percentage_of_at_least_three_cor_pred)+ "\n")
    percentage_of_at_least_four_cor_pred = (count_at_least_four_cor_pred/ count_actual_bsize_at_least_4) * 100
    print("percentage_of_at_least_four_cor_pred: ", percentage_of_at_least_four_cor_pred)
    f.write("percentage_of_at_least_four_cor_pred: "+ str(percentage_of_at_least_four_cor_pred)+ "\n")
    percentage_of_at_least_five_cor_pred = (count_at_least_five_cor_pred/ count_actual_bsize_at_least_5) *100
    print("percentage_of_at_least_five_cor_pred: ", percentage_of_at_least_five_cor_pred)
    f.write("percentage_of_at_least_five_cor_pred: "+ str(percentage_of_at_least_five_cor_pred)+ "\n")
    percentage_of_all_cor_pred = (count_all_cor_pred/ len(test_target)) *100
    print("percentage_of_all_cor_pred: ", percentage_of_all_cor_pred)
    f.write("percentage_of_all_cor_pred: "+ str(percentage_of_all_cor_pred)+ "\n")
    #calculate Recall@n for whom we generated at least one correct prediction in test data
    test_recall_for_one_cor_pred = recall_test_for_one_cor_pred/ count_at_least_one_cor_pred
    print("Recall@n for whom we generated at least one correct prediction in test data: ", test_recall_for_one_cor_pred)
    f.write("Recall@n for whom we generated at least one correct prediction in test data:"+ str(test_recall_for_one_cor_pred))
    for x6 in range(1,7):
        percentage_of_one_cor_pred = (count_cor_pred[x6,1]/ target_basket_size[x6]) *100
        print("percentage of_one cor pred for target basket size {}: {}".format(x6, percentage_of_one_cor_pred))
        percentage_of_two_cor_pred = (count_cor_pred[x6,2]/ target_basket_size[x6]) *100
        print("percentage of_two cor pred for target basket size {}: {}".format(x6, percentage_of_two_cor_pred))
        percentage_of_three_cor_pred = (count_cor_pred[x6,3]/ target_basket_size[x6]) *100
        print("percentage of_three cor pred for target basket size {}: {}".format(x6, percentage_of_three_cor_pred))
        percentage_of_four_cor_pred = (count_cor_pred[x6,4]/ target_basket_size[x6]) *100
        print("percentage of_four cor pred for target basket size {}: {}".format(x6, percentage_of_four_cor_pred))
        percentage_of_five_cor_pred = (count_cor_pred[x6,5]/ target_basket_size[x6]) *100
        print("percentage of_five cor pred for target basket size {}: {}".format(x6, percentage_of_five_cor_pred))
        percentage_of_at_least_six_cor_pred = (count_cor_pred[x6,6]/ target_basket_size[x6]) *100
        print("percentage of_at_least_six cor pred for target basket size {}: {}".format(x6, percentage_of_at_least_six_cor_pred))

    for x7 in range(1,7):
        print("total count of target basket size of {}: {}".format(x7, target_basket_size[x7]))

    for x6 in range(1,7):
        print("one cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,1]))
        print("two cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,2]))
        print("three cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,3]))
        print("four cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,4]))
        print("five cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,5]))
        print("six or more cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,6]))
    
    print("total correct predictions: ", total_correct_preds)
    avg_cor_rec_per_student = (total_correct_preds/ count_at_least_one_cor_pred)
    print("average number of courses per student correctly recommended: ", avg_cor_rec_per_student)

    test_rec_info = pd.DataFrame(rec_info, columns=['bsize', 'target_courses', 'rec_courses', 'n_rel_rec', 'recall_score', 'target_semester'])

    test_rec_info.to_json('./Others/DREAM/test_rec_info.json', orient='records', lines=True)
    test_rec_info.to_csv('./Others/DREAM/test_rec_info.csv')

    # # recall scores for different basket sizes
    # recall_bsize = dict(sorted(recall_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in recall_bsize.items():
    #     bsize = k
    #     sum = 0
    #     for r in v:
    #         sum += r
    #     recall = sum/len(v)
    #     print("prior basket size: ", bsize)
    #     print("number of instances: ", len(v))
    #     print("recall score for test data: ", recall)
    
    # # number of non_CIS courses out of missed courses for different number of prior semesters
    # missed_bsize = dict(sorted(missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in missed_bsize.items():
    #     bsize = k
    #     tot_non_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     per_of_non_CIS = (tot_non_CIS/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of non CIS courses out of missed courses for test data: ", per_of_non_CIS)
    # # number of retake courses out of missed courses for different number of prior semesters
    # retake_bsize = dict(sorted(retake_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in retake_bsize.items():
    #     bsize = k
    #     tot_retake, tot_missed = v
    #     per_of_retaken_courses = (tot_retake/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print("percentage of retaken courses out of missed courses for test data: ", per_of_retaken_courses)
    
    # # calculate average nonCIS courses in test data
    # non_CIS_bsize = dict(sorted(non_CIS_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in non_CIS_bsize.items():
    #     bsize = k
    #     # sum5 = 0
    #     # for r in v:
    #     #     sum5 += r
    #     sum5, un_c, ct, freq1 = v
    #     avg_non_CIS = sum5/ct
    #     avg_pop = freq1/sum5
    #     print("prior basket size: ", bsize)
    #     print("number of instances: ", ct)
    #     print("total non_CIS_courses: ", sum5)
    #     print("average non_CIS courses for test data: ",avg_non_CIS)
    #     print("total unique non_CIS courses for test data: ",len(un_c))
    #     print("average popularity of non_CIS courses for test data: ", avg_pop)
    #     if bsize ==19: print(un_c)
    
    #  # number of CIS courses missed out of missed courses for different number of prior semesters
    # CIS_missed_bsize = dict(sorted(CIS_missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in CIS_missed_bsize.items():
    #     bsize = k
    #     tot_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     per_of_CIS_missed = (tot_CIS/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of CIS courses out of missed courses: ", per_of_CIS_missed)
    

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

        
    f.write("\n") 
    f.close()
    return term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false

if __name__ == '__main__':
    # train_data = pd.read_json('./Filtered_data/train_sample_augmented.json', orient='records', lines= True)
    # train_all, train_set_without_target, target, item_dict, user_dict, reversed_item_dict, reversed_user_dict, max_len = preprocess_train_data(train_data)
    # valid_data = pd.read_json('./valid_data_all.json', orient='records', lines= True)
    # valid_all, valid_set_without_target, valid_target, user_dict2, reversed_user_dict2 = preprocess_valid_data(valid_data, reversed_user_dict, item_dict)
    train_data = pd.read_json('./Filtered_data/train_sample_augmented_CR.json', orient='records', lines= True)
    train_data_unique = pd.read_json('./train_data_all_CR.json', orient='records', lines= True)
    # train_all, train_set_without_target, target, item_dict, user_dict, reversed_item_dict, reversed_user_dict, max_len = preprocess_train_data(train_data)
    train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = dataprocess.preprocess_train_data_part1(train_data) 
    print("number of items:", len(item_dict))
    term_dict_train, frequency_of_courses_train, count_course_avg_train, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(train_data_unique, reversed_item_dict)
    valid_data = pd.read_json('./valid_data_all_CR.json', orient='records', lines= True)
    valid_data, user_dict2, reversed_user_dict2 = dataprocess.preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    test_data = pd.read_json('./test_data_all_CR.json', orient='records', lines= True)
    #test_data_excluding_summer_term = remove_summer_term_from_test(test_data)
    #test_data, user_dict3, reversed_user_dict3 = dataprocess.preprocess_test_data_part1(test_data_excluding_summer_term, reversed_user_dict, item_dict, reversed_user_dict2)
    
    test_data, user_dict3, reversed_user_dict3 = dataprocess.preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
    test_all, test_set_without_target, test_target = dataprocess.preprocess_test_data_part2(test_data)
    # test_all = pd.read_json('./Others/DREAM/test_sample_all.json', orient='records', lines=True)
    # test_set_without_target= pd.read_json('./Others/DREAM/test_sample_without_target.json', orient='records', lines=True)
    # test_target = pd.read_json('./Others/DREAM/test_target_set.json', orient='records', lines=True)
    #term_dict = course_allocation(test_target, reversed_item_dict)
    #print(term_dict[1221])
    #offered_courses = calculate_offered_courses(test_all)
    offered_courses = offered_courses.offered_course_cal('./all_data_CR.json')
    data_dir= './Others/DREAM/'
    output_dir = data_dir + "/output_dir"
    utils.create_folder(output_dir)
    output_path= output_dir+ "/test_prediction.txt"
    term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, output_path)
    print("config of test: 64, 3, 0.4")
    count_total_course = {}
    for keys, values in term_dict.items():
        count_course_dict = values
        count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
        count3 = 0
        for cnt in count_course_dict.values():
            count3 += cnt
        count_total_course[keys] = count3
        term_dict[keys] = count_course_dict
    #sorting the courses in term dictionary based on number of occurences of courses in descending order
    for keys, values in term_dict_predicted.items():
        count_course_dict = values
        count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
        term_dict_predicted[keys] = count_course_dict
    for keys, values in term_dict_predicted_true.items():
        count_course_dict = values
        count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
        term_dict_predicted_true[keys] = count_course_dict
    
    for keys, values in term_dict_predicted_false.items():
        count_course_dict = values
        count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
        term_dict_predicted_false[keys] = count_course_dict
    
    all_data_en_pred = pd.read_json('./all_data_en_pred_filtered.json', orient='records', lines= True)
    term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(all_data_en_pred, reversed_item_dict)

    # valid_data_unique = pd.read_json('./Filtered_data/valid_sample_filtered_unique.json', orient='records', lines= True)
    # term_dict_valid, frequency_of_courses2, count_course_avg2, course_sd_main2, course_number_terms2 = calculate_avg_n_actual_courses(valid_data_unique, reversed_item_dict)

    # avg_mse_for_course_allocation, avg_mse_for_course_allocation_considering_not_predicted_courses = calculate_mse_for_course_allocation(term_dict, term_dict_predicted)
    # avg_rmse_for_course_allocation, avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation), math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list = calculate_mse_for_course_allocation(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg, course_sd_main, course_number_terms, term_dict_all, output_dir)
    avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    avg_rmsse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_msse_for_course_allocation_considering_not_predicted_courses)
    mean_error, std_dev_error = calculate_std_dev(error_list)
    mean_ab_error, std_dev_ab_error = calculate_std_dev(ab_error_list)
    mean_st_error, std_dev_st_error = calculate_std_dev(st_error_list)

    #print("avg mse for # of allocated course where we are predicting a course at least once: ",avg_mse_for_course_allocation)
    #print("avg_mse_for_course_allocation_considering all courses available in test data: ",avg_mse_for_course_allocation_considering_not_predicted_courses)
    #print("avg rmse for # of allocated course where we are predicting a course at least once: ",avg_rmse_for_course_allocation)
    print("avg_mae_for_course_allocation_considering all courses available in test data: ",avg_mae_for_course_allocation_considering_not_predicted_courses)
    print("avg_rmse_for_course_allocation_considering all courses available in test data: ",avg_rmse_for_course_allocation_considering_not_predicted_courses)
    print("avg_mase_for_course_allocation_considering all courses available in test data: ",avg_mase_for_course_allocation_considering_not_predicted_courses)
    print("avg_rmsse_for_course_allocation_considering all courses available in test data: ",avg_rmsse_for_course_allocation_considering_not_predicted_courses)
    print("mean of errors: ", mean_error)
    print("standard_deviation for errors: ", std_dev_error)
    print("mean of absolute errors: ", mean_ab_error)
    print("standard_deviation for absolute errors: ", std_dev_ab_error)
    print("mean of normalized errors: ", mean_st_error)
    print("standard_deviation for normalized errors: ", std_dev_st_error)
    # print(term_dict[1221])
    #print(term_dict_predicted[1221])
    #print(reversed_item_dict)

    # #using avg # students took the course in the all past offerings
    # avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list = calculate_mse_for_course_allocation_avg(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg, course_sd_main, course_number_terms, term_dict_all, output_dir)
    # avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    # avg_rmsse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_msse_for_course_allocation_considering_not_predicted_courses)
    # mean_error, std_dev_error = calculate_std_dev(error_list)
    # mean_ab_error, std_dev_ab_error = calculate_std_dev(ab_error_list)
    # mean_st_error, std_dev_st_error = calculate_std_dev(st_error_list)
    # #print("avg mse for # of allocated course where we are predicting a course at least once: ",avg_mse_for_course_allocation)
    # #print("avg_mse_for_course_allocation_considering all courses available in test data: ",avg_mse_for_course_allocation_considering_not_predicted_courses)
    # #print("avg rmse for # of allocated course where we are predicting a course at least once: ",avg_rmse_for_course_allocation)
    # print("avg_mae_for_course_allocation_considering all courses available in test data: ",avg_mae_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmse_for_course_allocation_considering all courses available in test data: ",avg_rmse_for_course_allocation_considering_not_predicted_courses)
    # print("avg_mase_for_course_allocation_considering all courses available in test data: ",avg_mase_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmsse_for_course_allocation_considering all courses available in test data: ",avg_rmsse_for_course_allocation_considering_not_predicted_courses)
    # print("mean of errors: ", mean_error)
    # print("standard_deviation for errors: ", std_dev_error)
    # print("mean of absolute errors: ", mean_ab_error)
    # print("standard_deviation for absolute errors: ", std_dev_ab_error)
    # print("mean of normalized errors: ", mean_st_error)
    # print("standard_deviation for normalized errors: ", std_dev_st_error)
    # # print(term_dict[1221])
    # # print(term_dict_predicted[1221])
    # # #print(reversed_item_dict)

    # #using avg number of students took the course in the last 4 semesters
    # avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list = calculate_mse_for_course_allocation_avg_last_4(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg, course_sd_main, course_number_terms, term_dict_all, output_dir)
    # avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    # avg_rmsse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_msse_for_course_allocation_considering_not_predicted_courses)
    # mean_error, std_dev_error = calculate_std_dev(error_list)
    # mean_ab_error, std_dev_ab_error = calculate_std_dev(ab_error_list)
    # mean_st_error, std_dev_st_error = calculate_std_dev(st_error_list)
    # #print("avg mse for # of allocated course where we are predicting a course at least once: ",avg_mse_for_course_allocation)
    # #print("avg_mse_for_course_allocation_considering all courses available in test data: ",avg_mse_for_course_allocation_considering_not_predicted_courses)
    # #print("avg rmse for # of allocated course where we are predicting a course at least once: ",avg_rmse_for_course_allocation)
    # print("avg_mae_for_course_allocation_considering all courses available in test data: ",avg_mae_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmse_for_course_allocation_considering all courses available in test data: ",avg_rmse_for_course_allocation_considering_not_predicted_courses)
    # print("avg_mase_for_course_allocation_considering all courses available in test data: ",avg_mase_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmsse_for_course_allocation_considering all courses available in test data: ",avg_rmsse_for_course_allocation_considering_not_predicted_courses)
    # print("mean of errors: ", mean_error)
    # print("standard_deviation for errors: ", std_dev_error)
    # print("mean of absolute errors: ", mean_ab_error)
    # print("standard_deviation for absolute errors: ", std_dev_ab_error)
    # print("mean of normalized errors: ", mean_st_error)
    # print("standard_deviation for normalized errors: ", std_dev_st_error)
    # # print(term_dict[1221])
    # # print(term_dict_predicted[1221])
    # # #print(reversed_item_dict)

    # #using # students took the course in the last offering
    # avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list = calculate_mse_for_course_allocation_last_offering(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg, course_sd_main, course_number_terms, term_dict_all, output_dir)
    # avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    # avg_rmsse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_msse_for_course_allocation_considering_not_predicted_courses)
    # mean_error, std_dev_error = calculate_std_dev(error_list)
    # mean_ab_error, std_dev_ab_error = calculate_std_dev(ab_error_list)
    # mean_st_error, std_dev_st_error = calculate_std_dev(st_error_list)
    # #print("avg mse for # of allocated course where we are predicting a course at least once: ",avg_mse_for_course_allocation)
    # #print("avg_mse_for_course_allocation_considering all courses available in test data: ",avg_mse_for_course_allocation_considering_not_predicted_courses)
    # #print("avg rmse for # of allocated course where we are predicting a course at least once: ",avg_rmse_for_course_allocation)
    # print("avg_mae_for_course_allocation_considering all courses available in test data: ",avg_mae_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmse_for_course_allocation_considering all courses available in test data: ",avg_rmse_for_course_allocation_considering_not_predicted_courses)
    # print("avg_mase_for_course_allocation_considering all courses available in test data: ",avg_mase_for_course_allocation_considering_not_predicted_courses)
    # print("avg_rmsse_for_course_allocation_considering all courses available in test data: ",avg_rmsse_for_course_allocation_considering_not_predicted_courses)
    # print("mean of errors: ", mean_error)
    # print("standard_deviation for errors: ", std_dev_error)
    # print("mean of absolute errors: ", mean_ab_error)
    # print("standard_deviation for absolute errors: ", std_dev_ab_error)
    # print("mean of normalized errors: ", mean_st_error)
    # print("standard_deviation for normalized errors: ", std_dev_st_error)
    # # print(term_dict[1221])
    # # print(term_dict_predicted[1221])
    # #print(reversed_item_dict)
    
