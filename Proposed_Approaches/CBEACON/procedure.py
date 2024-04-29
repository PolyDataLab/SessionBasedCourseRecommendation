import tensorflow as tf
import numpy as np

import sys
import utils
import time
import re
#from utils import *
import math
import pandas as pd


def train_network(sess, net, train_generator, validate_generator, nb_epoch, 
                  total_train_batches, total_validate_batches, display_step,
                  early_stopping_k, epsilon, tensorboard_dir, output_dir,
                  test_generator, total_test_batches):
    summary_writer = None
    if tensorboard_dir is not None:
        summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_dir)
    # Add ops to save and restore all the variables.
    saver = tf.compat.v1.train.Saver()

    val_best_performance = [sys.float_info.max]
    patience_cnt = 0
    for epoch in range(0, nb_epoch):
        print("\n=========================================")
        print("@Epoch#" + str(epoch))

        train_loss = 0.0
        train_recall = 0.0

        for batch_id, data in train_generator:
            start_time = time.time()
            loss, recall, summary, values, indices = net.train_batch(data['S'], data['L'], data['Y'], data['K'])
            #loss, recall, summary = net.train_batch(data['S'], data['L'], data['Y'])
            train_loss += loss
            avg_train_loss = train_loss / (batch_id + 1)

            train_recall += recall
            avg_train_recall = train_recall / (batch_id + 1)
            
            # Write logs at every iteration
            if summary_writer is not None:
                summary_writer.add_summary(summary, epoch * total_train_batches + batch_id)

                loss_sum = tf.compat.v1.Summary()
                loss_sum.value.add(tag="Losses/Train_Loss", simple_value=avg_train_loss)
                summary_writer.add_summary(loss_sum, epoch * total_train_batches + batch_id)

                recall_sum = tf.compat.v1.Summary()
                recall_sum.value.add(tag="Recalls/Train_Recall", simple_value=avg_train_recall)
                summary_writer.add_summary(recall_sum, epoch * total_train_batches + batch_id)


            if batch_id % display_step == 0 or batch_id == total_train_batches - 1:
                running_time = time.time() - start_time
                print("Training | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_train_batches) 
                    + " | Loss= " + "{:.8f}".format(avg_train_loss)  
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_train_recall) 
                    + " | Time={:.2f}".format(running_time) + "s")
                

            if batch_id >= total_train_batches - 1:
                break

        print("\n-------------- VALIDATION LOSS--------------------------")
        val_loss = 0.0
        val_recall = 0.0
        for batch_id, data in validate_generator:
            loss, recall, summary, values, indices = net.validate_batch(data['S'], data['L'], data['Y'])
            #loss, recall, summary = net.validate_batch(data['S'], data['L'], data['Y'])
           
            val_loss += loss
            avg_val_loss = val_loss / (batch_id + 1)

            val_recall += recall
            avg_val_recall = val_recall / (batch_id + 1)

            # Write logs at every iteration
            if summary_writer is not None:
                summary_writer.add_summary(summary, epoch * total_validate_batches + batch_id)

                loss_sum = tf.compat.v1.Summary()
                loss_sum.value.add(tag="Losses/Val_Loss", simple_value=avg_val_loss)
                summary_writer.add_summary(loss_sum, epoch * total_validate_batches + batch_id)

                recall_sum = tf.compat.v1.Summary()
                recall_sum.value.add(tag="Recalls/Val_Recall", simple_value=avg_val_recall)
                summary_writer.add_summary(recall_sum, epoch * total_validate_batches + batch_id)

            if batch_id % display_step == 0 or batch_id == total_validate_batches - 1:
                print("Validating | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_validate_batches) 
                    + " | Loss = " + "{:.8f}".format(avg_val_loss)
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_val_recall))
            
            if batch_id >= total_validate_batches - 1:
                break

        print("\n-------------- TEST LOSS--------------------------")
        test_loss = 0.0
        test_recall = 0.0
        #print_ind = True
        for batch_id, data in test_generator:
            loss, recall, _, values, indices = net.validate_batch(data['S'], data['L'], data['Y'])
            #loss, recall, _ = net.validate_batch(data['S'], data['L'], data['Y'])
    
            test_loss += loss
            avg_test_loss = test_loss / (batch_id + 1)

            test_recall += recall
            avg_test_recall = test_recall / (batch_id + 1)

            # Write logs at every iteration
            if summary_writer is not None:
                #summary_writer.add_summary(summary, epoch * total_test_batches + batch_id)

                loss_sum = tf.compat.v1.Summary()
                loss_sum.value.add(tag="Losses/Test_Loss", simple_value=avg_test_loss)
                summary_writer.add_summary(loss_sum, epoch * total_test_batches + batch_id)

                recall_sum = tf.compat.v1.Summary()
                recall_sum.value.add(tag="Recalls/Test_Recall", simple_value=avg_test_recall)
                summary_writer.add_summary(recall_sum, epoch * total_test_batches + batch_id)

            if batch_id % display_step == 0 or batch_id == total_test_batches - 1:
                print("Testing | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_test_batches) 
                    + " | Loss = " + "{:.8f}".format(avg_test_loss)
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_test_recall))
            
            if batch_id >= total_test_batches - 1:
                break
        #print_ind = False

        if summary_writer is not None:
            I_B= net.get_item_bias()
            item_probs = net.item_probs

            I_B_corr = np.corrcoef(I_B, item_probs)
            I_B_summ = tf.compat.v1.Summary()
            I_B_summ.value.add(tag="CorrCoef/Item_Bias", simple_value=I_B_corr[1][0])
            summary_writer.add_summary(I_B_summ, epoch)

        avg_val_loss = val_loss / total_validate_batches
        print("\n@ The validation's loss = " + str(avg_val_loss))
        imprv_ratio = (val_best_performance[-1] - avg_val_loss)/val_best_performance[-1]
        if imprv_ratio > epsilon:
            print("# The validation's loss is improved from " + "{:.8f}".format(val_best_performance[-1]) + \
                  " to " + "{:.8f}".format(avg_val_loss))
            val_best_performance.append(avg_val_loss)

            patience_cnt = 0

            save_dir = output_dir + "/epoch_" + str(epoch)
            utils.create_folder(save_dir)

            save_path = saver.save(sess, save_dir + "/model.ckpt")
            print("The model is saved in: %s" % save_path)
        else:
            patience_cnt += 1

        if patience_cnt >= early_stopping_k:
            print("# The training is early stopped at Epoch " + str(epoch))
            break

#calculate recall after using filtering function
def recall_cal(top_item_list, target_item_list, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred):
    #print("top_items: ", top_item_list)
    #print("target basket: ", target_item_list)
    t_length= len(target_item_list)
    correct_preds= len((set(top_item_list) & set(target_item_list)))
    actual_bsize= t_length
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

#recommend top_k items for target basket using filtering function
def recommend_top_k(user_baskets, prob_item, offered_course_list, top_k1, item_dict):
    prob_item = dict(sorted(prob_item.items(), key=lambda item: item[1], reverse= True))
    top_items= []
    count =0
    #top_k1= 5 #top_k
    for item1 in prob_item.keys():
        #using filtering function
        if not utils.filtering(item1, user_baskets, offered_course_list, item_dict):
            top_items.append(item1)
            count+= 1
        if(count== top_k1): break
    return top_items

def course_CIS_dept_filtering(course):
    list_of_terms = ["CAP", "CDA", "CEN", "CGS", "CIS", "CNT", "COP", "COT", "CTS", "IDC","IDS"]
    flag = 0
    for term in list_of_terms:
        if course.find(term)!= -1:
            flag = 1
    return flag 

#Tune the model using validation data to set hypermataters based on lower validation loss
def tune(net, data_generator, total_batches, display_step, item_dict, rev_item_dict, output_file, offered_courses_valid, nb_validate, batch_size):
    f = open(output_file, "w")
    val_loss = 0.0
    val_recall = 0.0
    recall_main= 0.0
    recall_average = 0.0
    cor_preds_total = 0
    actual_bsize_total= 0
    count2 = 0.0
    count_at_least_one_cor_pred = 0
    #recall_test_for_one_cor_pred = 0.0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    #count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
    recall_temp =0
    recall_bsize = {}
    missed_bsize = {}
    retake_bsize = {}
    for batch_id, data in data_generator:
        loss, recall, _, values, indices = net.validate_batch(data['S'], data['L'], data['Y'])
        #loss, recall, _ = net.validate_batch(data['S'], data['L'], data['Y'])
             
        val_loss += loss
        avg_val_loss = val_loss / (batch_id + 1)

        val_recall += recall
        avg_val_recall = val_recall / (batch_id + 1)
        recall_1= 0.0
        count1 = 0

        for i, (seq_val, seq_ind) in enumerate(zip(values, indices)):
            if batch_id==total_batches-1:
                if i==(nb_validate % batch_size): break # not considering extra instances of last batch
            #items in the target basket 
            target_basket= data['O'][i].split(" ")
            target_basket1= []
            target_basket2 = []
            for x1 in range(len(target_basket)):
                if(len(target_basket[x1])>0):
                    target_basket1.append(item_dict[target_basket[x1]])
                    target_basket2.append(target_basket[x1])
            target_basket = target_basket1.copy()
            #len_target_basket= len(target_basket)
            #items in the user baskets
            user_baskets= data['S'][i]
            prior_bsize = data['L'][i]
            target_semester = data['T'][i]
            # target_basket2 = target_basket.copy()
            prior_courses = []
            for basket3 in user_baskets:
                for item4 in basket3:
                    if rev_item_dict[item4] not in prior_courses:
                        prior_courses.append(rev_item_dict[item4])

            #creating dictionary with item's index as key and probability as value
            prob_item= {}
            for (v, idx) in zip(seq_val, seq_ind):
                prob_item[idx] = v

            top_items = recommend_top_k(user_baskets, prob_item, offered_courses_valid[int(target_semester)], len(target_basket), item_dict)
            rec_basket2 = []
            for item3 in top_items:
                rec_basket2.append(rev_item_dict[item3])
            
            recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred  = recall_cal(top_items, target_basket, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)  
            recall_1 += recall_temp
            #recall_1 += recall_cal(top_items, target_basket) 
            count1 += 1
            if prior_bsize not in recall_bsize:
                recall_bsize[prior_bsize]= [recall_temp]
            else:
                recall_bsize[prior_bsize] += [recall_temp]
             #number of non-CIS and retake courses out of missed courses    
            n_missed = 0
            n_retake = 0
            n_non_CIS =0
            for course2 in target_basket2:
                if course2 not in rec_basket2:
                    n_missed += 1
                    if course_CIS_dept_filtering(course2)==0:
                        n_non_CIS +=1
                    if course2 in prior_courses:
                        n_retake += 1
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
            # t_length= len(target_basket)
            # correct_preds2= len((set(top_items) & set(target_basket)))
            # actual_bsize2= t_length
            # cor_preds_total += correct_preds2
            # actual_bsize_total += actual_bsize2


        recall_main= recall_1/ count1
        recall_average += recall_1
        count2 += count1
        # Write logs at every iteration
        if batch_id % display_step == 0 or batch_id == total_batches - 1:
            print(str(batch_id + 1) + "/" + str(total_batches) + " | Loss = " + "{:.8f}".format(avg_val_loss)
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_val_recall))
            print("For batch id "+ str(batch_id + 1) + ": validation Recall@n"+ " after using filtering function: "+ str(recall_main))

        if batch_id >= total_batches - 1:
            break
    avg_val_recall = val_recall / total_batches
    f.write(str(avg_val_recall) + "\n")
    recall_average = recall_average/count2
    # recall_average = cor_preds_total/ actual_bsize_total
    print (": average recall@n"+ " for all validating batches after using filtering function: "+ str(recall_average))
    f.write("average recall@n"+ " for all validating batches after using filtering function: "+ str(recall_average))
     # recall scores for different basket sizes
    recall_bsize = dict(sorted(recall_bsize.items(), key=lambda item: item[0], reverse= False))
    for k, v in recall_bsize.items():
        bsize = k
        sum = 0
        for r in v:
            sum += r
        recall2 = sum/len(v)
        print("prior basket size: ", bsize)
        print("number of instances: ", len(v))
        print("recall score for validation data: ", recall2)
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
    #     print(" percentage of non CIS courses out of missed courses for validation data: ", per_of_non_CIS)
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
    #     print("percentage of retaken courses out of missed courses for validation data: ", per_of_retaken_courses)
    f.close()

def generate_prediction_for_training(net, data_generator, total_train_batches, display_step, item_dict, inv_item_dict, output_file, offered_courses_train):
    f = open(output_file, "w")
    train_loss = 0.0
    train_recall = 0.0
    recall_main= 0.0 #recall after using filtering function
    recall_average = 0.0
    count2 =0
    count_at_least_one_cor_pred = 0
    #recall_test_for_one_cor_pred = 0.0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    #count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    recall_temp =0
    
    for batch_id, data in data_generator:
        loss, recall, _, values, indices = net.validate_batch(data['S'], data['L'], data['Y'])
        #loss, recall, _ = net.train_batch(data['S'], data['L'], data['Y'])
            
        train_loss += loss
        avg_train_loss = train_loss / (batch_id + 1)

        train_recall += recall
        avg_train_recall = train_recall / (batch_id + 1)
        recall_1= 0.0
        count1 = 0

        for i, (seq_val, seq_ind) in enumerate(zip(values, indices)):
            #items in the target basket 
            target_basket= data['O'][i].split(" ")
            for x1 in range(len(target_basket)):
                target_basket[x1]= item_dict[target_basket[x1]]
            #len_target_basket= len(target_basket)
            #items in the user baskets
            user_baskets= data['S'][i]
            #len_user_baskets = data['L'][i]
            target_semester = data['T'][i]

            #creating dictionary with item's index as key and probability as value
            prob_item= {}
            for (v, idx) in zip(seq_val, seq_ind):
                prob_item[idx] = v
            
            #print("prob item: ")
            #print(prob_item)
            top_items = recommend_top_k(user_baskets, prob_item, offered_courses_train[int(target_semester)], len(target_basket), item_dict)
            recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred  = recall_cal(top_items, target_basket, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)  
            recall_1 += recall_temp 
            count1 += 1
        recall_main= recall_1/ count1
        recall_average += recall_1
        count2 += count1

        # Write logs at every iteration
        if batch_id % display_step == 0 or batch_id == total_train_batches - 1:
            print(str(batch_id + 1) + "/" + str(total_train_batches) + " | Loss = " + "{:.8f}".format(avg_train_loss)
                    + " | Recall@"+ str(net.top_k) + " = " + "{:.8f}".format(avg_train_recall))
            print("For batch id "+ str(batch_id + 1) + ": train Recall@n"+ " after using filtering function: "+ str(recall_main))

        if batch_id >= total_train_batches - 1:
            break
    avg_train_recall = train_recall / total_train_batches
    f.write(str(avg_train_recall) + "\n")
    recall_average = recall_average/count2
    print (": average recall@n"+ " for all training batches after using filtering function: "+ str(recall_average))
    f.write("average recall@n"+ " for all training batches after using filtering function: "+ str(recall_average))
    f.close()

# calculating term dictionary where key = semester and value = course with number of actual enrollments
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
            count_course[item] = count_course[item] + 1
        #if semester==1221 and item=="COP4710": print("Count of course COP4710 in 1221 semester:", count_course[item])
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
                count_course[reversed_item_dict[item]] = count_course[reversed_item_dict[item]] + 1
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

# calculate standard deviation of errors
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

#calculate mse, mae, rmse
def calculate_mse_for_course_allocation(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg, course_sd_main, course_number_terms, term_dict_all_prior, output_dir):
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
        if semester %5==0:
            prior_semester = semester-4
        else:
            prior_semester = semester-3

        if semester in term_dict_predicted:
            count_course_predicted = term_dict_predicted[semester]
            count_course_predicted_true = term_dict_predicted_true[semester]
            count_course_predicted_false = term_dict_predicted_false[semester]

            
            for item1 in count_course.keys():
                f.write("Semester: ")
                f.write(str(semester)+ " ")
                f.write("Course ID: ")
                f.write(str(item1)+ " ")
                count_course_prior = find_prior_term(item1, prior_semester, term_dict_all_prior)

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
                        row = [semester, item1, count_course[item1], 0, 0, 0, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1], count_course_prior[item1]]
                    else:
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
    course_allocation_actual_predicted.to_csv(output_dir+'/course_allocation_actual_predicted_updated_new_v2.csv')
    return avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list

# generate recommendation for test data and use them for course enrollment prediction
def generate_prediction(net, data_generator, total_test_batches, display_step, item_dict, inv_item_dict, output_file, offered_courses_test, output_path, output_dir2, nb_test, batch_size):
    f = open(output_file, "w") #recommendation without filtering function
    f1 = open(output_path, "w") #recommendation using filtering function

    recall_main= 0.0
    recall_average = 0.0
    count2 = 0
    count_at_least_one_cor_pred = 0
    recall_test_for_one_cor_pred = 0.0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
    recall_temp =0
    total_correct_preds = 0
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
    count12 = 0
    recall_bsize = {}
    missed_bsize = {}
    retake_bsize = {}
    rec_info = []
    for batch_id, data in data_generator:
        values, indices = net.generate_prediction(data['S'], data['L'])
        recall_1= 0.0
        count1 = 0
        bid= batch_id + 1
        f1.write("Batch ID: "+ str(bid)+ "\n")
        count_iter = 0
        for i, (seq_val, seq_ind) in enumerate(zip(values, indices)):
            #dealing with last batch
            if batch_id==total_test_batches-1:
                if count_iter==(nb_test % batch_size): break # not considering extra instances of last batch
            count_iter += 1

            f.write("Target:" + data['O'][i])
            f1.write("User ID:" + data['U'][i])
            #items in the target basket 
            #target_basket1= data['O'][i].split(" ")
            target_basket1 = re.split('[\\s]+', data['O'][i])
            target_basket = []
            for item7 in target_basket1:
                if(len(item7)>0):
                    target_basket.append(item7)
            
            f1.write(", Target basket:" + str(target_basket))
            # print("User ID: ", data['U'][i])
            # print(", Target basket:" + str(target_basket))

            #len_target_basket = len(target_basket)
            target_semester = data['T'][i]
            target_basket2 = []
            #index1= 0
            target_basket3 = []
            for x1 in range(len(target_basket)):
                if(len(target_basket[x1])>0):
                    #target_basket[x1]= item_dict[target_basket[x1]]
                    target_basket2.append(item_dict[target_basket[x1]])
                    target_basket3.append(target_basket[x1])
                    #index1 +=1
            #target_basket = target_basket2.copy()
            # if target_semester=='1221':
            #     for item12 in target_basket:
            #         if item12=='COP4710':
            #             count12+=1

            f1.write(", Target basket with ids:" + str(target_basket2))
            #len_target_basket= len(target_basket)
            #items in the user baskets
            user_baskets= data['S'][i]
            prior_bsize = data['L'][i]
            prior_courses = []
            for basket3 in user_baskets:
                for item4 in basket3:
                    if inv_item_dict[item4] not in prior_courses:
                        prior_courses.append(inv_item_dict[item4])
            #len_user_baskets = data['L'][i]
            #values2, indices2 = net.generate_prediction(data['S'][i], data['L'][i])

            #creating dictionary with item's index as key and probability as value
            prob_item= {}
            for (v, idx) in zip(seq_val, seq_ind):
                f.write("|" + str(inv_item_dict[idx]) + ":" + str(v))
                prob_item[idx] = v
            
            # for (v, idx) in zip(values2, indices2):
            #     f.write("|" + str(inv_item_dict[idx]) + ":" + str(v))
            #     prob_item[idx] = v

            f.write("\n")
            #prob_item = dict(sorted(prob_item.items(), key=lambda item: item[1], reverse= True))
            #print("prob item: ")
            #print(prob_item)
            #top_k1 = len_target_basket
            #recall_temp =0.0
            top_items = recommend_top_k(user_baskets, prob_item, offered_courses_test[int(target_semester)], len(target_basket2), item_dict)
            rec_basket2 = []
            for item3 in top_items:
                rec_basket2.append(inv_item_dict[item3])
            #print(top_items)
            #f1.write(str(user)+ ": ")
            f1.write(", Recommended basket: ")
            for item3 in top_items:
                f1.write(str(inv_item_dict[item3])+ " ")
            f1.write("\n")
            recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred  = recall_cal(top_items, target_basket2, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)  
            t_length= len(target_basket2)
            if t_length>=2: count_actual_bsize_at_least_2 += 1
            if t_length>=3: count_actual_bsize_at_least_3 += 1
            if t_length>=4: count_actual_bsize_at_least_4 += 1
            if t_length>=5: count_actual_bsize_at_least_5 += 1
            if t_length>=6: count_actual_bsize_at_least_6 += 1
            rel_rec = len((set(top_items) & set(target_basket2)))
            row = [t_length, target_basket3, rec_basket2, rel_rec, recall_temp, target_semester]
            rec_info.append(row)
            # test_rec_info = pd.DataFrame(rec_info, columns=['bsize', 'target_courses', 'rec_courses', 'n_rel_rec', 'recall_score', 'target_semester'])
            # test_rec_info.to_json('./Others/DREAM/test_rec_info.json', orient='records', lines=True)
            # test_rec_info.to_csv('./Others/DREAM/test_rec_info.csv')
            if recall_temp>0:  recall_test_for_one_cor_pred += recall_temp
            correct_preds2= len((set(top_items) & set(target_basket2)))
            total_correct_preds += correct_preds2
            if prior_bsize not in recall_bsize:
                recall_bsize[prior_bsize]= [recall_temp]
            else:
                recall_bsize[prior_bsize] += [recall_temp]
             # number of non-CIS and retake courses out of missed courses    
            n_missed = 0
            n_retake = 0
            n_non_CIS =0
            for course2 in target_basket3:
                if course2 not in rec_basket2:
                    n_missed += 1
                    if course_CIS_dept_filtering(course2)==0:
                        n_non_CIS +=1
                    if course2 in prior_courses:
                        n_retake += 1
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

            if t_length>=6: target_basket_size[6] += 1 
            else: target_basket_size[t_length] += 1 
            recall_1 += recall_temp
            #recall_1 += recall_cal(top_items, target_basket) 
            #course allocation for courses in target basket
            # term_dict = calculate_term_dict(term_dict, int(target_semester), target_basket, inv_item_dict)
            term_dict = calculate_term_dict_2(term_dict, int(target_semester), target_basket, inv_item_dict)

            #course allocation for predicted courses
            term_dict_predicted = calculate_term_dict(term_dict_predicted, int(target_semester), top_items, inv_item_dict)
            term_dict_predicted_true = calculate_term_dict_true(term_dict_predicted_true, int(target_semester), target_basket2, top_items, inv_item_dict)
            term_dict_predicted_false = calculate_term_dict_false(term_dict_predicted_false, int(target_semester), target_basket2, top_items, inv_item_dict)
            count1 += 1
        recall_main= recall_1/ count1
        recall_average += recall_1
        count2 += count1


        if batch_id % display_step == 0 or batch_id == total_test_batches - 1:
            print(str(batch_id + 1) + "/" + str(total_test_batches))

            print("For batch id "+ str(batch_id + 1) + ": test recall@n"+ " after using filtering function: "+ str(recall_main))

            f1.write(str(batch_id + 1) + "/" + str(total_test_batches))

            f1.write("For batch id "+ str(batch_id + 1) + ": test recall@n" + " after using filtering function: "+ str(recall_main))


        if batch_id >= total_test_batches - 1:
            break
    recall_average = recall_average/count2
    print("total count = ", count2)
    #print("Count of COP4710 in 1221 semester: ", count12)
    print (": average recall@n"+ " for all testing batches after using filtering function: "+ str(recall_average))
    f1.write("average recall@n"+ " for all testing batches after using filtering function: "+ str(recall_average))
    print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
    percentage_of_at_least_one_cor_pred = (count_at_least_one_cor_pred/ count2) *100
    print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
    f1.write("percentage_of_at_least_one_cor_pred: "+ str(percentage_of_at_least_one_cor_pred)+ "\n")

    percentage_of_at_least_two_cor_pred = (count_at_least_two_cor_pred/ count_actual_bsize_at_least_2) *100
    print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
    f1.write("percentage_of_at_least_two_cor_pred: "+ str(percentage_of_at_least_two_cor_pred)+ "\n")
    percentage_of_at_least_three_cor_pred = (count_at_least_three_cor_pred/ count_actual_bsize_at_least_3) *100
    print("percentage_of_at_least_three_cor_pred: ", percentage_of_at_least_three_cor_pred)
    f1.write("percentage_of_at_least_three_cor_pred: "+ str(percentage_of_at_least_three_cor_pred)+ "\n")
    percentage_of_at_least_four_cor_pred = (count_at_least_four_cor_pred/ count_actual_bsize_at_least_4) * 100
    print("percentage_of_at_least_four_cor_pred: ", percentage_of_at_least_four_cor_pred)
    f1.write("percentage_of_at_least_four_cor_pred: "+ str(percentage_of_at_least_four_cor_pred)+ "\n")
    percentage_of_at_least_five_cor_pred = (count_at_least_five_cor_pred/ count_actual_bsize_at_least_5) *100
    print("percentage_of_at_least_five_cor_pred: ", percentage_of_at_least_five_cor_pred)
    f1.write("percentage_of_at_least_five_cor_pred: "+ str(percentage_of_at_least_five_cor_pred)+ "\n")
    percentage_of_all_cor_pred = (count_all_cor_pred/ count2) *100
    print("percentage_of_all_cor_pred: ", percentage_of_all_cor_pred)
    f1.write("percentage_of_all_cor_pred: "+ str(percentage_of_all_cor_pred)+ "\n")
    #calculate Recall@n for whom we generated at least one correct prediction in test data
    test_recall_for_one_cor_pred = recall_test_for_one_cor_pred/ count_at_least_one_cor_pred
    print("Recall@n for whom we generated at least one correct prediction in test data: ", test_recall_for_one_cor_pred)
    f1.write("Recall@n for whom we generated at least one correct prediction in test data:"+ str(test_recall_for_one_cor_pred))
    f1.write("\n") 

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
    # recall scores for different basket sizes
    # recall_bsize = dict(sorted(recall_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in recall_bsize.items():
    #     bsize = k
    #     sum = 0
    #     for r in v:
    #         sum += r
    #     recall2 = sum/len(v)
    #     print("prior basket size: ", bsize)
    #     print("number of instances: ", len(v))
    #     print("recall score for test data: ", recall2)
    
    #  # number of non_CIS courses out of missed courses for different number of prior semesters
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
    
    test_rec_info = pd.DataFrame(rec_info, columns=['bsize', 'target_courses', 'rec_courses', 'n_rel_rec', 'recall_score', 'target_semester'])
    test_rec_info.to_json('./Course_Beacon/test_rec_info.json', orient='records', lines=True)
    test_rec_info.to_csv('./Course_Beacon/test_rec_info.csv')
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
    #calculate average for total number of times a course is taken at each semester using training data 
    all_data_en_pred = pd.read_json('./all_data_en_pred_filtered.json', orient='records', lines= True)
    term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(all_data_en_pred, inv_item_dict)

    # valid_data_unique = pd.read_json('./Filtered_data/valid_sample_filtered_unique.json', orient='records', lines= True)
    # term_dict_valid, frequency_of_courses2, count_course_avg2, course_sd_main2, course_number_terms2 = calculate_avg_n_actual_courses(valid_data_unique, reversed_item_dict)

    # avg_mse_for_course_allocation, avg_mse_for_course_allocation_considering_not_predicted_courses = calculate_mse_for_course_allocation(term_dict, term_dict_predicted)
    # avg_rmse_for_course_allocation, avg_rmse_for_course_allocation_considering_not_predicted_courses = math.sqrt(avg_mse_for_course_allocation), math.sqrt(avg_mse_for_course_allocation_considering_not_predicted_courses)
    avg_mse_for_course_allocation_considering_not_predicted_courses, avg_mae_for_course_allocation_considering_not_predicted_courses, avg_msse_for_course_allocation_considering_not_predicted_courses, avg_mase_for_course_allocation_considering_not_predicted_courses, error_list, ab_error_list, st_error_list = calculate_mse_for_course_allocation(term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, count_total_course, item_dict, count_course_avg, course_sd_main, course_number_terms, term_dict_all, output_dir2)
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
    # #print(term_dict_predicted[1221])
    #print(reversed_item_dict)
    f.close()
    f1.close()
    print(" ==> PREDICTION HAS BEEN DONE!")



def recent_model_dir(dir):
    folder_list = utils.list_directory(dir, True)
    folder_list = sorted(folder_list, key=get_epoch)
    return folder_list[-1]


def get_epoch(x):
    idx = x.index('_') + 1
    return int(x[idx:])
