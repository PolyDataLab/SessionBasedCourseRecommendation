import pandas as pd
import numpy as np
import utils
import json
import re
#gets a text file as argument (sequence of baskets for training)
# calculate item_list from training data
def sequence_of_baskets(input_file):
    item_list = []
    with open('./train_main.txt', 'w') as f:
        for line in input_file:
            elements = line.split("|")
            if len(elements) >= 5:
                f.write(elements[0])
                f.write("|")
                f.write(elements[1])
                basket_sequence = elements[2:]
                for basket in basket_sequence:
                    if(len(basket)>0):
                        f.write('|')
                        f.write(basket)
                        basket1 = re.split('[\\s]+', basket)
                        for item in basket1:
                            if(len(item)>0 and item not in item_list):
                                item_list.append(item)
                f.write('\n')

            # elif len(elements)>5:
            #     for i in range(4,len(elements)):
            #         f.write(elements[0])
            #         f.write("|")
            #         f.write(elements[1])
            #         basket_sequence = elements[2:i+1]
            #         for basket in basket_sequence:
            #             if(len(basket)>0):
            #                 f.write('|')
            #                 f.write(basket)
            #                 basket1 = re.split('[\\s]+', basket)
            #                 for item in basket1:
            #                     if(len(item)>0 and item not in item_list):
            #                         item_list.append(item)
            #         f.write('\n')        
    #return basket_sequence
    return item_list
#gets a text file as argument (sequence of baskets for validating)
#delete items which are not available in training data
def sequence_of_baskets_valid(input_file, item_list):

    with open('./validate_main2.txt', 'w') as f:
        for line in input_file:
            elements = line.split("|")
            if len(elements) >= 5:
                f.write(elements[0])
                f.write("|")
                f.write(elements[1])
                basket_sequence = elements[2:]
                for basket in basket_sequence:
                    if(len(basket)>0):
                        #f.write('|')
                        #f.write(basket)
                        basket1 = re.split('[\\s]+', basket)
                        iid =0
                        for item in basket1:
                            if len(item)>0 and item in item_list:
                                if(iid==0): 
                                    f.write('|')
                                    f.write(item)
                                    iid+=1
                                else:
                                    f.write(' ')
                                    f.write(item)
                                    iid += 1
                        # if(j<3):
                        #     f.write('|')
                        # j= j+1
                f.write('\n')


            # elif len(elements)>5:
            #     for i in range(4,len(elements)):
            #         f.write(elements[0])
            #         f.write("|")
            #         f.write(elements[1])
            #         basket_sequence = elements[2:i+1]
            #         for basket in basket_sequence:
            #             if(len(basket)>0):
            #                 #f.write('|')
            #                 #f.write(basket)
            #                 basket1 = re.split('[\\s]+', basket)
            #                 iid =0
            #                 for item in basket1:
            #                     if len(item)>0 and item in item_list:
            #                         if(iid==0): 
            #                             f.write('|')
            #                             f.write(item)
            #                             iid+=1
            #                         else:
            #                             f.write(' ')
            #                             f.write(item)
            #                             iid += 1
            #         f.write('\n')


#filtering out sequences with less than 3 from validation data
def filter_valid_data(input_data):
    with open('./validate_main.txt', 'w') as f:
        for line in input_data:
            elements = line.split("|")
            if len(elements) >= 5:
                f.write(elements[0])
                f.write("|")
                f.write(elements[1])
                basket_sequence = elements[2:]
                for basket in basket_sequence:
                    f.write("|")
                    f.write(basket)
                f.write("\n")

#delete items which are not available in training and validation data
def delete_items_from_test_data(testing_instances, item_dict):
    new_file_link = './test_main2.txt'
    f = open(new_file_link, 'w')
    # item_dict2 = {}
    for line in testing_instances:
        elements = line.split("|")
        j=0
        basket_sequence = elements[2:]
        f.write(elements[0])
        f.write("|")
        f.write(elements[1])
        f.write("|")
        for basket1 in basket_sequence:
            basket = re.split('[\\s]+', basket1)
            #basket = basket1.split(" ")
            iid = 0
            cnt = 0
            for item in basket:
                if item in item_dict and len(item)>0:
                    if iid==0:
                        f.write(item)
                        cnt += 1
                        iid += 1
                    else:
                        f.write(" ")
                        f.write(item)
                        iid += 1
                        cnt+=1
                # else:
                #     item_dict2[item] = len(item_dict2)
            if(j<len(basket_sequence)-1 and cnt>0):
                f.write('|')
                j= j+1
        f.write('\n')
    #print(item_dict2)

    #reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    #return item_dict, reversed_item_dict, item_probs
#delete_lines_with the length less than 3 from test data
def delete_lines_with_less_lenghts_from_test(testing_instances):
     with open('./test_main.txt', 'w') as f:
        for line in testing_instances:
            elements = line.split("|")
            flag =0
            if len(elements)>=5:
                basket_sequence = elements[2:]
                for basket1 in basket_sequence:
                    if(len(basket1)>0):
                         flag += 1
            if(flag>=3):
                f.write(elements[0])
                f.write("|")
                f.write(elements[1])
                basket_sequence = elements[2:]
                for basket1 in basket_sequence:
                    if(len(basket1)>0):
                        #basket = re.split('[\\s]+', basket1)
                        #basket = basket1.split(" ")
                        f.write("|")
                        f.write(basket1)
            f.write("\n")  
#convert csv file to txt file where each line is a basket sequence
def convert_df_to_txt(input_file, input_path):
    f = open(input_path, "w")
    #dataset = pd.read_csv(input_file)
    dataset = input_file
    #dataframe
    #df1= pd.DataFrame(dataset)
    #print(df1)
    #df= df1.loc[:, 'userID':'Semester']
    #sequence of semesters
    #df["timestamp"]=df["Semester"].transform(lambda x:x.replace("Spring-", "1").replace("Summer-","2").replace("Fall-","3"))
    #20211 20212 20213 20221
    #df["timestamp"]=df["timestamp"].transform(lambda x:x[1:]+x[0])
    #sorting
    #df = df.sort_values(by="timestamp")
    #print(df) 
    #df["itemID"]=df["itemID"].transform(lambda x:x.replace("CIS",""))
    #print(data)
    #df["itemID"]=df["itemID"].transform(lambda x: int(x))
    #basket of courses in a semester
    #df["courses"]= df.groupby(["userID", "timestamp"])["itemID"].transform(lambda x: ' '.join(x))
    #df1=df[["userID", "timestamp","courses"]].drop_duplicates().reset_index()
    #print(df1)
   
    #sequence of baskets
    #df1["course_sequence"] = df1.groupby(["userID"])['courses'].transform(lambda x: "|".join(x))
    #df2 = df1[["userID","course_sequence"]].drop_duplicates().reset_index()
    for x in range(len(dataset)):
        f.write(str(dataset["userID"][x])+"|")
        f.write(str(dataset["last_semester"][x])+"|")
        baskets = dataset["baskets"][x]
        bid =0
        for basket in baskets:
            iid = 0
            for item in basket:
                f.write(item)
                if iid<len(basket)-1:
                    f.write(" ")
                    iid += 1
            if(bid<len(baskets)-1):        
                f.write("|")
                bid+=1
        #f.write(str(df2["course_sequence"][x]))
        f.write("\n")
    #print(df2)
    #text file generation
    #np.savetxt(r'./basket_seq2.txt', df2['course_sequence'].values, fmt='%s')
    return dataset

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
    test_set_without_summer.to_json('./test_sample_all_without_summer.json', orient='records', lines=True)
    return test_set_without_summer

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
    valid_set_without_summer.to_json('./valid_sample_all_without_summer.json', orient='records', lines=True)
    return valid_set_without_summer

if __name__ == '__main__':
    data_dir= './'
    input_dir = data_dir + "/input_dir"
    utils.create_folder(input_dir)
    # train_data = pd.read_json('./Filtered_data/train_sample_augmented_without_summer.json', orient='records', lines= True)
    # #file_path1 = './train_sample.csv'
    train_data = pd.read_json('./Filtered_data/train_sample_augmented.json', orient='records', lines= True)
    #train_data = pd.read_json('./Filtered_data/train_sample_augmented_merging_summer_fall.json', orient='records', lines= True)
    #file_path1 = './train_sample.csv'
    file_path1 = train_data
    input_path= input_dir+ "/train_data.txt"
    converted_txt_df1 = convert_df_to_txt(file_path1, input_path)
    # valid_data = pd.read_json('./valid_data_all_without_summer.json', orient='records', lines= True)
    #valid_data = pd.read_json('./valid_data_all_CR.json', orient='records', lines= True)
    valid_data = pd.read_json('./Filtered_data/valid_sample_all.json',  orient='records', lines= True)
    #valid_data = remove_summer_term_from_valid(valid_data)
    #valid_data = pd.read_json('./valid_data_all_merging_summer_fall.json', orient='records', lines= True)
    #print(converted_txt_df1)
    #file_path2 = './valid_sample.csv'
    file_path2 = valid_data
    input_path2= input_dir+ "/valid_data.txt"
    converted_txt_df2 = convert_df_to_txt(file_path2, input_path2)
    # test_data = pd.read_json('./test_data_all_without_summer.json', orient='records', lines= True)
    #test_data = pd.read_json('./test_data_all_CR.json', orient='records', lines= True)
    test_data = pd.read_json('./Filtered_data/test_sample_all.json',  orient='records', lines= True)
    #test_data = remove_summer_term_from_test(test_data)
    #test_data = pd.read_json('./test_data_all_merging_summer_fall.json', orient='records', lines= True)
    #file_path3 = './test_sample.csv'
    file_path3 = test_data
    input_path3= input_dir+ "/test_data.txt"
    converted_txt_df3 = convert_df_to_txt(file_path3, input_path3)
    #print(converted_txt_df3)
            
           
