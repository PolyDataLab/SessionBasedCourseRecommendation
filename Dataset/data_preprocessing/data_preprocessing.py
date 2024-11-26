import pandas as pd
from sklearn.model_selection import train_test_split
import re
import json
import random
import math
def course_name_update(myStr):
    str_list = re.split(" ", myStr)
    # for x in range(len(str_list)-1):
    #     output_str += str_list[x]
        # if(x< len(str_list)-2):
        #     output_str += " "
    return ''.join(str_list[:2])

def preprocess_data(input_file):
    column_name = ["userID", "Semester", "Semester_details", "course_name","credits", "letter_grade", "numerical_grade"]
    df = pd.read_csv(input_file, header = None, names = column_name, engine= 'python')
    df2 = df[["userID", "course_name", "Semester", "letter_grade", "numerical_grade"]]
    #dropped_courses = df2[df2['letter_grade'] == 'DR']
    #print(p_graded_courses)
    df2 = df2[df2.letter_grade != "DR"]
    df2 = df2[df2.letter_grade != "F"]
    df2 = df2[df2.letter_grade != "F0"]
    df2 = df2[df2.letter_grade != "D"]
    df2 = df2.reset_index()
    #print(df2) 
    df2["itemID"] = df2["course_name"].transform(lambda x: course_name_update(x))
    df2 = df2.sort_values(by= "Semester")
    # print(df2)
    frequency_of_courses = {}
    for item in df2["itemID"]:
        if item not in frequency_of_courses:
            frequency_of_courses[item] = 1
        else:
            frequency_of_courses[item] += 1
    list_of_frequency = []
    for item in df2["itemID"]:
        list_of_frequency.append(frequency_of_courses[item])
    df2["frequency_of_items"] = list_of_frequency
    df2 = df2[df2.frequency_of_items > 2]
    df2 = df2.reset_index()
    df2["grades"] = df2["numerical_grade"]
    data = df2[["userID", "itemID", "Semester", "grades"]]
    #data = df2[["userID", "itemID", "Semester"]]
    data = data.sort_values(by= "Semester")
    data.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_CR.csv', header = True)
    data.to_json('/Users/mkhan149/Downloads/Experiments/all_data_CR.json', orient='records', lines=True)

    #print(data)
    return data

def preprocess_data_allocation(input_file):
    column_name = ["userID", "Semester", "Semester_details", "course_name","credits", "letter_grade", "numerical_grade"]
    df = pd.read_csv(input_file, header = None, names = column_name, engine= 'python')
    df2 = df[["userID", "course_name", "Semester", "letter_grade", "numerical_grade"]]
    #dropped_courses = df2[df2['letter_grade'] == 'DR']
    # print(p_graded_courses)
    df2 = df2[df2.letter_grade != "DR"]
    df2 = df2[df2.letter_grade != "F"]
    df2 = df2[df2.letter_grade != "F0"]
    df2 = df2[df2.letter_grade != "D"]  # less than C
    df2 = df2.reset_index()
    #print(df2) 
    df2["itemID"] = df2["course_name"].transform(lambda x: course_name_update(x))
    #df2 = df2.sort_values(by= "Semester")
    #print(df2)
    frequency_of_courses = {}
    for item in df2["itemID"]:
        if item not in frequency_of_courses:
            frequency_of_courses[item] = 1
        else:
            frequency_of_courses[item] += 1
    list_of_frequency = []
    for item in df2["itemID"]:
        list_of_frequency.append(frequency_of_courses[item])
    df2["frequency_of_items"] = list_of_frequency
    df2 = df2[df2.frequency_of_items > 9]
    df2 = df2.reset_index()
    df2["grades"] = df2["numerical_grade"]
    data = df2[["userID", "itemID", "Semester", "grades"]]
    #count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(data)
    data = data.sort_values(by= "Semester")
    data.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_CR.csv', header = True)
    
    #print(data)
    return data

def preprocess_data_en_pred(input_file):
    column_name = ["userID", "Semester", "Semester_details", "course_name","credits", "letter_grade", "numerical_grade"]
    df = pd.read_csv(input_file, header = None, names = column_name, engine= 'python')
    df2 = df[["userID", "course_name", "Semester", "letter_grade"]]
    #dropped_courses = df2[df2['letter_grade'] == 'DR']
    #print(p_graded_courses)
    # df2 = df2[df2.letter_grade != "DR"]
    # df2 = df2[df2.letter_grade != "F"]
    # df2 = df2[df2.letter_grade != "F0"]
    # df2 = df2[df2.letter_grade != "D"]
    # df2 = df2.reset_index()
    #print(df2) 
    df2["itemID"] = df2["course_name"].transform(lambda x: course_name_update(x))
    #df2 = df2.sort_values(by= "Semester")
    #print(df2)
    # frequency_of_courses = {}
    # for item in df2["itemID"]:
    #     if item not in frequency_of_courses:
    #         frequency_of_courses[item] = 1
    #     else:
    #         frequency_of_courses[item] += 1
    # list_of_frequency = []
    # for item in df2["itemID"]:
    #     list_of_frequency.append(frequency_of_courses[item])
    # df2["frequency_of_items"] = list_of_frequency
    # df2 = df2[df2.frequency_of_items > 9]
    # df2 = df2.reset_index()
    data = df2[["userID", "itemID", "Semester"]]
    data = data.sort_values(by= "Semester")
    data["timestamp"]=data["Semester"]
    data = data[["userID", "itemID", "timestamp"]]

    baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 
    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    semester_info = {}
    #data = df[["userID", "itemID", "timestamp"]]
    userIDs = data.userID.values
    index = 0
    for user in userIDs:
        ts = data['timestamp'][index]
        if user not in semester_info:
            semester_info[user] = ts
        if user in semester_info:
            semester_info[user] = ts
        index +=1

    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    all_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=3:
        row = [user, b, baskets.iloc[index]['num_baskets'], semester_info[user], baskets.iloc[index]['timestamps']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        all_data.append(row)
        
    # all_data_en_pred = pd.DataFrame(all_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    # all_data_en_pred.to_json('/Users/mkhan149/Downloads/Experiments/all_data_en_pred.json', orient='records', lines=True)
    # all_data_en_pred.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_en_pred.csv', header = True)
    all_data_en_pred_non_filtered = pd.DataFrame(all_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    all_data_en_pred_non_filtered.to_json('/Users/mkhan149/Downloads/Experiments/all_data_en_pred_non_filtered.json', orient='records', lines=True)
    all_data_en_pred_non_filtered.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_en_pred_non_filtered.csv', header = True)
    
    #print(data)
    # return all_data_en_pred
    return all_data_en_pred_non_filtered

def preprocess_data_en_pred_filtered(input_file):
    column_name = ["userID", "Semester", "Semester_details", "course_name","credits", "letter_grade", "numerical_grade"]
    df = pd.read_csv(input_file, header = None, names = column_name, engine= 'python')
    df2 = df[["userID", "course_name", "Semester", "letter_grade"]]
    #dropped_courses = df2[df2['letter_grade'] == 'DR']
    #print(p_graded_courses)
    df2 = df2[df2.letter_grade != "DR"]
    df2 = df2[df2.letter_grade != "F"]
    df2 = df2[df2.letter_grade != "F0"]
    df2 = df2[df2.letter_grade != "D"]
    df2 = df2.reset_index()
    #print(df2) 
    df2["itemID"] = df2["course_name"].transform(lambda x: course_name_update(x))
    #df2 = df2.sort_values(by= "Semester")
    #print(df2)
    frequency_of_courses = {}
    for item in df2["itemID"]:
        if item not in frequency_of_courses:
            frequency_of_courses[item] = 1
        else:
            frequency_of_courses[item] += 1
    list_of_frequency = []
    for item in df2["itemID"]:
        list_of_frequency.append(frequency_of_courses[item])
    df2["frequency_of_items"] = list_of_frequency
    df2 = df2[df2.frequency_of_items > 9]
    df2 = df2.reset_index()
    data = df2[["userID", "itemID", "Semester"]]
    data = data.sort_values(by= "Semester")
    data["timestamp"]=data["Semester"]
    data = data[["userID", "itemID", "timestamp"]]

    baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 
    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    semester_info = {}
    #data = df[["userID", "itemID", "timestamp"]]
    userIDs = data.userID.values
    index = 0
    for user in userIDs:
        ts = data['timestamp'][index]
        if user not in semester_info:
            semester_info[user] = ts
        if user in semester_info:
            semester_info[user] = ts
        index +=1

    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    all_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info[user], baskets.iloc[index]['timestamps']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        all_data.append(row)
        
    # all_data_en_pred = pd.DataFrame(all_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    # all_data_en_pred.to_json('/Users/mkhan149/Downloads/Experiments/all_data_en_pred.json', orient='records', lines=True)
    # all_data_en_pred.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_en_pred.csv', header = True)
    all_data_en_pred_filtered = pd.DataFrame(all_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    all_data_en_pred_filtered.to_json('/Users/mkhan149/Downloads/Experiments/all_data_en_pred_filtered.json', orient='records', lines=True)
    all_data_en_pred_filtered.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_en_pred_filtered.csv', header = True)
    
    #print(data)
    # return all_data_en_pred
    return all_data_en_pred_filtered



def calculate_term_dict_2(term_dict, semester, basket):
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

def calculate_avg_n_actual_courses_2(input_data):
    data = input_data
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict_2(term_dict_all, semester, basket)
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
    
    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if item not in frequency_of_courses:
                    frequency_of_courses[item] = 1
                else:
                    frequency_of_courses[item] += 1
    return term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms

def another_filtering_all(input_data): # keeping courses if course_number_terms[course]>3
    data = input_data
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict_2(term_dict_all, semester, basket)
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

    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if item not in frequency_of_courses:
                    frequency_of_courses[item] = 1
                else:
                    frequency_of_courses[item] += 1

    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #grades = data['grades'][x]
        #index1 =0 
        baskets2= []
        ts2 = []
        # grades2 = []
        for x1 in range(len(baskets)):
            basket = baskets[x1]
            basket1 = []
            grade1 = []
            index3 = 0
            #grade = grades[x1]
            for course in basket:
                if course_number_terms[course]>3:
                    basket1.append(course)
                    #grade1.append(grade[index3])
                index3+=1
            if len(basket1)>0:
                baskets2.append(basket1)
                ts2.append(ts[x1])
                #grades2.append(grade1)

        data['baskets'][x]= baskets2
        data['timestamps'][x] = ts2
        data['num_baskets'][x] = len(baskets2)
        data['last_semester'][x]= ts2[-1]
        #data['grades'][x]= grades2
    
    users = data.userID.values
    all_data = []
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            all_data.append(row)
    
    data_set_filtered_all = pd.DataFrame(all_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    data_set_filtered_all.to_json('/Users/mkhan149/Downloads/Experiments/all_data_en_pred_filtered.json', orient='records', lines=True)
    

    return term_dict_all, data_set_filtered_all


def calculate_avg_n_actual_courses(input_data):
    data = input_data
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict_2(term_dict_all, semester, basket)
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
    
    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if item not in frequency_of_courses:
                    frequency_of_courses[item] = 1
                else:
                    frequency_of_courses[item] += 1

    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        grades = data['grades'][x]
        #index1 =0 
        baskets2= []
        ts2 = []
        grades2 = []
        for x1 in range(len(baskets)):
            basket = baskets[x1]
            basket1 = []
            grade1 = []
            index3 = 0
            grade = grades[x1]
            for course in basket:
                if course_number_terms[course]>3:
                    basket1.append(course)
                    grade1.append(grade[index3])
                index3+=1
            if len(basket1)>0:
                baskets2.append(basket1)
                ts2.append(ts[x1])
                grades2.append(grade1)

        data['baskets'][x]= baskets2
        data['timestamps'][x] = ts2
        data['num_baskets'][x] = len(baskets2)
        data['last_semester'][x]= ts2[-1]
        data['grades'][x]= grades2
    
    users = data.userID.values
    train_data = []
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps'], data.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_data.append(row)
    
    train_set_all = pd.DataFrame(train_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    train_set_all.to_json('/Users/mkhan149/Downloads/Experiments/train_data_all.json', orient='records', lines=True)
    
    frequency_of_courses_in_last_term = {}
    for baskets in train_set_all["baskets"]:
        basket = baskets[-1]
        for item in basket:
            if item not in frequency_of_courses_in_last_term:
                frequency_of_courses_in_last_term[item] = 1
            else:
                frequency_of_courses_in_last_term[item] += 1
    #row = [item1, count_course_avg[item1], course_sd_main[item1], course_number_terms[item1]]
    item_dict = {}
    for baskets in train_set_all["baskets"]:
        for basket in baskets:
            for course in basket:
                if course not in item_dict:
                    item_dict[course]= len(item_dict)
    item_list = list(item_dict.keys())
    course_info = []
    for item in item_list:
        if item in frequency_of_courses_in_last_term:
            row = [item, course_number_terms[item], frequency_of_courses[item], count_course_avg[item], course_sd_main[item], frequency_of_courses_in_last_term[item]]
        else:
            row = [item, course_number_terms[item], frequency_of_courses[item], count_course_avg[item], course_sd_main[item], 0]
        course_info.append(row)
    course_info = pd.DataFrame(course_info, columns=['Course_ID', 'n_offered_in_past', 'n_students_took_in_past', 'avg_n_students_per_offering', 'st_dev_students_per_offering', 'n_students_in_last_offering'])
    course_info.to_csv('/Users/mkhan149/Downloads/Experiments/course_information.csv')

    return train_set_all, course_number_terms, frequency_of_courses, count_course_avg, course_sd_main, frequency_of_courses_in_last_term, course_info


# preprocess data without summer courses
def preprocess_data_without_summer(input_file):
    column_name = ["userID", "Semester", "Semester_details", "course_name","credits", "letter_grade", "numerical_grade"]
    df = pd.read_csv(input_file, header = None, names = column_name, engine= 'python')
    df2 = df[["userID", "course_name", "Semester", "letter_grade"]]
    #dropped_courses = df2[df2['letter_grade'] == 'DR']
    # print(p_graded_courses)
    df2 = df2[df2.letter_grade != "DR"]
    df2 = df2[df2.letter_grade != "F"]
    df2 = df2[df2.letter_grade != "F0"]
    df2 = df2[df2.letter_grade != "D"]
    df2 = df2.reset_index()
    #print(df2) 
    df2["itemID"] = df2["course_name"].transform(lambda x: course_name_update(x))
    #df2 = df2.sort_values(by= "Semester")
    #print(df2)
    frequency_of_courses = {}
    for item in df2["itemID"]:
        if item not in frequency_of_courses:
            frequency_of_courses[item] = 1
        else:
            frequency_of_courses[item] += 1
    list_of_frequency = []
    for item in df2["itemID"]:
        list_of_frequency.append(frequency_of_courses[item])
    df2["frequency_of_items"] = list_of_frequency
    df2 = df2[df2.frequency_of_items > 2]
    #df2 = df2.reset_index()
    df2 = df2[df2.Semester %5 != 0]
    df2 = df2.reset_index()
    data_without_summer = df2[["userID", "itemID", "Semester"]]
    data_without_summer = data_without_summer.sort_values(by= "Semester")
    data_without_summer.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_without_summer.csv', header = True)
    #print(data)
    return data_without_summer

# preprocess data merging summer courses with fall courses
def preprocess_data_merging_summer_fall(input_file):
    column_name = ["userID", "Semester", "Semester_details", "course_name","credits", "letter_grade", "numerical_grade"]
    df = pd.read_csv(input_file, header = None, names = column_name, engine= 'python')
    df2 = df[["userID", "course_name", "Semester", "letter_grade"]]
    #dropped_courses = df2[df2['letter_grade'] == 'DR']
    # print(p_graded_courses)
    df2 = df2[df2.letter_grade != "DR"]
    df2 = df2[df2.letter_grade != "F"]
    df2 = df2[df2.letter_grade != "F0"]
    df2 = df2[df2.letter_grade != "D"]
    df2 = df2.reset_index()
    #print(df2) 
    df2["itemID"] = df2["course_name"].transform(lambda x: course_name_update(x))
    #df2 = df2.sort_values(by= "Semester")
    #print(df2)
    frequency_of_courses = {}
    for item in df2["itemID"]:
        if item not in frequency_of_courses:
            frequency_of_courses[item] = 1
        else:
            frequency_of_courses[item] += 1
    list_of_frequency = []
    for item in df2["itemID"]:
        list_of_frequency.append(frequency_of_courses[item])
    df2["frequency_of_items"] = list_of_frequency
    df2 = df2[df2.frequency_of_items > 2]
    #df2 = df2.reset_index()
    #df2 = df2[df2.Semester %5 != 0]
    df2 = df2.reset_index()
    data = df2[["userID", "itemID", "Semester"]]
    data = data.sort_values(by= "Semester")
    #merge summer with fall
    index1 =0
    for ts in data["Semester"]:
        if ts%5 ==0:
            ts += 3
        data["Semester"][index1] = ts
        index1 +=1 

    data.to_csv('/Users/mkhan149/Downloads/Experiments/all_data_merging_summer_fall.csv', header = True)
    #print(data)
    return data
def split_data_new(input_data):
    #data = pd.read_csv(input_file, names=['userID', 'itemID', 'Semester'])
    #data = input_data
    #train_set_all, test_set = train_test_split(data, test_size=0.1, random_state = 0)
    # train_set_all, test_set = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    # #train_set, valid_set = train_test_split(train_set_all, test_size=0.1, random_state = 0)
    # train_set, valid_set = train_set_all[: int(len(train_set_all) * 0.8)], train_set_all[int(len(train_set_all) * 0.8):]

    df = input_data
    #sequence of semesters
    df["timestamp"]=df["Semester"]
    data = df[["userID", "itemID", "timestamp"]]
    #sorting
    #data = data.sort_values(by="timestamp")
    #data["userID"] = data["userID"].fillna(-1).astype('int32')
    #data=data[data["userID"]!=-1].reset_index(drop=True)
    # test data
    df2 = data[data.timestamp == 1221] #last semester
    #print(df2)
    users_test = df2.userID.values
    #print(users_test)
    df_test1 = []
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test:
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test1.append(row)
    
    
    df_test = pd.DataFrame(df_test1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']

    users = baskets.userID.values
    test_data = []
    semester_info = 1221
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=3:
        row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        test_data.append(row)
    
    df3 = data[data.timestamp == 1218] #2nd last semester
    users_test2 = df3.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test2 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test2 and data["timestamp"][x] != 1221: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test2.append(row)
    
    df_test3 = pd.DataFrame(df_test2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1218
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=3:
        row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        test_data.append(row)

    df9 = data[data.timestamp == 1215] #3rd last semester
    users_test9 = df9.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test9 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test9 and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test9.append(row)
    
    df_test9 = pd.DataFrame(df_test9, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test9.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1215
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=3:
        row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        test_data.append(row)
        
    test_set_all = pd.DataFrame(test_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_set_all.to_json('/Users/mkhan149/Downloads/Experiments/test_data_all_non_filtered.json', orient='records', lines=True)

    # validation data
    df5 = data[data.timestamp == 1211] #4th last semester
    #print(df2)
    users_valid = df5.userID.values
    #print(users_test)
    df_valid1 = []
    semester_info = 1211
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1215: # excluding last three semester's data (test data)
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid1.append(row)
    
    
    df_valid = pd.DataFrame(df_valid1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']   
    users = baskets.userID.values
    valid_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df6 = data[data.timestamp == 1208] #5th last semester
    users_valid2 = df6.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid2 = []
    semester_info = 1208
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid2 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 4 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid2.append(row)
    
    df_valid3 = pd.DataFrame(df_valid2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df12 = data[data.timestamp == 1205] #6th last semester
    users_valid12 = df12.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid12 = []
    semester_info = 1205
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid12 and data["timestamp"][x] != 1208 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 5 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid12.append(row)
    
    df_valid12 = pd.DataFrame(df_valid12, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid12.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
        
    valid_set_all = pd.DataFrame(valid_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    valid_set_all.to_json('/Users/mkhan149/Downloads/Experiments/valid_data_all_non_filtered.json', orient='records', lines=True)

    df7 = data[data.timestamp != 1221] 
    df7 = df7[df7.timestamp != 1218]
    df7 = df7[df7.timestamp != 1215]
    df7 = df7[df7.timestamp != 1211] 
    df7 = df7[df7.timestamp != 1208]
    df7 = df7[df7.timestamp != 1205]
    df7 = df7.reset_index()

    baskets = df7.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    semester_info = {}
    #data = df[["userID", "itemID", "timestamp"]]
    userIDs = data.userID.values
    index = 0
    for user in userIDs:
        ts = data['timestamp'][index]
        if user not in semester_info and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        if user in semester_info and ts>semester_info[user] and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        index +=1

    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    train_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info[user], baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_data.append(row)
        
    train_set_all = pd.DataFrame(train_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    train_set_all.to_json('/Users/mkhan149/Downloads/Experiments/train_data_all_non_filtered.json', orient='records', lines=True)

    return train_set_all, valid_set_all, test_set_all

def split_data(input_data):
    #data = pd.read_csv(input_file, names=['userID', 'itemID', 'Semester'])
    #data = input_data
    #train_set_all, test_set = train_test_split(data, test_size=0.1, random_state = 0)
    # train_set_all, test_set = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    # #train_set, valid_set = train_test_split(train_set_all, test_size=0.1, random_state = 0)
    # train_set, valid_set = train_set_all[: int(len(train_set_all) * 0.8)], train_set_all[int(len(train_set_all) * 0.8):]

    df = input_data
    #sequence of semesters
    df["timestamp"]=df["Semester"]
    data = df[["userID", "itemID", "timestamp", "grades"]]
    #sorting
    #data = data.sort_values(by="timestamp")
    #data["userID"] = data["userID"].fillna(-1).astype('int32')
    #data=data[data["userID"]!=-1].reset_index(drop=True)
    # test data
    df2 = data[data.timestamp == 1221] #last semester
    #print(df2)
    users_test = df2.userID.values
    #print(users_test)
    df_test1 = []
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test:
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test1.append(row)
    
    
    df_test = pd.DataFrame(df_test1, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_test.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index() 
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 
    
    grades = df_test.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']

    users = baskets.userID.values
    test_data = []
    semester_info = 1221
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
    
    df3 = data[data.timestamp == 1218] #2nd last semester
    users_test2 = df3.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test2 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test2 and data["timestamp"][x] != 1221: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test2.append(row)
    
    df_test3 = pd.DataFrame(df_test2, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_test3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_test3.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1218
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)

    df9 = data[data.timestamp == 1215] #3rd last semester
    users_test9 = df9.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test9 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test9 and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test9.append(row)
    
    df_test9 = pd.DataFrame(df_test9, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_test9.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_test9.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1215
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
        
    test_set_all = pd.DataFrame(test_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    test_set_all.to_json('/Users/mkhan149/Downloads/Experiments/test_data_all_CR.json', orient='records', lines=True)

    # validation data
    df5 = data[data.timestamp == 1211] #4th last semester
    #print(df2)
    users_valid = df5.userID.values
    #print(users_test)
    df_valid1 = []
    semester_info = 1211
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1215: # excluding last three semester's data (test data)
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid1.append(row)
    
    
    df_valid = pd.DataFrame(df_valid1, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_valid.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_valid.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']   
    users = baskets.userID.values
    valid_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df6 = data[data.timestamp == 1208] #5th last semester
    users_valid2 = df6.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid2 = []
    semester_info = 1208
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid2 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 4 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid2.append(row)
    
    df_valid3 = pd.DataFrame(df_valid2, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_valid3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_valid3.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df12 = data[data.timestamp == 1205] #6th last semester
    users_valid12 = df12.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid12 = []
    semester_info = 1205
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid12 and data["timestamp"][x] != 1208 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 5 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid12.append(row)
    
    df_valid12 = pd.DataFrame(df_valid12, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_valid12.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_valid12.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
        
    valid_set_all = pd.DataFrame(valid_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    valid_set_all.to_json('/Users/mkhan149/Downloads/Experiments/valid_data_all_CR.json', orient='records', lines=True)

    df7 = data[data.timestamp != 1221] 
    df7 = df7[df7.timestamp != 1218]
    df7 = df7[df7.timestamp != 1215]
    df7 = df7[df7.timestamp != 1211] 
    df7 = df7[df7.timestamp != 1208]
    df7 = df7[df7.timestamp != 1205]
    df7 = df7.reset_index()

    baskets = df7.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df7.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    semester_info = {}
    #data = df[["userID", "itemID", "timestamp"]]
    userIDs = data.userID.values
    index = 0
    for user in userIDs:
        ts = data['timestamp'][index]
        if user not in semester_info and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        if user in semester_info and ts>semester_info[user] and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        index +=1

    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    train_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info[user], baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_data.append(row)
        
    train_set_all = pd.DataFrame(train_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    train_set_all.to_json('/Users/mkhan149/Downloads/Experiments/train_data_all_CR.json', orient='records', lines=True)

    return train_set_all, valid_set_all, test_set_all

def split_data_with_more_target_baskets(input_data):
    #data = pd.read_csv(input_file, names=['userID', 'itemID', 'Semester'])
    #data = input_data
    #train_set_all, test_set = train_test_split(data, test_size=0.1, random_state = 0)
    # train_set_all, test_set = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    # #train_set, valid_set = train_test_split(train_set_all, test_size=0.1, random_state = 0)
    # train_set, valid_set = train_set_all[: int(len(train_set_all) * 0.8)], train_set_all[int(len(train_set_all) * 0.8):]

    df = input_data
    #sequence of semesters
    df["timestamp"]=df["Semester"]
    data = df[["userID", "itemID", "timestamp", "grades"]]
    #sorting
    #data = data.sort_values(by="timestamp")
    #data["userID"] = data["userID"].fillna(-1).astype('int32')
    #data=data[data["userID"]!=-1].reset_index(drop=True)
    # test data
    df2 = data[data.timestamp == 1221] #last semester
    #print(df2)
    users_test = df2.userID.values
    #print(users_test)
    df_test1 = []
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test:
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test1.append(row)
    
    
    df_test = pd.DataFrame(df_test1, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_test.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index() 
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 
    
    grades = df_test.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']

    users = baskets.userID.values
    test_data = []
    semester_info = 1221
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
    
    df3 = data[data.timestamp == 1218] #2nd last semester
    users_test2 = df3.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test2 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test2: #and data["timestamp"][x] != 1221: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test2.append(row)
    
    df_test3 = pd.DataFrame(df_test2, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_test3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps']          


    grades = df_test3.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades']

    for idx in range(len(baskets['baskets'])):
        bskts = baskets.iloc[idx]['baskets']
        ts3 = semester_all.iloc[idx]['timestamps']
        grad1 = grades.iloc[idx]['grades']
        if 1221 in list(ts3):
            prev_bskt = bskts[-2]
            bskt = bskts[-1] 
            for item1 in bskt:
                # if item1 not in prev_bskt:
                prev_bskt.append(item1)
            bskts[-2] = prev_bskt
            bskts = bskts[ :-1]

            prev_grads = grad1[-2]
            grads = grad1[-1] 
            for g1 in grads:
                #if g1 not in prev_grads:
                prev_grads.append(g1)
            grad1[-2] = prev_grads

            grad1 = grad1[ :-1]
            ts3 = ts3[ :-1]
            baskets['baskets'][idx] = bskts
            semester_all['timestamps'][idx] = ts3  
            grades['grades'][idx] = grad1    

    baskets['grades'] = grades['grades']
    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1218
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)

    df9 = data[data.timestamp == 1215] #3rd last semester
    users_test9 = df9.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test9 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test9: #and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test9.append(row)
    
    df_test9 = pd.DataFrame(df_test9, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_test9.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_test9.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']
    for idx in range(len(baskets['baskets'])):
        bskts = baskets.iloc[idx]['baskets']
        ts3 = semester_all.iloc[idx]['timestamps']
        grad1 = grades.iloc[idx]['grades']
        if 1221 in ts3:
            if 1218 in ts3:
                prev_bskt = bskts[-2]
                bskt = bskts[-1] 
                for item1 in bskt:
                    # if item1 not in prev_bskt:
                    prev_bskt.append(item1)
                bskts[-2] = prev_bskt
                bskts = bskts[ :-1]

                prev_grads = grad1[-2]
                grads = grad1[-1] 
                for g1 in grads:
                    #if g1 not in prev_grads:
                    prev_grads.append(g1)
                grad1[-2] = prev_grads
                grad1 = grad1[ :-1]
                ts3 = ts3[ :-1]
                baskets['baskets'][idx] = bskts
                semester_all['timestamps'][idx] = ts3  
                grades['grades'][idx] = grad1   
                if 1215 in ts3:
                    bskts = baskets.iloc[idx]['baskets']
                    ts3 = semester_all.iloc[idx]['timestamps']
                    grad1 = grades.iloc[idx]['grades']
                    prev_bskt = bskts[-2]
                    bskt = bskts[-1] 
                    for item1 in bskt:
                        # if item1 not in prev_bskt:
                        prev_bskt.append(item1)
                    bskts[-2] = prev_bskt
                    bskts = bskts[ :-1]

                    prev_grads = grad1[-2]
                    grads = grad1[-1] 
                    for g1 in grads:
                        #if g1 not in prev_grads:
                        prev_grads.append(g1)
                    grad1[-2] = prev_grads
                    grad1 = grad1[ :-1]
                    ts3 = ts3[ :-1]
                    baskets['baskets'][idx] = bskts
                    semester_all['timestamps'][idx] = ts3  
                    grades['grades'][idx] = grad1    

            else:
                if 1215 in ts3:
                    prev_bskt = bskts[-2]
                    bskt = bskts[-1] 
                    for item1 in bskt:
                        # if item1 not in prev_bskt:
                        prev_bskt.append(item1)
                    bskts[-2] = prev_bskt
                    bskts = bskts[ :-1]

                    prev_grads = grad1[-2]
                    grads = grad1[-1] 
                    for g1 in grads:
                        #if g1 not in prev_grads:
                        prev_grads.append(g1)
                    grad1[-2] = prev_grads
                    grad1 = grad1[ :-1]
                    ts3 = ts3[ :-1]
                    baskets['baskets'][idx] = bskts
                    semester_all['timestamps'][idx] = ts3  
                    grades['grades'][idx] = grad1   
        if 1218 in ts3 and 1221 not in ts3:
            if 1215 in ts3:
                    prev_bskt = bskts[-2]
                    bskt = bskts[-1] 
                    for item1 in bskt:
                        # if item1 not in prev_bskt:
                        prev_bskt.append(item1)
                    bskts[-2] = prev_bskt
                    bskts = bskts[ :-1]

                    prev_grads = grad1[-2]
                    grads = grad1[-1] 
                    for g1 in grads:
                        #if g1 not in prev_grads:
                        prev_grads.append(g1)
                    grad1[-2] = prev_grads
                    grad1 = grad1[ :-1]
                    ts3 = ts3[ :-1]
                    baskets['baskets'][idx] = bskts
                    semester_all['timestamps'][idx] = ts3  
                    grades['grades'][idx] = grad1   


    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1215
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
        
    test_set_all = pd.DataFrame(test_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    test_set_all.to_json('/Users/mkhan149/Downloads/Experiments/test_data_all_CR_with_more_target_baskets.json', orient='records', lines=True)

    # validation data
    df5 = data[data.timestamp == 1211] #4th last semester
    #print(df2)
    users_valid = df5.userID.values
    #print(users_test)
    df_valid1 = []
    semester_info = 1211
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1215: # excluding last three semester's data (test data)
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid1.append(row)
    
    
    df_valid = pd.DataFrame(df_valid1, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_valid.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_valid.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']   
    users = baskets.userID.values
    valid_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df6 = data[data.timestamp == 1208] #5th last semester
    users_valid2 = df6.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid2 = []
    semester_info = 1208
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid2 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 4 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid2.append(row)
    
    df_valid3 = pd.DataFrame(df_valid2, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_valid3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_valid3.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df12 = data[data.timestamp == 1205] #6th last semester
    users_valid12 = df12.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid12 = []
    semester_info = 1205
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid12 and data["timestamp"][x] != 1208 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 5 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            gr = data['grades'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts, gr]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid12.append(row)
    
    df_valid12 = pd.DataFrame(df_valid12, columns=['userID', 'itemID', 'timestamp', 'grades'])

    baskets = df_valid12.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df_valid12.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
        
    valid_set_all = pd.DataFrame(valid_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    valid_set_all.to_json('/Users/mkhan149/Downloads/Experiments/valid_data_all_CR_with_more_target_baskets.json', orient='records', lines=True)

    df7 = data[data.timestamp != 1221] 
    df7 = df7[df7.timestamp != 1218]
    df7 = df7[df7.timestamp != 1215]
    df7 = df7[df7.timestamp != 1211] 
    df7 = df7[df7.timestamp != 1208]
    df7 = df7[df7.timestamp != 1205]
    df7 = df7.reset_index()

    baskets = df7.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    grades = df7.groupby(['userID', 'timestamp'])['grades'].apply(list).reset_index()
    grades = grades.groupby(['userID'])['grades'].apply(list).reset_index() 
    grades.columns = ['userID', 'grades'] 
    baskets['grades'] = grades['grades']

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    semester_info = {}
    #data = df[["userID", "itemID", "timestamp"]]
    userIDs = data.userID.values
    index = 0
    for user in userIDs:
        ts = data['timestamp'][index]
        if user not in semester_info and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        if user in semester_info and ts>semester_info[user] and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        index +=1

    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    train_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info[user], baskets.iloc[index]['timestamps'], baskets.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_data.append(row)
        
    train_set_all = pd.DataFrame(train_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    train_set_all.to_json('/Users/mkhan149/Downloads/Experiments/train_data_all_CR_with_more_target_baskets.json', orient='records', lines=True)

    return train_set_all, valid_set_all, test_set_all

def calculate_term_dict_test(test_data):
    index = 0
    target_baskets= []
    target_semesters = []
    for x in range(len(test_data)):
         b = test_data.iloc[index]['baskets'][-1]
         ts = test_data.iloc[index]['timestamps'][-1]
         target_semesters.append(ts)
         target_baskets.append(b)
         index+= 1
    term_dict = {}
    for x in range(len(target_baskets)):
        # for item in basket:
        basket = target_baskets[x]
        target_semester = target_semesters[x]
        term_dict = calculate_term_dict_2(term_dict, target_semester, basket)
    
    return term_dict


def split_data_without_summer(input_data):
    #data = pd.read_csv(input_file, names=['userID', 'itemID', 'Semester'])
    #data = input_data
    #train_set_all, test_set = train_test_split(data, test_size=0.1, random_state = 0)
    # train_set_all, test_set = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    # #train_set, valid_set = train_test_split(train_set_all, test_size=0.1, random_state = 0)
    # train_set, valid_set = train_set_all[: int(len(train_set_all) * 0.8)], train_set_all[int(len(train_set_all) * 0.8):]

    df = input_data
    #sequence of semesters
    df["timestamp"]=df["Semester"]
    data = df[["userID", "itemID", "timestamp"]]
    #sorting
    #data = data.sort_values(by="timestamp")
    #data["userID"] = data["userID"].fillna(-1).astype('int32')
    #data=data[data["userID"]!=-1].reset_index(drop=True)
    # test data
    df2 = data[data.timestamp == 1221] #last semester
    #print(df2)
    users_test = df2.userID.values
    #print(users_test)
    df_test1 = []
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test:
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test1.append(row)
    
    
    df_test = pd.DataFrame(df_test1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']

    users = baskets.userID.values
    test_data = []
    semester_info = 1221
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
    
    df3 = data[data.timestamp == 1218] #2nd last semester
    users_test2 = df3.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test2 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test2 and data["timestamp"][x] != 1221: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test2.append(row)
    
    df_test3 = pd.DataFrame(df_test2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1218
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)

    # df9 = data[data.timestamp == 1215] #3rd last semester
    # users_test9 = df9.userID.values
    # #print(users_test)
    # #users_all = data.userID.values
    # df_test9 = []
    # for x in range(len(data)):
    #     #df_test = data[data["userID"]==user]
    #     user = data["userID"][x]
    #     if user in users_test9 and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218: #excluding the last semester's data
    #         uID = data['userID'][x]
    #         iID = data['itemID'][x]
    #         ts = data['timestamp'][x]
    #         #b = baskets.iloc[index]['baskets'][0:]
    #         row = [uID, iID, ts]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         df_test9.append(row)
    
    # df_test9 = pd.DataFrame(df_test9, columns=['userID', 'itemID', 'timestamp'])

    # baskets = df_test9.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    # baskets.columns = ['userID', 'baskets']
    # semester_all.columns = ['userID', 'timestamps'] 

    # baskets['num_baskets'] = baskets.baskets.apply(len)
    # baskets['timestamps'] = semester_all['timestamps']
    # users = baskets.userID.values
    # semester_info = 1215
    # for user in users:
    #     index = baskets[baskets['userID'] == user].index.values[0]
    #     b = baskets.iloc[index]['baskets'][0:]
    #     #b = baskets.iloc[index]['baskets'][0:]
    #     if baskets.iloc[index]['num_baskets']>=3:
    #         row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         test_data.append(row)
        
    test_set_all_without_summer = pd.DataFrame(test_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_set_all_without_summer.to_json('/Users/mkhan149/Downloads/Experiments/test_data_all_without_summer.json', orient='records', lines=True)

    # validation data
    df5 = data[data.timestamp == 1211] #4th last semester
    #print(df2)
    users_valid = df5.userID.values
    #print(users_test)
    df_valid1 = []
    semester_info = 1211
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1215: # excluding last three semester's data (test data)
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid1.append(row)
    
    
    df_valid = pd.DataFrame(df_valid1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']   
    users = baskets.userID.values
    valid_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df6 = data[data.timestamp == 1208] #5th last semester
    users_valid2 = df6.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid2 = []
    semester_info = 1208
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid2 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 4 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid2.append(row)
    
    df_valid3 = pd.DataFrame(df_valid2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    # df12 = data[data.timestamp == 1205] #6th last semester
    # users_valid12 = df12.userID.values
    # #print(users_test)
    # #users_all = data.userID.values
    # df_valid12 = []
    # semester_info = 1205
    # for x in range(len(data)):
    #     #df_test = data[data["userID"]==user]
    #     user = data["userID"][x]
    #     if user in users_valid12 and data["timestamp"][x] != 1208 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 5 semester's data
    #         uID = data['userID'][x]
    #         iID = data['itemID'][x]
    #         ts = data['timestamp'][x]
    #         #b = baskets.iloc[index]['baskets'][0:]
    #         row = [uID, iID, ts]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         df_valid12.append(row)
    
    # df_valid12 = pd.DataFrame(df_valid12, columns=['userID', 'itemID', 'timestamp'])

    # baskets = df_valid12.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    # baskets.columns = ['userID', 'baskets']
    # semester_all.columns = ['userID', 'timestamps'] 

    # baskets['num_baskets'] = baskets.baskets.apply(len)
    # baskets['timestamps'] = semester_all['timestamps']
    # users = baskets.userID.values
    # for user in users:
    #     index = baskets[baskets['userID'] == user].index.values[0]
    #     b = baskets.iloc[index]['baskets'][0:]
    #     #b = baskets.iloc[index]['baskets'][0:]
    #     if baskets.iloc[index]['num_baskets']>=3:
    #         row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         valid_data.append(row)
        
    valid_set_all_without_summer = pd.DataFrame(valid_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    valid_set_all_without_summer.to_json('/Users/mkhan149/Downloads/Experiments/valid_data_all_without_summer.json', orient='records', lines=True)

    df7 = data[data.timestamp != 1221] 
    df7 = df7[df7.timestamp != 1218]
    #df7 = df7[df7.timestamp != 1215]
    df7 = df7[df7.timestamp != 1211] 
    df7 = df7[df7.timestamp != 1208]
    #df7 = df7[df7.timestamp != 1205]
    df7 = df7.reset_index()

    baskets = df7.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    semester_info = {}
    #data = df[["userID", "itemID", "timestamp"]]
    userIDs = data.userID.values
    index = 0
    for user in userIDs:
        ts = data['timestamp'][index]
        if user not in semester_info and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        if user in semester_info and ts>semester_info[user] and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        index +=1

    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    train_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info[user], baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_data.append(row)
        
    train_set_all_without_summer = pd.DataFrame(train_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    train_set_all_without_summer.to_json('/Users/mkhan149/Downloads/Experiments/train_data_all_without_summer.json', orient='records', lines=True)

    return train_set_all_without_summer, valid_set_all_without_summer, test_set_all_without_summer



#split data merging summer with fall
def split_data_merging_summer_fall(input_data):
    #data = pd.read_csv(input_file, names=['userID', 'itemID', 'Semester'])
    #data = input_data
    #train_set_all, test_set = train_test_split(data, test_size=0.1, random_state = 0)
    # train_set_all, test_set = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    # #train_set, valid_set = train_test_split(train_set_all, test_size=0.1, random_state = 0)
    # train_set, valid_set = train_set_all[: int(len(train_set_all) * 0.8)], train_set_all[int(len(train_set_all) * 0.8):]

    df = input_data
    #sequence of semesters
    df["timestamp"]=df["Semester"]
    data = df[["userID", "itemID", "timestamp"]]
    #sorting
    #data = data.sort_values(by="timestamp")
    #data["userID"] = data["userID"].fillna(-1).astype('int32')
    #data=data[data["userID"]!=-1].reset_index(drop=True)
    # test data
    df2 = data[data.timestamp == 1221] #last semester
    #print(df2)
    users_test = df2.userID.values
    #print(users_test)
    df_test1 = []
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test:
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test1.append(row)
    
    
    df_test = pd.DataFrame(df_test1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']

    users = baskets.userID.values
    test_data = []
    semester_info = 1221
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
    
    df3 = data[data.timestamp == 1218] #2nd last semester
    users_test2 = df3.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test2 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test2 and data["timestamp"][x] != 1221: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test2.append(row)
    
    df_test3 = pd.DataFrame(df_test2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1218
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)

    # df9 = data[data.timestamp == 1215] #3rd last semester
    # users_test9 = df9.userID.values
    # #print(users_test)
    # #users_all = data.userID.values
    # df_test9 = []
    # for x in range(len(data)):
    #     #df_test = data[data["userID"]==user]
    #     user = data["userID"][x]
    #     if user in users_test9 and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218: #excluding the last semester's data
    #         uID = data['userID'][x]
    #         iID = data['itemID'][x]
    #         ts = data['timestamp'][x]
    #         #b = baskets.iloc[index]['baskets'][0:]
    #         row = [uID, iID, ts]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         df_test9.append(row)
    
    # df_test9 = pd.DataFrame(df_test9, columns=['userID', 'itemID', 'timestamp'])

    # baskets = df_test9.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    # baskets.columns = ['userID', 'baskets']
    # semester_all.columns = ['userID', 'timestamps'] 

    # baskets['num_baskets'] = baskets.baskets.apply(len)
    # baskets['timestamps'] = semester_all['timestamps']
    # users = baskets.userID.values
    # semester_info = 1215
    # for user in users:
    #     index = baskets[baskets['userID'] == user].index.values[0]
    #     b = baskets.iloc[index]['baskets'][0:]
    #     #b = baskets.iloc[index]['baskets'][0:]
    #     if baskets.iloc[index]['num_baskets']>=3:
    #         row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         test_data.append(row)
        
    test_set_all_merging_summer_fall = pd.DataFrame(test_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_set_all_merging_summer_fall.to_json('/Users/mkhan149/Downloads/Experiments/test_data_all_merging_summer_fall.json', orient='records', lines=True)

    # validation data
    df5 = data[data.timestamp == 1211] #4th last semester
    #print(df2)
    users_valid = df5.userID.values
    #print(users_test)
    df_valid1 = []
    semester_info = 1211
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1215: # excluding last three semester's data (test data)
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid1.append(row)
    
    
    df_valid = pd.DataFrame(df_valid1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']   
    users = baskets.userID.values
    valid_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df6 = data[data.timestamp == 1208] #5th last semester
    users_valid2 = df6.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid2 = []
    semester_info = 1208
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid2 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 4 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid2.append(row)
    
    df_valid3 = pd.DataFrame(df_valid2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    # df12 = data[data.timestamp == 1205] #6th last semester
    # users_valid12 = df12.userID.values
    # #print(users_test)
    # #users_all = data.userID.values
    # df_valid12 = []
    # semester_info = 1205
    # for x in range(len(data)):
    #     #df_test = data[data["userID"]==user]
    #     user = data["userID"][x]
    #     if user in users_valid12 and data["timestamp"][x] != 1208 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 5 semester's data
    #         uID = data['userID'][x]
    #         iID = data['itemID'][x]
    #         ts = data['timestamp'][x]
    #         #b = baskets.iloc[index]['baskets'][0:]
    #         row = [uID, iID, ts]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         df_valid12.append(row)
    
    # df_valid12 = pd.DataFrame(df_valid12, columns=['userID', 'itemID', 'timestamp'])

    # baskets = df_valid12.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    # baskets.columns = ['userID', 'baskets']
    # semester_all.columns = ['userID', 'timestamps'] 

    # baskets['num_baskets'] = baskets.baskets.apply(len)
    # baskets['timestamps'] = semester_all['timestamps']
    # users = baskets.userID.values
    # for user in users:
    #     index = baskets[baskets['userID'] == user].index.values[0]
    #     b = baskets.iloc[index]['baskets'][0:]
    #     #b = baskets.iloc[index]['baskets'][0:]
    #     if baskets.iloc[index]['num_baskets']>=3:
    #         row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
    #         #row = [user, b, baskets.iloc[index]['num_baskets']]
    #         valid_data.append(row)
        
    valid_set_all_merging_summer_fall = pd.DataFrame(valid_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    valid_set_all_merging_summer_fall.to_json('/Users/mkhan149/Downloads/Experiments/valid_data_all_merging_summer_fall.json', orient='records', lines=True)

    df7 = data[data.timestamp != 1221] 
    df7 = df7[df7.timestamp != 1218]
    #df7 = df7[df7.timestamp != 1215]
    df7 = df7[df7.timestamp != 1211] 
    df7 = df7[df7.timestamp != 1208]
    #df7 = df7[df7.timestamp != 1205]
    df7 = df7.reset_index()

    baskets = df7.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    semester_info = {}
    #data = df[["userID", "itemID", "timestamp"]]
    userIDs = data.userID.values
    index = 0
    for user in userIDs:
        ts = data['timestamp'][index]
        if user not in semester_info and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        if user in semester_info and ts>semester_info[user] and ts!= 1221 and ts!= 1218 and ts!= 1215 and ts!= 1211 and ts!= 1208 and ts!= 1205:
            semester_info[user] = ts
        index +=1

    baskets['num_baskets'] = baskets.baskets.apply(len)
    users = baskets.userID.values
    train_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info[user], baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_data.append(row)
        
    train_set_all_merging_summer_fall = pd.DataFrame(train_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    train_set_all_merging_summer_fall.to_json('/Users/mkhan149/Downloads/Experiments/train_data_all_merging_summer_fall.json', orient='records', lines=True)

    return train_set_all_merging_summer_fall, valid_set_all_merging_summer_fall, test_set_all_merging_summer_fall



#unique data in validation and test set (no augmentation of sequence for one user)
def split_data2(input_data):
    #data = pd.read_csv(input_file, names=['userID', 'itemID', 'Semester'])
    #data = input_data
    #train_set_all, test_set = train_test_split(data, test_size=0.1, random_state = 0)
    # train_set_all, test_set = data[: int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    # #train_set, valid_set = train_test_split(train_set_all, test_size=0.1, random_state = 0)
    # train_set, valid_set = train_set_all[: int(len(train_set_all) * 0.8)], train_set_all[int(len(train_set_all) * 0.8):]

    df = input_data
    #sequence of semesters
    df["timestamp"]=df["Semester"]
    data = df[["userID", "itemID", "timestamp"]]
    #sorting
    #data = data.sort_values(by="timestamp")
    #data["userID"] = data["userID"].fillna(-1).astype('int32')
    #data=data[data["userID"]!=-1].reset_index(drop=True)
    # test data
    df2 = data[data.timestamp == 1221] #last semester
    #print(df2)
    users_test = df2.userID.values
    #print(users_test)
    df_test1 = []
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test:
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test1.append(row)
    
    
    df_test = pd.DataFrame(df_test1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']

    users = baskets.userID.values
    test_data = []
    semester_info = 1221
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
    
    df3 = data[data.timestamp == 1218] #2nd last semester
    users_test2 = df3.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test2 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test2 and user not in users_test and data["timestamp"][x] != 1221: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test2.append(row)
    
    df_test3 = pd.DataFrame(df_test2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1218
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)

    df9 = data[data.timestamp == 1215] #3rd last semester
    users_test9 = df9.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_test9 = []
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_test9 and user not in users_test and user not in users_test2 and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218: #excluding the last semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_test9.append(row)
    
    df_test9 = pd.DataFrame(df_test9, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_test9.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    semester_info = 1215
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            test_data.append(row)
        
    test_set_unique = pd.DataFrame(test_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_set_unique.to_json('/Users/mkhan149/Downloads/Experiments/test_data_unique_CR.json', orient='records', lines=True)

    # validation data
    df5 = data[data.timestamp == 1211] #4th last semester
    #print(df2)
    users_valid = df5.userID.values
    #print(users_test)
    df_valid1 = []
    semester_info = 1211
    
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid and data["timestamp"][x] != 1221 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1215: # excluding last three semester's data (test data)
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid1.append(row)
    
    
    df_valid = pd.DataFrame(df_valid1, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']   
    users = baskets.userID.values
    valid_data = []
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df6 = data[data.timestamp == 1208] #5th last semester
    users_valid2 = df6.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid2 = []
    semester_info = 1208
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid2 and user not in users_valid and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 4 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid2.append(row)
    
    df_valid3 = pd.DataFrame(df_valid2, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid3.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
    
    df12 = data[data.timestamp == 1205] #6th last semester
    users_valid12 = df12.userID.values
    #print(users_test)
    #users_all = data.userID.values
    df_valid12 = []
    semester_info = 1205
    for x in range(len(data)):
        #df_test = data[data["userID"]==user]
        user = data["userID"][x]
        if user in users_valid12 and user not in users_valid and user not in users_valid2 and data["timestamp"][x] != 1208 and data["timestamp"][x] != 1211 and data["timestamp"][x] != 1215 and data["timestamp"][x] != 1218 and data["timestamp"][x] != 1221: #excluding last 5 semester's data
            uID = data['userID'][x]
            iID = data['itemID'][x]
            ts = data['timestamp'][x]
            #b = baskets.iloc[index]['baskets'][0:]
            row = [uID, iID, ts]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            df_valid12.append(row)
    
    df_valid12 = pd.DataFrame(df_valid12, columns=['userID', 'itemID', 'timestamp'])

    baskets = df_valid12.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    semester_all = baskets.groupby(['userID'])['timestamp'].apply(list).reset_index()  
    baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  
    baskets.columns = ['userID', 'baskets']
    semester_all.columns = ['userID', 'timestamps'] 

    baskets['num_baskets'] = baskets.baskets.apply(len)
    baskets['timestamps'] = semester_all['timestamps']
    users = baskets.userID.values
    for user in users:
        index = baskets[baskets['userID'] == user].index.values[0]
        b = baskets.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if baskets.iloc[index]['num_baskets']>=3:
            row = [user, b, baskets.iloc[index]['num_baskets'], semester_info, baskets.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_data.append(row)
        
    valid_set_unique = pd.DataFrame(valid_data, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    valid_set_unique.to_json('/Users/mkhan149/Downloads/Experiments/valid_data_unique_CR.json', orient='records', lines=True)

    return valid_set_unique, test_set_unique

def filter_data(input_data1, input_data2, input_data3):
    train_data = input_data1
    itemIDs = {}
    index=0
    for baskets in train_data['baskets']:
        for basket in baskets:
            for item in basket:
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)
        index +=1   
    # baskets['num_baskets'] = baskets.baskets.apply(len)
    item_list = list(itemIDs.keys())
    #augmentation of training data 
    users = train_data.userID.values
    train_valid = []
    for user in users:
        index = train_data[train_data['userID'] == user].index.values[0]
        if(train_data.iloc[index]['num_baskets']==3):
            b = train_data.iloc[index]['baskets'][0:]
            #b = baskets.iloc[index]['baskets'][0:]
            #if baskets.iloc[index]['num_baskets']>=2:
            row = [user, b, train_data.iloc[index]['num_baskets'], train_data.iloc[index]['last_semester'], train_data.iloc[index]['timestamps'],  train_data.iloc[index]['grades']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_valid.append(row)
        elif train_data.iloc[index]['num_baskets']>3:
            nb = train_data.iloc[index]['num_baskets']
            for i in range(2, nb):
                b = train_data.iloc[index]['baskets'][0:i+1]
                g = train_data.iloc[index]['grades'][0:i+1]
                #b = baskets.iloc[index]['baskets'][0:]
                #if baskets.iloc[index]['num_baskets']>=2:
                #max_len = max(max_len, data.iloc[index]['num_baskets'])
                num_baskets = len(b)
                last_sem = train_data.iloc[index]['timestamps'][i]
                timestamp_updated = train_data.iloc[index]['timestamps'][0:i+1]
                row = [user, b, num_baskets, last_sem, timestamp_updated, g]
                #row = [user, b, baskets.iloc[index]['num_baskets']]
                train_valid.append(row)
        
    train_set_augmented = pd.DataFrame(train_valid, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    train_set_augmented.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/train_sample_augmented_CR.json', orient='records', lines=True)

    valid_data = input_data2
    index=0
    for baskets in valid_data['baskets']:
        new_baskets = []
        new_grades = []
        ts = []
        tsindex = 0
        grades = valid_data['grades'][index]
        #index7 =0

        for basket in baskets:
            new_basket = []
            new_grade = []
            index5 =0
            grade_b = grades[tsindex]
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
                    new_grade.append(grade_b[index5])
                index5 += 1
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                new_grades.append(new_grade)
                ts.append(valid_data['timestamps'][index][tsindex])
            tsindex += 1
        valid_data['baskets'][index] = new_baskets
        valid_data['num_baskets'][index] = len(new_baskets)
        valid_data['timestamps'][index] = ts
        valid_data['grades'][index] = new_grades
        index +=1  
    users = valid_data.userID.values
    valid_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = valid_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if valid_data.iloc[index]['num_baskets']>=3:
            row = [user, b, valid_data.iloc[index]['num_baskets'], valid_data.iloc[index]['last_semester'], valid_data.iloc[index]['timestamps'], valid_data.iloc[index]['grades'] ]
            valid_all.append(row)
        index +=1
        #if index==30: break
    valid_set_filtered = pd.DataFrame(valid_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    valid_set_filtered.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_all_CR.json', orient='records', lines=True)

    test_data = input_data3
    # baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  #
    # baskets.columns = ['userID', 'baskets']

    #baskets['num_baskets'] = baskets.baskets.apply(len)
    index=0
    for baskets in test_data['baskets']:
        new_baskets = []
        new_grades = []
        ts = []
        tsindex = 0
        grades = test_data['grades'][index]
        for basket in baskets:
            new_basket = []
            new_grade = []
            index5 =0
            grade_b = grades[tsindex]
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
                    new_grade.append(grade_b[index5])
                index5 += 1
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                new_grades.append(new_grade)
                ts.append(test_data['timestamps'][index][tsindex])
            tsindex += 1
        test_data['baskets'][index] = new_baskets
        test_data['num_baskets'][index] = len(new_baskets)
        test_data['timestamps'][index] = ts
        test_data['grades'][index] = new_grades
        index +=1  
    users = test_data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = test_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if test_data.iloc[index]['num_baskets']>=3:
            row = [user, b, test_data.iloc[index]['num_baskets'], test_data.iloc[index]['last_semester'], test_data.iloc[index]['timestamps'], test_data.iloc[index]['grades']]
            test_all.append(row)
        index +=1
        #if index==30: break
    test_set_filtered = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps', 'grades'])
    test_set_filtered.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_all_CR.json', orient='records', lines=True)

    return train_set_augmented, valid_set_filtered, test_set_filtered


def filter_data_without_summer(input_data1, input_data2, input_data3):
    train_data = input_data1
    itemIDs = {}
    index=0
    for baskets in train_data['baskets']:
        for basket in baskets:
            for item in basket:
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)
        index +=1   
    # baskets['num_baskets'] = baskets.baskets.apply(len)
    item_list = list(itemIDs.keys())
    #augmentation of training data 
    users = train_data.userID.values
    train_valid = []
    for user in users:
        index = train_data[train_data['userID'] == user].index.values[0]
        if(train_data.iloc[index]['num_baskets']==3):
            b = train_data.iloc[index]['baskets'][0:]
            #b = baskets.iloc[index]['baskets'][0:]
            #if baskets.iloc[index]['num_baskets']>=2:
            row = [user, b, train_data.iloc[index]['num_baskets'], train_data.iloc[index]['last_semester'], train_data.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_valid.append(row)
        elif train_data.iloc[index]['num_baskets']>3:
            nb = train_data.iloc[index]['num_baskets']
            for i in range(2, nb):
                b = train_data.iloc[index]['baskets'][0:i+1]
                #b = baskets.iloc[index]['baskets'][0:]
                #if baskets.iloc[index]['num_baskets']>=2:
                #max_len = max(max_len, data.iloc[index]['num_baskets'])
                num_baskets = len(b)
                last_sem = train_data.iloc[index]['timestamps'][i]
                timestamp_updated = train_data.iloc[index]['timestamps'][0:i+1]
                row = [user, b, num_baskets, last_sem, timestamp_updated]
                #row = [user, b, baskets.iloc[index]['num_baskets']]
                train_valid.append(row)
        
    train_set_augmented_without_summer = pd.DataFrame(train_valid, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    train_set_augmented_without_summer.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/train_sample_augmented_without_summer.json', orient='records', lines=True)

    valid_data = input_data2
    index=0
    for baskets in valid_data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(valid_data['timestamps'][index][tsindex])
            tsindex += 1
        valid_data['baskets'][index] = new_baskets
        valid_data['num_baskets'][index] = len(new_baskets)
        valid_data['timestamps'][index] = ts
        index +=1  
    users = valid_data.userID.values
    valid_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = valid_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if valid_data.iloc[index]['num_baskets']>=3:
            row = [user, b, valid_data.iloc[index]['num_baskets'], valid_data.iloc[index]['last_semester'], valid_data.iloc[index]['timestamps'] ]
            valid_all.append(row)
        index +=1
        #if index==30: break
    valid_set_filtered_without_summer = pd.DataFrame(valid_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    valid_set_filtered_without_summer.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_all_without_summer.json', orient='records', lines=True)

    test_data = input_data3
    # baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  #
    # baskets.columns = ['userID', 'baskets']

    #baskets['num_baskets'] = baskets.baskets.apply(len)
    index=0
    for baskets in test_data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(test_data['timestamps'][index][tsindex])
            tsindex += 1
        test_data['baskets'][index] = new_baskets
        test_data['num_baskets'][index] = len(new_baskets)
        test_data['timestamps'][index] = ts
        index +=1  
    users = test_data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = test_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if test_data.iloc[index]['num_baskets']>=3:
            row = [user, b, test_data.iloc[index]['num_baskets'], test_data.iloc[index]['last_semester'], test_data.iloc[index]['timestamps']]
            test_all.append(row)
        index +=1
        #if index==30: break
    test_set_filtered_without_summer = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_set_filtered_without_summer.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_all_without_summer.json', orient='records', lines=True)

    return train_set_augmented_without_summer, valid_set_filtered_without_summer, test_set_filtered_without_summer


def filter_data_merging_summer_fall(input_data1, input_data2, input_data3):
    train_data = input_data1
    itemIDs = {}
    index=0
    for baskets in train_data['baskets']:
        for basket in baskets:
            for item in basket:
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)
        index +=1   
    # baskets['num_baskets'] = baskets.baskets.apply(len)
    item_list = list(itemIDs.keys())
    #augmentation of training data 
    users = train_data.userID.values
    train_valid = []
    for user in users:
        index = train_data[train_data['userID'] == user].index.values[0]
        if(train_data.iloc[index]['num_baskets']==3):
            b = train_data.iloc[index]['baskets'][0:]
            #b = baskets.iloc[index]['baskets'][0:]
            #if baskets.iloc[index]['num_baskets']>=2:
            row = [user, b, train_data.iloc[index]['num_baskets'], train_data.iloc[index]['last_semester'], train_data.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_valid.append(row)
        elif train_data.iloc[index]['num_baskets']>3:
            nb = train_data.iloc[index]['num_baskets']
            for i in range(2, nb):
                b = train_data.iloc[index]['baskets'][0:i+1]
                #b = baskets.iloc[index]['baskets'][0:]
                #if baskets.iloc[index]['num_baskets']>=2:
                #max_len = max(max_len, data.iloc[index]['num_baskets'])
                num_baskets = len(b)
                last_sem = train_data.iloc[index]['timestamps'][i]
                timestamp_updated = train_data.iloc[index]['timestamps'][0:i+1]
                row = [user, b, num_baskets, last_sem, timestamp_updated]
                #row = [user, b, baskets.iloc[index]['num_baskets']]
                train_valid.append(row)
        
    train_set_augmented_merging_summer_fall = pd.DataFrame(train_valid, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    train_set_augmented_merging_summer_fall.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/train_sample_augmented_merging_summer_fall.json', orient='records', lines=True)

    valid_data = input_data2
    index=0
    for baskets in valid_data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(valid_data['timestamps'][index][tsindex])
            tsindex += 1
        valid_data['baskets'][index] = new_baskets
        valid_data['num_baskets'][index] = len(new_baskets)
        valid_data['timestamps'][index] = ts
        index +=1  
    users = valid_data.userID.values
    valid_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = valid_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if valid_data.iloc[index]['num_baskets']>=3:
            row = [user, b, valid_data.iloc[index]['num_baskets'], valid_data.iloc[index]['last_semester'], valid_data.iloc[index]['timestamps'] ]
            valid_all.append(row)
        index +=1
        #if index==30: break
    valid_set_filtered_merging_summer_fall = pd.DataFrame(valid_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    valid_set_filtered_merging_summer_fall.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_all_merging_summer_fall.json', orient='records', lines=True)

    test_data = input_data3
    # baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  #
    # baskets.columns = ['userID', 'baskets']

    #baskets['num_baskets'] = baskets.baskets.apply(len)
    index=0
    for baskets in test_data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(test_data['timestamps'][index][tsindex])
            tsindex += 1
        test_data['baskets'][index] = new_baskets
        test_data['num_baskets'][index] = len(new_baskets)
        test_data['timestamps'][index] = ts
        index +=1  
    users = test_data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = test_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if test_data.iloc[index]['num_baskets']>=3:
            row = [user, b, test_data.iloc[index]['num_baskets'], test_data.iloc[index]['last_semester'], test_data.iloc[index]['timestamps']]
            test_all.append(row)
        index +=1
        #if index==30: break
    test_set_filtered_merging_summer_fall = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_set_filtered_merging_summer_fall.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_all_merging_summer_fall.json', orient='records', lines=True)

    return train_set_augmented_merging_summer_fall, valid_set_filtered_merging_summer_fall, test_set_filtered_merging_summer_fall

def filter_data_unique(input_data1, input_data2, input_data3):
    train_data = input_data1
    itemIDs = {}
    index=0
    for baskets in train_data['baskets']:
        for basket in baskets:
            for item in basket:
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)
        index +=1   
    # baskets['num_baskets'] = baskets.baskets.apply(len)
    item_list = list(itemIDs.keys())

    valid_data = input_data2
    index=0
    for baskets in valid_data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(valid_data['timestamps'][index][tsindex])
            tsindex += 1
        valid_data['baskets'][index] = new_baskets
        valid_data['num_baskets'][index] = len(new_baskets)
        valid_data['timestamps'][index] = ts
        index +=1  
    users = valid_data.userID.values
    valid_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = valid_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if valid_data.iloc[index]['num_baskets']>=3:
            row = [user, b, valid_data.iloc[index]['num_baskets'], valid_data.iloc[index]['last_semester'], valid_data.iloc[index]['timestamps'] ]
            valid_all.append(row)
        index +=1
        #if index==30: break
    valid_set_filtered_unique = pd.DataFrame(valid_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    valid_set_filtered_unique.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/valid_sample_filtered_unique_CR.json', orient='records', lines=True)

    test_data = input_data3
    # baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  #
    # baskets.columns = ['userID', 'baskets']

    #baskets['num_baskets'] = baskets.baskets.apply(len)
    index=0
    for baskets in test_data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(test_data['timestamps'][index][tsindex])
            tsindex += 1
        test_data['baskets'][index] = new_baskets
        test_data['num_baskets'][index] = len(new_baskets)
        test_data['timestamps'][index] = ts
        index +=1  
    users = test_data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = test_data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if test_data.iloc[index]['num_baskets']>=3:
            row = [user, b, test_data.iloc[index]['num_baskets'], test_data.iloc[index]['last_semester'], test_data.iloc[index]['timestamps']]
            test_all.append(row)
        index +=1
        #if index==30: break
    test_set_filtered_unique = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    test_set_filtered_unique.to_json('/Users/mkhan149/Downloads/Experiments/Filtered_data/test_sample_filtered_unique_CR.json', orient='records', lines=True)

    return valid_set_filtered_unique, test_set_filtered_unique

def calculate_actual_enrollment_filtered(term_dict, term_dict_all_prior):
   
    output_path1= "/Users/mkhan149/Downloads/Experiments/test_course_allocation_filtered.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    # error_list = []
    # ab_error_list = []
    # st_error_list = []
    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        count_course_all = term_dict_all_prior[semester]
 
        for item1 in count_course.keys():
            f.write("Semester: ")
            f.write(str(semester)+ " ")
            f.write("Course ID: ")
            f.write(str(item1)+ " ")
            f.write("actual: ")
            f.write(str(count_course[item1])+ " ")
            f.write(str(count_course_all[item1])+ " ")
            row = [semester, item1, count_course[item1], count_course_all[item1]]
            course_allocation.append(row)
   
    f.close()
    course_allocation_actual_filtered = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'actual_n_all'])
    course_allocation_actual_filtered.to_csv('/Users/mkhan149/Downloads/Experiments/course_allocation_actual_filtered.csv')
    return course_allocation_actual_filtered

def calculate_actual_enrollment_non_filtered(term_dict, term_dict_all_prior):
   
    output_path1= "/Users/mkhan149/Downloads/Experiments/test_course_allocation_non_filtered.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    # error_list = []
    # ab_error_list = []
    # st_error_list = []
    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        count_course_all = term_dict_all_prior[semester]
 
        for item1 in count_course.keys():
            f.write("Semester: ")
            f.write(str(semester)+ " ")
            f.write("Course ID: ")
            f.write(str(item1)+ " ")
            f.write("actual: ")
            f.write(str(count_course[item1])+ " ")
            f.write(str(count_course_all[item1])+ " ")
            row = [semester, item1, count_course[item1], count_course_all[item1]]
            course_allocation.append(row)
   
    f.close()
    course_allocation_actual_non_filtered = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'actual_n_all'])
    course_allocation_actual_non_filtered.to_csv('/Users/mkhan149/Downloads/Experiments/course_allocation_actual_non_filtered.csv')
    return course_allocation_actual_non_filtered

def calculate_actual_enrollment_filtered(term_dict, term_dict_all_prior):
   
    output_path1= "/Users/mkhan149/Downloads/Experiments/test_course_allocation_filtered.txt"
    f = open(output_path1, "w") #generating text file with recommendation using filtering function
    course_allocation = []
    # error_list = []
    # ab_error_list = []
    # st_error_list = []
    for keys in term_dict.keys():
        semester = keys
        count_course = term_dict[semester]
        count_course_all = term_dict_all_prior[semester]
 
        for item1 in count_course.keys():
            f.write("Semester: ")
            f.write(str(semester)+ " ")
            f.write("Course ID: ")
            f.write(str(item1)+ " ")
            f.write("actual: ")
            f.write(str(count_course[item1])+ " ")
            f.write(str(count_course_all[item1])+ " ")
            row = [semester, item1, count_course[item1], count_course_all[item1]]
            course_allocation.append(row)
   
    f.close()
    course_allocation_actual_filtered = pd.DataFrame(course_allocation, columns=['Semester', 'Course_ID', 'actual_n', 'actual_n_all'])
    course_allocation_actual_filtered.to_csv('/Users/mkhan149/Downloads/Experiments/course_allocation_actual_filtered_updated.csv')
    return course_allocation_actual_filtered

def feature_generation(term_dict_train):
    avg_enrollment = []
    course_dict = {}
    for keys in term_dict_train.keys():
        semester = keys
        count_course = term_dict_train[semester]
        sum = 0
        cnt = 0
        for course, values in count_course.items():
            if course not in course_dict:
                course_dict[course] = [[semester, values]]
            else:
                #list1 = course_dict[course]
                #list2.append(list1)
                #list2.append([semester, values])
                course_dict[course].append([semester, values])
            sum += values
            cnt += 1
        avg_enrollment.append(float(sum/cnt))
                    
    list_of_semesters = list(term_dict_train.keys())
    course_df = []
    for keys, values in course_dict.items():
        course = keys
        semester_enr = values
        list_of_enr = {}
        for s in list_of_semesters:
            list_of_enr[s] = 0

        for s_en in semester_enr:
            sem, enr = s_en
            list_of_enr[sem] = enr
        n_of_enr = list(list_of_enr.values())
        features = []
        features = n_of_enr.copy()
        if course[3]=='1':
            cat = [1, 0, 0, 0, 0] 
            features += cat
        elif course[3]=='2':
            cat = [0, 1, 0, 0, 0] 
            features += cat
        elif course[3]=='3':
            cat = [0, 0, 1, 0, 0] 
            features += cat
        elif course[3]=='4':
            cat = [0, 0, 0, 1, 0] 
            features += cat
        elif course[3]=='5':
            cat = [0, 0, 0, 0, 1] 
            features += cat
        row = [course]
        row+= features
        course_df.append(row)
    columns2 = ['course_id']
    columns2 += list_of_semesters
    cat_feat = ['level1', 'level2', 'level3', 'level4', 'level5']
    columns2 += cat_feat
    course_df = pd.DataFrame(course_df, columns=columns2)
    course_df.to_csv('/Users/mkhan149/Downloads/Experiments/course_df.csv')
    course_df.to_json('/Users/mkhan149/Downloads/Experiments/course_df.json', orient='records', lines=True)
    row1 = []
    row1.append(avg_enrollment)
    avg_enrollment1 = pd.DataFrame(row1, columns=list_of_semesters)
    avg_enrollment1.to_csv('/Users/mkhan149/Downloads/Experiments/avg_enrollment_prior_term.csv')
    avg_enrollment1.to_json('/Users/mkhan149/Downloads/Experiments/avg_enrollment_prior_term.json', orient='records', lines=True)
    
def feature_generation_all(term_dict_all):
    avg_enrollment = []
    course_dict = {}
    for keys in term_dict_all.keys():
        semester = keys
        count_course = term_dict_all[semester]
        sum = 0
        cnt = 0
        for course, values in count_course.items():
            if course not in course_dict:
                course_dict[course] = [[semester, values]]
            else:
                #list1 = course_dict[course]
                #list2.append(list1)
                #list2.append([semester, values])
                course_dict[course].append([semester, values])
            sum += values
            cnt += 1
        avg_enrollment.append(float(sum/cnt))
                    
    list_of_semesters = list(term_dict_all.keys())
    course_df = []
    for keys, values in course_dict.items():
        course = keys
        semester_enr = values
        list_of_enr = {}
        for s in list_of_semesters:
            list_of_enr[s] = 0

        for s_en in semester_enr:
            sem, enr = s_en
            list_of_enr[sem] = enr
        n_of_enr = list(list_of_enr.values())
        features = []
        features = n_of_enr.copy()
        if course[3]=='1':
            cat = [1, 0, 0, 0, 0] 
            features += cat
        elif course[3]=='2':
            cat = [0, 1, 0, 0, 0] 
            features += cat
        elif course[3]=='3':
            cat = [0, 0, 1, 0, 0] 
            features += cat
        elif course[3]=='4':
            cat = [0, 0, 0, 1, 0] 
            features += cat
        elif course[3]=='5':
            cat = [0, 0, 0, 0, 1] 
            features += cat
        row = [course]
        row+= features
        course_df.append(row)
    columns2 = ['course_id']
    columns2 += list_of_semesters
    cat_feat = ['level1', 'level2', 'level3', 'level4', 'level5']
    columns2 += cat_feat
    course_df = pd.DataFrame(course_df, columns=columns2)
    course_df.to_csv('/Users/mkhan149/Downloads/Experiments/course_df_all.csv')
    course_df.to_json('/Users/mkhan149/Downloads/Experiments/course_df_all.json', orient='records', lines=True)
    row1 = []
    row1.append(avg_enrollment)
    avg_enrollment1 = pd.DataFrame(row1, columns=list_of_semesters)
    avg_enrollment1.to_csv('/Users/mkhan149/Downloads/Experiments/avg_enrollment_prior_term_all.csv')
    avg_enrollment1.to_json('/Users/mkhan149/Downloads/Experiments/avg_enrollment_prior_term_all.json', orient='records', lines=True)
 

if __name__ == '__main__':
    data = preprocess_data('/Users/mkhan149/Downloads/Experiments/dataKni.csv')
    #data2 = preprocess_data_allocation('/Users/mkhan149/Downloads/Experiments/dataKni.csv')
    # data_en_pred = preprocess_data_en_pred('/Users/mkhan149/Downloads/Experiments/dataKni.csv')
    # term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses_2(data_en_pred)

    # data_en_pred_filtered = preprocess_data_en_pred_filtered('/Users/mkhan149/Downloads/Experiments/dataKni.csv')
    # term_dict_all_filtered, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms = calculate_avg_n_actual_courses_2(data_en_pred_filtered)
    # term_dict_all_filtered, data_set_filtered_all = another_filtering_all(data_en_pred_filtered)
    # term_dict_all_filtered = dict(sorted(term_dict_all_filtered.items(), key=lambda item: item[0], reverse= True))
    # feature_generation_all(term_dict_all_filtered)

    # data_without_summer = preprocess_data_without_summer('/Users/mkhan149/Downloads/Experiments/dataKni.csv')
    # data_merging_summer_fall = preprocess_data_merging_summer_fall('/Users/mkhan149/Downloads/Experiments/dataKni.csv')
    #print(data)
    # train_data, valid_data, test_data = split_data(data)
    train_set_all, valid_data, test_data = split_data(data)
    train_set_all_with_more_target_baskets, valid_data_with_more_target_baskets, test_data_with_more_target_baskets = split_data_with_more_target_baskets(data)
    #train_set_all, valid_data, test_data = split_data(data2)
    # train_set_all_non_filtered, valid_data_non_filtered, test_data_non_filtered = split_data_new(data)
    # term_dict_test = calculate_term_dict_test(test_data_non_filtered)
    # count_total_course = {}
    # for keys, values in term_dict_test.items():
    #     count_course_dict = values
    #     count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
    #     count3 = 0
    #     for cnt in count_course_dict.values():
    #         count3 += cnt
    #     count_total_course[keys] = count3
    #     term_dict_test[keys] = count_course_dict
    # term_dict_test = dict(sorted(term_dict_test.items(), key=lambda item: item[0], reverse= True))
    # course_allocation_actual_non_filtered = calculate_actual_enrollment_non_filtered(term_dict_test, term_dict_all)

    # df_all_rows = pd.concat([train_set_all, valid_data])
    # df_all_data = pd.concat([df_all_rows, test_data])
    #for course allocation problem preprocessing data where a course appeared in more than 3 terms
    # train_set_all, course_number_terms, frequency_of_courses, count_course_avg, course_sd_main, frequency_of_courses_in_last_term, course_info = calculate_avg_n_actual_courses(train_set_all)
    #course allocation problem - to apply random forest, SVM
    # term_dict_train, frequency_of_courses2, count_course_avg2, course_sd_main2, course_number_terms2 = calculate_avg_n_actual_courses_2(train_set_all)
    # term_dict_train = dict(sorted(term_dict_train.items(), key=lambda item: item[0], reverse= True))
    # feature_generation(term_dict_train)
    # train_set_all_without_summer, valid_data_without_summer, test_data_without_summer = split_data_without_summer(data_without_summer)
    # train_set_all_merging_summer_fall, valid_data_merging_summer_fall, test_data_merging_summer_fall = split_data_merging_summer_fall(data_merging_summer_fall)

    # valid_data_unique, test_data_unique = split_data2(data2)
    valid_data_unique, test_data_unique = split_data2(data)

    train_data_augmented, valid_data_filtered, test_data_filtered = filter_data(train_set_all, valid_data, test_data)

    # term_dict_test_filtered = calculate_term_dict_test(test_data_filtered)
    # count_total_course = {}
    # for keys, values in term_dict_test_filtered.items():
    #     count_course_dict = values
    #     count_course_dict = dict(sorted(count_course_dict.items(), key=lambda item: item[1], reverse= True))
    #     count3 = 0
    #     for cnt in count_course_dict.values():
    #         count3 += cnt
    #     count_total_course[keys] = count3
    #     term_dict_test_filtered[keys] = count_course_dict
    # term_dict_test_filtered = dict(sorted(term_dict_test_filtered.items(), key=lambda item: item[0], reverse= True))
    # course_allocation_actual_filtered = calculate_actual_enrollment_filtered(term_dict_test_filtered, term_dict_all_filtered)

    # train_data_augmented_without_summer, valid_data_filtered_without_summer, test_data_filtered_without_summer = filter_data_without_summer(train_set_all_without_summer, valid_data_without_summer, test_data_without_summer)
    # train_data_augmented_merging_summer_fall, valid_data_filtered_merging_summer_fall, test_data_filtered_merging_summer_fall = filter_data_merging_summer_fall(train_set_all_merging_summer_fall, valid_data_merging_summer_fall, test_data_merging_summer_fall )
    
    valid_data_filtered_unique, test_data_filtered_unique = filter_data_unique(train_set_all, valid_data_unique, test_data_unique)
