import scipy.sparse as sp
import numpy as np, os, re, itertools, math
import random


#if the course is available in the user's previous baskets or course is not available in the offered list for that semester, we filer out this course and
def filtering (item3, user_baskets, offered_course_list, item_dict):
    user_prev_items = []
    #Making a list of previous courses of given semester(s) for the user
    for basket1 in user_baskets:
        #basket_items= basket1.split(" ")
        for item1 in basket1:
            if item1 not in user_prev_items:
                user_prev_items.append(item1)
    offered_course_list_new = []
    for item8 in offered_course_list:
        if item8 in item_dict:
            offered_course_list_new.append(item_dict[item8])
    
    if item3 in user_prev_items or item3 not in offered_course_list_new:
        return True
    else:
        return False
    
    
# argument is training and validating data (lines of baskets)
#return item dictionary of all items, reversed item dictionary and item's probability (frequency) over total number of items
def build_knowledge(training_instances, validate_instances):
    MAX_SEQ_LENGTH = 0
    item_freq_dict = {}

    for line in training_instances:
        elements = line.split("|")

        if len(elements) - 2 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 2

        if len(elements) == 5:
            basket_seq = elements[2:]
        else:
            basket_seq = [elements[-1]]
           #basket_seq = elements[1:]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            for item_obs in item_list:
                if item_obs not in item_freq_dict:
                    item_freq_dict[item_obs] = 1
                else:
                    item_freq_dict[item_obs] += 1
    items_train = sorted(list(item_freq_dict.keys()))
    item_dict_train = dict()
    for item in items_train:
        item_dict_train[item] = len(item_dict_train)

    for line in validate_instances:
        elements = line.split("|")

        if len(elements) - 2 > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = len(elements) - 2

        #label = int(elements[0])
        if len(elements) >= 5:
            basket_seq = elements[2:]
        # else:
        #     basket_seq = [elements[-1]]
            #basket_seq = elements[1:]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            for item_obs in item_list:
                if item_obs not in item_freq_dict:
                    item_freq_dict[item_obs] = 1
                else:
                    item_freq_dict[item_obs] += 1

    items = sorted(list(item_freq_dict.keys()))
    item_dict = dict()
    item_probs = []
    for item in items:
        item_dict[item] = len(item_dict)
        item_probs.append(item_freq_dict[item])

    item_probs = np.asarray(item_probs, dtype=np.float32)
    item_probs /= np.sum(item_probs)

    reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
    return MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_dict_train


#create tuples for each pair of courses
def add_tuple(t, pairs):
    assert len(t) == 2
    if t[0] != t[1]:
        if t not in pairs:
            pairs[t] = 1
        else:
            pairs[t] += 1

#generate batch of basket sequences using raw lines
def seq_batch_generator(raw_lines, item_dict, batch_size, is_train=True):
    total_batches = compute_total_batches(len(raw_lines), batch_size)
    
    O = []
    S = []
    L = []
    Y = []
    U = [] #userID
    T = [] # target semester

    batch_id = 0
    while 1:
        lines = raw_lines[:]

        if is_train:
            np.random.shuffle(lines)

        for line in lines:
            elements = line.split("|")

            #label = float(elements[0])
            bseq = elements[2:-1]
            #bseq = elements[-3:-1] # 2 prior baskets
            tbasket = elements[-1]
            user_info = elements[0] 
            last_semester = elements[1]

            # Keep the length for dynamic_rnn
            L.append(len(bseq))

            # Keep the original last basket
            O.append(tbasket)

            # keep the user id
            U.append(user_info)

            # keep last semester's info
            T.append(last_semester)

            # Add the target basket
            target_item_list = re.split('[\\s]+', tbasket)
            titem = []
            for item4 in target_item_list:
                if(len(item4)>0):
                   titem.append(item4) 
            
            #Y.append(create_binary_vector(target_item_list, item_dict))
            Y.append(create_binary_vector(titem, item_dict))

            s = []
            for basket in bseq:
                item_list = re.split('[\\s]+', basket)
                # random.shuffle(item_list)
                id_list = []
                for item in item_list:
                    if len(item)>0:
                        id_list.append(item_dict[item])
                        #id_list = [item_dict[item] for item in item_list]
                
                s.append(id_list.copy())
            S.append(s)

            if len(S) % batch_size == 0:
                yield batch_id, {'S': np.asarray(S, dtype = object), 'L': np.asarray(L, dtype = object), 'Y': np.asarray(Y, dtype = object), 'O': np.asarray(O, dtype = object), 'U': np.asarray(U, dtype = object), 'T': np.asarray(T, dtype = object)}
                S = []
                L = []
                O = []
                Y = []
                U = []
                T = []
                batch_id += 1

            if batch_id == total_batches:
                batch_id = 0
                if not is_train:
                    break

#generate batch of basket sequences using raw lines
def seq_batch_generator_differently(raw_lines, item_dict, batch_size, is_train=True):
    total_batches = compute_total_batches(len(raw_lines), batch_size)
    
    O = []
    S = []
    L = []
    Y = []
    U = [] #userID
    T = [] # target semester

    batch_id = 0
    while 1:
        lines = raw_lines[:]

        if is_train:
            np.random.shuffle(lines)

        for line in lines:
            elements = line.split("|")

            #label = float(elements[0])
            bseq = elements[2:-1]
            #bseq = elements[-3:-1] # 2 prior baskets
            tbasket = elements[-1]
            user_info = elements[0] 
            last_semester = elements[1]

            # Keep the length for dynamic_rnn
            #L.append(len(bseq))
            L.append(1)

            # Keep the original last basket
            O.append(tbasket)

            # keep the user id
            U.append(user_info)

            # keep last semester's info
            T.append(last_semester)

            # Add the target basket
            target_item_list = re.split('[\\s]+', tbasket)
            titem = []
            for item4 in target_item_list:
                if(len(item4)>0):
                   titem.append(item4) 
            
            #Y.append(create_binary_vector(target_item_list, item_dict))
            Y.append(create_binary_vector(titem, item_dict))

            # s = []
            # for basket in bseq:
            #     item_list = re.split('[\\s]+', basket)
            #     # random.shuffle(item_list)
            #     id_list = []
            #     for item in item_list:
            #         if len(item)>0:
            #             id_list.append(item_dict[item])
            #             # id_list.append(item)
            #             #id_list = [item_dict[item] for item in item_list]
                
            #     s.append(id_list.copy())
            #     # #one hot vector for each basket of courses
            #     # s.append(create_binary_vector(id_list, item_dict))
            # S.append(s)
            
            #s = []
            id_list = []
            for basket in bseq:
                item_list = re.split('[\\s]+', basket)
                # random.shuffle(item_list)
                # id_list = []
                for item in item_list:
                    if len(item)>0:
                        if item_dict[item] not in id_list:
                            id_list.append(item_dict[item])
                        #id_list = [item_dict[item] for item in item_list]
                #s.append(id_list.copy())
            S.append(id_list.copy())

            if len(S) % batch_size == 0:
                yield batch_id, {'S': np.asarray(S, dtype = object), 'L': np.asarray(L, dtype = object), 'Y': np.asarray(Y, dtype = object), 'O': np.asarray(O, dtype = object), 'U': np.asarray(U, dtype = object), 'T': np.asarray(T, dtype = object)}
                S = []
                L = []
                O = []
                Y = []
                U = []
                T = []
                batch_id += 1

            if batch_id == total_batches:
                batch_id = 0
                if not is_train:
                    break

# def sequence_of_baskets_training(input_file):
#     with open('./Others/LSTM/train_main_new.txt', 'w') as f:
#         for line in input_file:
#             elements = line.split("|")
#             if len(elements) > 5:
#                 f.write(elements[0])
#                 f.write("|")
#                 f.write(elements[1])
#                 basket_sequence = elements[2:]
#                 for basket in basket_sequence:
#                     if(len(basket)>0):
#                         f.write('|')
#                         f.write(basket)
#                 f.write('\n')
#             elif len(elements)==5:
#                     for i in range(3,len(elements)):
#                         f.write(elements[0])
#                         f.write("|")
#                         f.write(elements[1])
#                         basket_sequence = elements[2:i+1]
#                         for basket in basket_sequence:
#                             if(len(basket)>0):
#                                 f.write('|')
#                                 f.write(basket)
#                         f.write('\n')        


#create one hot vector for any basket of items
def create_binary_vector(item_list, item_dict):
    v = np.zeros(len(item_dict), dtype='int32')
    for item in item_list:
        #if len(item)>0:
        v[item_dict[item]] = 1
    return v

#list the directory
def list_directory(dir, dir_only=False):
    rtn_list = []
    for f in os.listdir(dir):
        if dir_only and os.path.isdir(os.path.join(dir, f)):
            rtn_list.append(f)
        elif not dir_only and os.path.isfile(os.path.join(dir, f)):
            rtn_list.append(f)
    return rtn_list


def create_folder(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def read_file_as_lines(file_path):
    with open(file_path, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines


def recent_model_dir(dir):
    folderList = list_directory(dir, True)
    folderList = sorted(folderList, key=get_epoch)
    return folderList[-1]


def get_epoch(x):
    idx = x.index('_') + 1
    return int(x[idx:])


def compute_total_batches(nb_intances, batch_size):
    total_batches = int(nb_intances / batch_size)
    if nb_intances % batch_size != 0:
        total_batches += 1
    return total_batches


def create_identity_matrix(nb_items):
    return sp.identity(nb_items, dtype="float32").tocsr()

def create_zero_matrix(nb_items):
    return sp.csr_matrix((nb_items, nb_items), dtype="float32")
