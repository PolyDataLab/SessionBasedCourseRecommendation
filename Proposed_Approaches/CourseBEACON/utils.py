import scipy.sparse as sp
import numpy as np, os, re, itertools, math


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

    reversed_item_dict = dict(zip(item_dict_train.values(), item_dict_train.keys()))
    return MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_dict_train

# def update_item_dict(testing_instances, item_dict, reversed_item_dict, item_probs):
#     #MAX_SEQ_LENGTH = 0
#     item_dict_temp = {}

#     for line in testing_instances:
#         elements = line.split("|")

#         if len(elements) >= 3:
#             basket_seq = elements[1:]

#         for basket in basket_seq:
#             item_list = re.split('[\\s]+', basket)
#             for item_obs in item_list:
#                 if item_obs not in item_dict_temp:
#                     item_dict_temp[item_obs] = 1
#                 # else:
#                 #     item_freq_dict[item_obs] += 1

#     items = sorted(list(item_dict_temp.keys()))
#     for item in items:
#         if item not in item_dict and len(item)>0:
#             item_dict[item] = len(item_dict)
#             item_probs.append(item_dict_temp[item])

#     item_probs = np.asarray(item_probs, dtype=np.float32)
#     item_probs /= np.sum(item_probs)

#     reversed_item_dict = dict(zip(item_dict.values(), item_dict.keys()))
#     return item_dict, reversed_item_dict, item_probs


#build sparse adjacency matrix using training and validation data
def build_sparse_adjacency_matrix_v2(training_instances, validate_instances, item_dict):
    NB_ITEMS = len(item_dict)

    pairs = {}
    for line in training_instances:
        elements = line.split("|")

        if len(elements) == 5:
            basket_seq = elements[2:]
        else:
            basket_seq = [elements[-1]]
            #basket_seq = elements[1:]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            id_list = [item_dict[item] for item in item_list]

            for t in list(itertools.product(id_list, id_list)):
                add_tuple(t, pairs)

    for line in validate_instances:
        elements = line.split("|")

        #label = int(elements[0])
        #if label != 1 and len(elements) == 3:
        if len(elements) == 5:
            basket_seq = elements[2:]
        # else:
        #     basket_seq = [elements[-1]]
            #basket_seq = elements[1:]

        for basket in basket_seq:
            item_list = re.split('[\\s]+', basket)
            id_list = [item_dict[item] for item in item_list]

            for t in list(itertools.product(id_list, id_list)):
                add_tuple(t, pairs)

    return create_sparse_matrix(pairs, NB_ITEMS)

#create tuples for each pair of courses
def add_tuple(t, pairs):
    assert len(t) == 2
    if t[0] != t[1]:
        if t not in pairs:
            pairs[t] = 1
        else:
            pairs[t] += 1

#create sparse matrix for all pairs of courses 
def create_sparse_matrix(pairs, NB_ITEMS):
    row = [p[0] for p in pairs]
    col = [p[1] for p in pairs]
    data = [pairs[p] for p in pairs]

    adj_matrix = sp.csc_matrix((data, (row, col)), shape=(NB_ITEMS, NB_ITEMS), dtype="float32")
    nb_nonzero = len(pairs)
    density = nb_nonzero * 1.0 / NB_ITEMS / NB_ITEMS
    print("Density: {:.6f}".format(density))

    return sp.csc_matrix(adj_matrix, dtype="float32")

#generate batch of basket sequences using raw lines
def seq_batch_generator(raw_lines, item_dict, batch_size, is_train=True):
    total_batches = compute_total_batches(len(raw_lines), batch_size)
    
    O = []
    S = []
    L = []
    Y = []
    U = [] #userID
    T = [] # target semester
    K = [] #top_k

    batch_id = 0
    while 1:
        lines = raw_lines[:]
        # print("come here again")
        # print("length of S: ", len(S))

        if is_train:
            np.random.shuffle(lines)

        for line in lines:
            #if not is_train and batch_id==39: print("length of S at first: ", len(S))
            elements = line.split("|")

            #label = float(elements[0])
            bseq = elements[2:-1]
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
            
            K.append(len(titem))
            
            #Y.append(create_binary_vector(target_item_list, item_dict))
            Y.append(create_binary_vector(titem, item_dict))

            s = []
            for basket in bseq:
                item_list = re.split('[\\s]+', basket)
                id_list = []
                for item in item_list:
                    if len(item)>0:
                        id_list.append(item_dict[item])
                        #id_list = [item_dict[item] for item in item_list]
                s.append(id_list.copy())
            S.append(s)

            if len(S) % batch_size == 0:
                yield batch_id, {'S': np.asarray(S, dtype = object), 'L': np.asarray(L, dtype = object), 'Y': np.asarray(Y, dtype = object), 'O': np.asarray(O, dtype = object), 'U': np.asarray(U, dtype = object), 'T': np.asarray(T, dtype = object), 'K': np.asarray(K, dtype = object)}
                S = []
                L = []
                O = []
                Y = []
                U = []
                T = []
                K = []
                batch_id += 1
            
            # if not is_train and batch_id==39 and len(S) % batch_size != 0:
            #     print("batch id: ", batch_id)
            #     print("length of S: ", len(S))


            if batch_id == total_batches:
                batch_id = 0
                if not is_train:
                    break

# def sequence_of_baskets_training(input_file):
#     with open('./train_main_new.txt', 'w') as f:
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


#create binary vector for any basket of items
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

#normalize the adjacency matrix
def normalize_adj(adj_matrix):
    """Symmetrically normalize adjacency matrix."""
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    #normalized_matrix = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    normalized_matrix = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt)

    return normalized_matrix.tocsr()

def remove_diag(adj_matrix):
    new_adj_matrix = sp.csr_matrix(adj_matrix)
    new_adj_matrix.setdiag(0.0)
    new_adj_matrix.eliminate_zeros()
    return new_adj_matrix
