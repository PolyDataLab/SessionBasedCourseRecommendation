import tensorflow as tf
import scipy.sparse as sp
import utils
import preprocess

# Model hyper-parameters
tf.compat.v1.flags.DEFINE_string("data_dir", "./Others/LSTM", "The input data directory (default: None)")
# tf.compat.v1.flags.DEFINE_integer("nb_hop", 1, "The order of the real adjacency matrix (default:1)")

config = tf.compat.v1.flags.FLAGS
print("---------------------------------------------------")
print("Data_dir = " + str(config.data_dir))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

SEED_VALUES = [2, 9, 15, 44, 50, 55, 58, 79, 85, 92]

# ----------------------- MAIN PROGRAM -----------------------
data_dir = config.data_dir
#print(data_dir)
output_dir = data_dir + "/adj_matrix"
#output_dir = "./adj_matrix"
#training_file = data_dir + "/Main_dataset/train.txt"
#training_file = data_dir + "/train2.txt"
training_file = data_dir + '/input_dir/train_data.txt'

#training_file = "./Main_dataset/train.txt"
#validate_file = data_dir + "/Main_dataset/validate.txt"
#validate_file = data_dir + "/validate2.txt"
validate_file = data_dir + '/input_dir/valid_data.txt'

testing_file = data_dir + '/input_dir/test_data.txt'

#validate_file = ".//Main_dataset/validate.txt"
print("***************************************************************************************")
print("Output Dir: " + output_dir)

print("@Create output directory")
utils.create_folder(output_dir)

# Load train, validate & test
print("@Load train,validate&test data")
training_instances2 = utils.read_file_as_lines(training_file)

validate_instances2 = utils.read_file_as_lines(validate_file)

#testing_instances = utils.read_file_as_lines(testing_file)

#data preprocessing (training data)
item_list_train = preprocess.sequence_of_baskets(training_instances2)
#print(len(item_list_train))
#print(len(item_list_train))
training_file_main= data_dir +'/train_main.txt'
#print(training_file_main)
training_instances= utils.read_file_as_lines(training_file_main)
#print(training_instances)
nb_train = len(training_instances)
print(" + Total training sequences: ", nb_train)

# utils.sequence_of_baskets_training(training_instances2)
# training_file2= './Others/LSTM/train_main_new.txt'
# training_instances3 = utils.read_file_as_lines(training_file2)

#data preprocessing (validation data)
preprocess.sequence_of_baskets_valid(validate_instances2, item_list_train)
validate_file2 = data_dir + '/validate_main2.txt'
validate_instances3 = utils.read_file_as_lines(validate_file2)
nb_validate2 = len(validate_instances3)
#print(" + Total validating sequences before removing: ", nb_validate2)
#removing sequences with the length of less than 3
preprocess.filter_valid_data(validate_instances3)
validating_file_main= data_dir +'/validate_main.txt'
validate_instances= utils.read_file_as_lines(validating_file_main)
nb_validate = len(validate_instances)
print(" + Total validating sequences: ", nb_validate)

testing_instances2 = utils.read_file_as_lines(testing_file)
# nb_test = len(testing_instances2)
# print(" + Total testing sequences: ", nb_test)

# Create dictionary
print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, reversed_item_dict, item_probs, item_dict_train = utils.build_knowledge(training_instances, validate_instances)
#print(item_dict)
#update item dict with test data
#item_dict, reversed_item_dict, item_probs = utils.update_item_dict(testing_instances2, item_dict, reversed_item_dict, item_probs)
#print(item_dict)

preprocess.delete_items_from_test_data(testing_instances2, item_dict_train)
testing_file_main2= data_dir +'/test_main2.txt'
testing_instances3= utils.read_file_as_lines(testing_file_main2)
# delete lines with less than 2 lengths
preprocess.delete_lines_with_less_lenghts_from_test(testing_instances3)
testing_file_main= data_dir +'/test_main.txt'
testing_instances= utils.read_file_as_lines(testing_file_main)
nb_test = len(testing_instances)
print(" + Total testing sequences: ", nb_test)

print(" + Total validating sequences: ", nb_validate)
print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)
# for i in item_dict:
#     print(i)
