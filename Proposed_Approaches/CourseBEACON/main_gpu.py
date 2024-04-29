import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import os
import utils
import models
import procedure
import offered_courses

# Parameters
# GPU & Seed

tf.compat.v1.flags.DEFINE_string("device_id", "828BD340-1C5E-5751-9874-5FF047FFDB54", "GPU device is to be used in training (default: None)")
tf.compat.v1.flags.DEFINE_integer("seed", 89, "Seed value for reproducibility (default: 89)")

# Model hyper-parameters
tf.compat.v1.flags.DEFINE_string("data_dir", "./Course_Beacon", "The input data directory (default: None)")
tf.compat.v1.flags.DEFINE_string("output_dir", "./Course_Beacon", "The output directory (default: None)")
tf.compat.v1.flags.DEFINE_string("tensorboard_dir","./Course_Beacon", "The tensorboard directory (default: None)")

tf.compat.v1.flags.DEFINE_integer("emb_dim", 64, "The dimensionality of embedding (default: 2)")
tf.compat.v1.flags.DEFINE_integer("rnn_unit", 128, "The number of hidden units of RNN (default: 4)")
tf.compat.v1.flags.DEFINE_integer("hidden_layer", 1, "The number of hidden layers of RNN (default: 2)")
tf.compat.v1.flags.DEFINE_integer("nb_hop", 1, "The number of neighbor hops  (default: 1)")
tf.compat.v1.flags.DEFINE_float("alpha", 0.3, "The reguralized hyper-parameter (default: 0.5)")
tf.compat.v1.flags.DEFINE_integer("matrix_type", 1, "The type of adjacency matrix (0=zero,1=real,default:1)")

# Training hyper-parameters
tf.compat.v1.flags.DEFINE_integer("nb_epoch", 25, "Number of epochs (default: 15)")
tf.compat.v1.flags.DEFINE_integer("early_stopping_k", 5, "Early stopping patience (default: 5)")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.compat.v1.flags.DEFINE_float("epsilon", 1e-8, "The epsilon threshold in training (default: 1e-8)")
tf.compat.v1.flags.DEFINE_float("dropout_rate", 0.4, "Dropout keep probability for RNN (default: 0.3)")
tf.compat.v1.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")
tf.compat.v1.flags.DEFINE_integer("display_step", 10, "Show loss/acc for every display_step batches (default: 10)")
tf.compat.v1.flags.DEFINE_string("rnn_cell_type", "LSTM", " RNN Cell Type like LSTM, GRU, etc. (default: LSTM)")
tf.compat.v1.flags.DEFINE_integer("top_k", 10, "Top K Accuracy (default: 10)")
tf.compat.v1.flags.DEFINE_boolean("train_mode", True, "Turn on/off the training mode (default: False)")
tf.compat.v1.flags.DEFINE_boolean("tune_mode", True, "Turn on/off the tunning mode (default: False)")
tf.compat.v1.flags.DEFINE_boolean("prediction_mode", True, "Turn on/off the testing mode (default: False)")

config = tf.compat.v1.flags.FLAGS
print("---------------------------------------------------")
print("SeedVal = " + str(config.seed))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

#print_ind = False

# for reproducibility
np.random.seed(config.seed)
#tf.set_random_seed(config.seed)
tf.random.set_seed(config.seed)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf_upgrade_v2.set_verbosity()

gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.log_device_placement = False

# ----------------------- MAIN PROGRAM -----------------------

data_dir = config.data_dir
output_dir = config.output_dir
tensorboard_dir=config.tensorboard_dir

#training_file = data_dir + "/Main_dataset/train.txt"
#validate_file = data_dir + "/Main_dataset/validate.txt"
#testing_file = data_dir + "/Main_dataset/test.txt"

training_file2 = data_dir + "/train_main.txt"
validate_file2 = data_dir + "/validate_main.txt"
testing_file = data_dir + "/test_main.txt"
#testing_file = data_dir + "/input_dir/test_data.txt"

print("***************************************************************************************")
print("Output Dir: " + output_dir)

# Create directories
print("@Create directories")
utils.create_folder(output_dir + "/models")
utils.create_folder(output_dir + "/topN")

if tensorboard_dir is not None:
    utils.create_folder(tensorboard_dir)

# Load train, validate & test
print("@Load train,validate&test data")
training_instances = utils.read_file_as_lines(training_file2)
nb_train = len(training_instances)
print(" + Total training sequences: ", nb_train)

# #more training data dividing sequence of length 3 to two sequences of length 2 and 3
# utils.sequence_of_baskets_training(training_instances)
# training_file2= './Course_Beacon/train_main_new.txt'
# training_instances2 = utils.read_file_as_lines(training_file2)
# nb_train = len(training_instances2)
# print(" + Total training sequences: ", nb_train)

validate_instances = utils.read_file_as_lines(validate_file2)
nb_validate = len(validate_instances)
print(" + Total validating sequences: ", nb_validate)

testing_instances = utils.read_file_as_lines(testing_file)
nb_test = len(testing_instances)
print(" + Total testing sequences: ", nb_test)

# Create dictionary
print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict, rev_item_dict, item_probs, item_dict_train = utils.build_knowledge(training_instances, validate_instances)
#MAX_SEQ_LENGTH += 2

#update item dict with test data
#item_dict, rev_item_dict, item_probs = utils.update_item_dict(testing_instances, item_dict, rev_item_dict, item_probs)

print("#Statistic")
NB_ITEMS = len(item_dict_train)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

matrix_type = config.matrix_type
if matrix_type == 0:
    print("@Create an zero adjacency matrix")
    adj_matrix = utils.create_zero_matrix(NB_ITEMS)
else:
    print("@Load the normalized adjacency matrix")
    matrix_fpath = data_dir + "/adj_matrix/r_matrix_" + str(config.nb_hop)+ "w.npz"
    adj_matrix = sp.load_npz(matrix_fpath)
    print(" + Real adj_matrix has been loaded from" + matrix_fpath)

#print(adj_matrix)

print("@Compute #batches in train/validation/test")
total_train_batches = utils.compute_total_batches(nb_train, config.batch_size)
total_validate_batches = utils.compute_total_batches(nb_validate, config.batch_size)
total_test_batches = utils.compute_total_batches(nb_test, config.batch_size)
print(" + #batches in train ", total_train_batches)
print(" + #batches in validate ", total_validate_batches)
print(" + #batches in test ", total_test_batches)

#calculate offered courses at each semester
#offered_courses_train = utils.calculate_offered_courses(training_instances, item_dict)
#print(offered_courses_train)
#offered_courses_valid = utils.calculate_offered_courses(validate_instances, item_dict)
#offered_courses_test = utils.calculate_offered_courses(testing_instances, item_dict)
offered_courses_all = offered_courses.offered_course_cal('./all_data_CR.json')

model_dir = output_dir + "/models"
if config.train_mode:
    with tf.compat.v1.Session(config=gpu_config) as sess:
        # Training
        # ==================================================
        # Create data generator
        train_generator = utils.seq_batch_generator(training_instances, item_dict_train, config.batch_size)
        #train_generator = utils.seq_batch_generator(training_instances2, item_dict, config.batch_size)
        validate_generator = utils.seq_batch_generator(validate_instances, item_dict_train, config.batch_size, False)
        test_generator = utils.seq_batch_generator(testing_instances, item_dict_train, config.batch_size, False)
        
        # Initialize the network
        print(" + Initialize the network")
        net = models.Beacon(sess, config.emb_dim, config.rnn_unit, config.hidden_layer, config.alpha, MAX_SEQ_LENGTH, item_probs, adj_matrix, config.top_k, 
                             config.batch_size, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)

        print(" + Initialize parameters")
        sess.run(tf.compat.v1.global_variables_initializer())

        print("================== TRAINING ====================")
        print("@Start training")
        procedure.train_network(sess, net, train_generator, validate_generator, config.nb_epoch,
                                total_train_batches, total_validate_batches, config.display_step,
                                config.early_stopping_k, config.epsilon, tensorboard_dir, model_dir,
                                test_generator, total_test_batches)

        # Reset before re-loading
    tf.compat.v1.reset_default_graph()

if config.prediction_mode or config.tune_mode:
    with tf.compat.v1.Session(config=gpu_config) as sess:
        print(" + Initialize the network")

        net = models.Beacon(sess, config.emb_dim, config.rnn_unit, config.hidden_layer, config.alpha, MAX_SEQ_LENGTH, item_probs, adj_matrix, config.top_k, 
                        config.batch_size, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)

        print(" + Initialize parameters")
        sess.run(tf.compat.v1.global_variables_initializer())

        print("===============================================\n")
        print("@Restore the model from " + model_dir)
        # Reload the best model
        saver = tf.compat.v1.train.Saver()
        recent_dir = utils.recent_model_dir(model_dir)
        saver.restore(sess, model_dir + "/" + recent_dir + "/model.ckpt")
        print("Model restored from file: %s" % recent_dir)

        #calculating recall for training set using filtering function for recommendation
        if config.prediction_mode:
            print("@Start generating prediction for training set")
            train_generator1 = utils.seq_batch_generator(training_instances, item_dict_train, config.batch_size, False)
            # total_train_batches = utils.compute_total_batches(len(training_instances), config.batch_size)
            procedure.generate_prediction_for_training(net, train_generator1, total_train_batches, config.display_step, item_dict_train,
                        rev_item_dict, output_dir + "/topN/train_recall.txt", offered_courses_all)

        # Tunning
        # ==================================================
        if config.tune_mode:
            print("@Start tunning")
            validate_generator = utils.seq_batch_generator(validate_instances, item_dict_train, config.batch_size, False)
            procedure.tune(net, validate_generator, total_validate_batches, config.display_step, item_dict,
                        rev_item_dict, output_dir + "/topN/val_recall.txt", offered_courses_all, nb_validate, config.batch_size)

        # Testing
        # ==================================================
        output_dir2 = data_dir + "/output_dir"
        utils.create_folder(output_dir2)
        output_path= output_dir2+ "/prediction2.txt"
        if config.prediction_mode:
            test_generator = utils.seq_batch_generator(testing_instances, item_dict, config.batch_size, False)

            print("@Start generating prediction")
            procedure.generate_prediction(net, test_generator, total_test_batches, config.display_step, item_dict,
                        rev_item_dict, output_dir + "/topN/prediction.txt", offered_courses_all, output_path, output_dir2, nb_test, config.batch_size)
        
    tf.compat.v1.reset_default_graph()    
