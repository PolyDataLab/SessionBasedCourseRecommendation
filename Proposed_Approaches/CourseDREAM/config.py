class Config(object):
    def __init__(self):
        self.TRAININGSET_DIR = './Others/DREAM/train_sample_all.json'
        self.VALIDATIONSET_DIR = './Others/DREAM/valid_sample_without_target.json'
        self.TESTSET_DIR = './Others/DREAM/target_set.json'
        self.NEG_SAMPLES = './Others/DREAM/neg_sample.pickle'
        self.MODEL_DIR = './Others/DREAM/runs/'
        self.cuda = False
        self.clip = 10
        #self.epochs = 200
        self.epochs = 10 #25
        #self.batch_size = 256
        self.batch_size = 32
        #self.seq_len = 20
        self.seq_len = 20
        self.learning_rate = 0.01  # Initial Learning Rate
        self.log_interval = 1  # num of batches between two logging
        self.basket_pool_type = 'avg'  # ['avg', 'max']
        self.rnn_type = 'LSTM'  # ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']
        self.rnn_layer_num = 3 # 1 2 3
        self.dropout = 0.4 #0.4 0.5
        #self.num_product = 26991+1 
        self.num_product = 618
        #self.num_product = 617
        #self.num_product = 601
        #self.embedding_dim = 32  # Embedding Layer
        self.embedding_dim = 64 # 8 16 32 
        #self.neg_num = 500  
        self.neg_num = 150
        self.top_k = 10  # Top K 