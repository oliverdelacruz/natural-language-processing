# NLU Project
# Configuration file
# Description: The script setup all the parameters for training and saving the output

# Import libraries
import tensorflow as tf
import os
import sys
import time
from pathlib import PurePath

# Define parent directory
project_dir = str(PurePath(__file__).parent)  # root of git-project

# Labels for output and data
label_output = "runs"
label_data = "data"

# Define output directory
timestamp = str(int(time.time()))
output_dir = os.path.abspath(os.path.join(os.path.curdir, label_output, timestamp))

# Define buckets and max length
max_length = 150
buckets = [(5,7), (10,12), (15,17), (20,22), (25, 27)]
buckets.extend([(i, i+2) for i in range(30, max_length + 15, 20)])

# Setup constant parameters
flags = tf.app.flags

# Define directory parameters
flags.DEFINE_string('output_dir', output_dir, 'The directory where all results are stored')
flags.DEFINE_string('data_dir', os.path.join(project_dir, label_data), 'The directory where all input data are stored')
flags.DEFINE_string('file_dir', os.path.join(project_dir, r"runs\final\checkpoints\model-83349"), 'The directory where all input data are stored')

# Define model parameters
flags.DEFINE_bool('debug', True, 'Run in debug mode')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('rnn_size', 256, 'Number of hidden units')
flags.DEFINE_integer('rnn_size_reduced', 256, 'Number of hidden units')
flags.DEFINE_integer('embedding_dim', 100,'The dimension of the embedded vectors')
flags.DEFINE_string('model_name', 'seq2seq', 'Name of the trained model')
flags.DEFINE_integer('vocab_size', 22500, 'Total number of different words')
flags.DEFINE_float('grad_clip', 10, 'Limitation of the gradient')
flags.DEFINE_integer('max_seq_length', max_length, 'Maximum sequence length')
flags.DEFINE_integer('vocab_tags', 4, 'Number of special tags')
flags.DEFINE_float('decay_learning_rate', 0.9, 'The decaying factor for learning rate')
flags.DEFINE_float('dropout_prob_keep', 0.75, 'The dropout probability to keep the units')
flags.DEFINE_integer('n_unk', 1, 'The number of maximum unks allowed per sentence')
flags.DEFINE_integer('n_units_attention', 256, 'The number of units for the attention')

# Define training parameters
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('n_epochs', 5, 'Number of epochs')

# Define general parameters
flags.DEFINE_integer('summary_every', 5, "generate a summary every `n` step. This is for visualization purposes")
flags.DEFINE_integer('n_checkpoints_to_keep', 5,'keep maximum `integer` of chekpoint files')
flags.DEFINE_integer('evaluate_every', 15000,'evaluate trained model every n step')
flags.DEFINE_integer('save_every', 10000, 'store checkpoint every n step')

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Obtain the current paremeters
def get_configuration():
    global FLAGS
    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)
    return FLAGS

# Print the current paramets
def print_configuration():
    print("Parameters: ")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
