# NLU Project
# Main file
# Description: The script loads a model and performs training or predictions

# Import site-packages libraries

# Runs the main script and all the dependencies
from configuration import get_configuration
from configuration import print_configuration
from models.seq2seq_beam_search_model import Seq2seq
from models.se2seq_model_attention_luong import Seq2seq_att_luong
from models.se2seq_model_attention_bahdanau import Seq2seq_att_bahdanau
from utils import Preprocessing
import os
import time

def main():
    # Setup and get current configuration
    config = get_configuration()
    # Print parameters
    print_configuration()
    # Perform preprocessing
    preprocess = Preprocessing(config = config)
    train_input_encoder, train_input_decoder, \
    test_input_encoder, test_input_decoder, = preprocess.prepare_data()
    # Initialize model class - train or infer: select mode
    Seq2seq(config, train_input_encoder, train_input_decoder, test_input_encoder,
            test_input_decoder, preprocess.dict_vocab_reverse, mode = None)
if __name__ == '__main__':
    main()
