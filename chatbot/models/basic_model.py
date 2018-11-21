# NLU Project
# Main file
# Description: The script loads a model and performs training or predictions

# Import site-package libraries
import os
import tensorflow as tf
import numpy as np

class BasicModel():

    def __init__(self, config, train_input_encoder, train_input_decoder,
                test_input_encoder, test_input_decoder, dict_vocab_reverse, mode = None, train_emb = False):
        """
        This is the initialization, it is done for every model.
        It initializes all the parameters needed for training the neural network.
        """

        # Save the configuration parameters in the class attributes
        self.config = config
        self.model_name = config.model_name
        self.out_dir = config.output_dir
        self.lr = self.config.lr
        self.rnn_size = config.rnn_size
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size + config.vocab_tags
        self.debug = config.debug
        self.grad_clip = config.grad_clip
        self.allow_soft_placement = config.allow_soft_placement
        self.log_device_placement = config.log_device_placement
        self.num_checkpoint = config.n_checkpoints_to_keep
        self.num_epoch = config.n_epochs
        self.max_seq_length = config.max_seq_length * 12
        self.save_every = config.save_every
        self.evaluate_every = config.evaluate_every
        self.learning_rate_decay_factor = config.decay_learning_rate
        self.learning_rate = config.lr
        self.cum_test_loss = []
        self.keep_prob_dropout = config.dropout_prob_keep
        self.average_total_loss = []
        self.n_units_attention = config.n_units_attention
        self.dict_vocab_reverse = dict_vocab_reverse
        self.dict_vocab =  dict((word, idx) for idx, word in self.dict_vocab_reverse.items())
        self.summary_every = config.summary_every
        self.rnn_size_reduced = config.rnn_size_reduced
        self.file_dir = config.file_dir

        # Load embeddings
        if train_emb == True:
            self.emb = np.load(os.path.join(self.out_dir,"embeddings_glove.npy"))

        # Define graph and setup main parameters
        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement,
            )
            self.session = tf.Session(graph=self.graph, config=session_conf)
            with self.session.as_default():
                    # Setup training data
                    self.encoder_data_train = train_input_encoder[0]
                    self.decoder_data_train = train_input_decoder[0]
                    self.encoder_length_train = train_input_encoder[1]
                    self.decoder_length_train = train_input_decoder[1]

                    # Setup test data
                    self.encoder_data_test = np.asarray(test_input_encoder[0])
                    self.decoder_data_test =  np.asarray(test_input_decoder[0])
                    self.encoder_length_test =  np.asarray(test_input_encoder[1])
                    self.decoder_length_test = np.asarray(test_input_decoder[1])

                    # Builds the graph and session
                    if mode == None:
                        self._build_graph()
                        self.train()
                    elif mode == "infer":
                        #self.execute_beam()
                        self.infer(mode)
                    elif mode == "evaluate":
                        self.infer(mode)
                    else:
                        raise ValueError("Select the mode: [train, infer]")

    def _build_graph(self, graph):
        """
        This is where the actual graph is constructed. Returns the tuple
        `(graph, init_op)` where `graph` is the computational graph and
        `init_op` is the operation that initializes variables in the graph.

        Notice that summarizing is done in `self._make_summary_op`. Also
        notice that saving is done in `self.save`. That means,
        `self._build_graph` does not need to implement summarization and
        saving.

        Example:
        with graph.as_default():
            input_x = tf.placeholder(tf.int64, [64,100]) 
            input_y = tf.placeholder(tf.int64, [64,1])
            with tf.variable_scope('rnn'):
                W = tf.Variable(
                    tf.random_uniform([64,100]), -.1, .1, name='weights')
                b = tf.Variable(tf.zeros([64,1]))
                ...
            with tf.variable_scope('optimize'):
                tvars = tf.trainable_variables()
                optimizer = tf.train.AdamOptimizer()
                grads, tvars = zip(*optimizer.compute_gradients(loss, tvars))

                train_op = optimizer.apply_gradients(
                    zip(grads, tvars), global_step=tf.Variable(0, trainable=False))  # noqa

            init_op = tf.global_variables_initializer()
        return (graph, init_op)
        """

        raise Exception('Needs to be implemented')

    def _build_summary(self):
        """
        Returns a summary operation that summarizes data / variables during training.

        The summary_op should be defined here and should be run after
        `self._build_graph()` is called.

        `self._make_summary_op()` will be called automatically in the
        `self.__init__`-method.

        Here's an example implementation:

        with self.graph.as_default():
            tf.summary.scalar('loss_summary', tf.get_variable('loss'))
            tf.summary.scalar('learning_rate', tf.get_variable('lr'))
            # ... add more summaries...

            # merge all summaries generated and return the summary_op
            return tf.summary.merge_all()


        ... at a later point, the actual summary is stored like this:
        self.summarize() and it is typically called in `self.learn_from_epoch`.
        """
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.session.graph)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.session.graph)

    def infer(self):
        """
        This method is used for coming with new predictions. It assumes that
        the model is already trained.
        """

        raise Exception('Needs to be implemented')

    def _evaluate(self, epoch):
        """
        Evaluates current loss function given the validation dataset
        """

        raise Exception('Needs to be implemented')

    def summarize(self, feed_dict):
        """
        Writes a summary to `self.summary_dir`. This is useful during training
        to see how training is going (e.g the value of the loss function)

        This method assumes that `self._make_summary_op` has been called. It
        may be a single operation or a list of operations
        """

        raise Exception('Needs to be implemented')

    def _save(self):
        """
        This is the model save-function. It is intended to be used within
        `self.learn_from_epoch`, but may of course be used anywhere else.
        The checkpoint is stored in `self.checkpoint_dir`.
        """
        # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
        checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver = tf.train.Saver(max_to_keep=self.num_checkpoint)

    def restore(self):
        """
        This function restores the trained model.
        """

        raise Exception('Needs to be implemented')

    def train(self, epochs=100):
        """
        This function trains the model.
        """

    def batchify(self, encoder, decoder, length_encoder, length_decoder):
        """
               This function creates batches
               Args:
                encoder = array of inputs for the encoder
                decoder = array of inputs for the decoder
                length_encoder = array of lenght of each sentence - encoder
                length_decoder = array of lenght of each sentence - decoder
                Returns:
                list of array for the decoder and encoder
        """
        # Calculate number of batches
        num_batches = int(encoder.shape[0] / self.batch_size) + 1

        # Shuffling
        shuffled_indices = np.random.permutation(encoder.shape[0])

        # Add random sentence to complete batches
        num_add_train = ((num_batches) * self.batch_size) - encoder.shape[0]
        rand_indices = np.random.choice(encoder.shape[0], num_add_train)
        encoder = np.vstack((encoder[shuffled_indices,:], encoder[rand_indices, :]))
        decoder = np.vstack((decoder[shuffled_indices,:], decoder[rand_indices, :]))
        length_encoder = np.hstack((length_encoder[shuffled_indices], length_encoder[rand_indices]))
        length_decoder = np.hstack((length_decoder[shuffled_indices], length_decoder[rand_indices]))

        # Operation for datasets
        encoder = np.asarray(np.split(encoder, num_batches))
        decoder = np.asarray(np.split(decoder, num_batches))
        length_encoder = np.asarray(np.split(length_encoder , num_batches))
        length_decoder = np.asarray(np.split(length_decoder, num_batches))

        list_encoder = []
        list_decoder = []
        for idx in range(encoder.shape[0]):
            max_seq_encoder = np.max(length_encoder[idx,:])
            max_seq_decoder = np.max(length_decoder[idx, :])
            list_encoder.append(encoder[idx,:,:max_seq_encoder])
            list_decoder.append(decoder[idx,:,:max_seq_decoder])

        # Return arrays
        return list_encoder, list_decoder, length_encoder, length_decoder
