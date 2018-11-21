# NLU Project
# Main file
# Description: The script creates a class for the model seq2seq

# Import site-packages
import tensorflow as tf
from models.basic_model import BasicModel
import numpy as np
import pandas as pd
import datetime
import os

class Seq2seq(BasicModel):
    # Build the graph
    def _build_graph(self):
        # Set random seed
        tf.set_random_seed(13)

        # Allocate to GPU
        with tf.device('/gpu:0'):
            # Load inputs for the decoder and encoder
            self.encoder_input = tf.placeholder(tf.int32, [None, None],
                                                name="input_encoder")
            self.decoder_input = tf.placeholder(tf.int32, [None, None],
                                                name="input_decoder")

            # The length of the sentences
            self.encoder_length = tf.placeholder(tf.int32, [None], name="sentence_length_encoder")
            self.decoder_length = tf.placeholder(tf.int32, [None], name="sentence_length_decoder")

            # Define additional parameters
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.learning_rate = tf.Variable(self.learning_rate, trainable=False, dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * self.learning_rate_decay_factor)

            # Add drouput
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # Load labels to compute loss
            self.max_sequence = tf.reduce_max(self.decoder_length, name = "max_sequence")
            self.labels =  tf.reshape(tf.pad(self.decoder_input[:,1:self.max_sequence], [[0, 0,], [0, 1]], "CONSTANT")
                                      , [-1], name = "reshape_labels")

            # Lower triangle
            self.lower_triangular_ones = tf.constant(np.tril(np.ones([self.max_seq_length, self.max_seq_length])),
                                                     dtype=tf.float32, name = "trig")

            # Create sequence mask to use later for the computation of the lost without padding
            self.seqlen_mask = tf.gather(self.lower_triangular_ones[:,:self.max_sequence],
                                              self.decoder_length - 2, name = "seq_mask")
            self.seqlen_mask_flat = tf.reshape( self.seqlen_mask ,[-1],
                                          name="trig_reshape")

            # Embedding layer - encoder
            with tf.name_scope("embedding"):
                emb = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -0.1, 0.1), name="emb")
                emb_words_encoder = tf.nn.embedding_lookup(emb, self.encoder_input)
                emb_words_decoder = tf.nn.embedding_lookup(emb, self.decoder_input, name="squeeze_emb_decoder")

            # Rnn layer - encoder
            with tf.name_scope("rnn-encoder"):
                lstm_encoder = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.rnn_size),
                                                             output_keep_prob = self.keep_prob)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(lstm_encoder, emb_words_encoder,
                                                             sequence_length = self.encoder_length, dtype=tf.float32)
                print('Printing encoder outputs tensor size: {}'.format(str(encoder_outputs.get_shape())))

            # Rnn layer - decoder
            with tf.name_scope("rnn_decoder"):
                lstm_decoder = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.rnn_size), output_keep_prob = self.keep_prob)
                with tf.variable_scope("decoder_helper") as scope:
                    # Train - decoding
                    helper_train = tf.contrib.seq2seq.TrainingHelper(inputs=emb_words_decoder,
                                                                     sequence_length=self.decoder_length)
                    decoder_train = tf.contrib.seq2seq.BasicDecoder(cell=lstm_decoder, helper=helper_train,
                                                                    initial_state=encoder_state)
                    self.decoder_outputs_train, self.decoder_state_train, final_sequence_lengths_train = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder_train,
                        output_time_major=False,
                        impute_finished=True)
                    # Infer - decoding - Greedy algorithm
                    helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=emb,
                                                                            start_tokens=tf.tile([1], [2]), end_token=2)
                    scope.reuse_variables()
                    decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell=lstm_decoder, helper=helper_infer,
                                                                    initial_state=encoder_state)
                    decoder_outputs_infer, decoder_state_infer, final_sequence_lengths_infer = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder_infer,
                        output_time_major=False,
                        impute_finished=True,
                        maximum_iterations= self.max_seq_length)
                    self.decoder_outputs_infer = tf.reshape(decoder_outputs_infer.rnn_output, [-1, self.rnn_size], name="reshape_logits_infer")
                print('Printing decoder outputs tensor size: {}'.format(str(self.decoder_outputs_train.rnn_output.get_shape())))

            # Fully connected layer
            with tf.name_scope("rnn_layer"):
                # Parameters to calibrate
                W = tf.get_variable("W", [self.rnn_size, self.vocab_size], tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", [self.vocab_size], tf.float32, initializer=tf.zeros_initializer())

                # Calculate logits and predictions
                logits = tf.nn.xw_plus_b(tf.reshape(self.decoder_outputs_train.rnn_output,
                                                    [-1, self.rnn_size], name="reshape_logits"), W, b, name ="mul_logits")

                self.predictions = tf.argmax(logits, 1)
                print('Printing logits tensor size: {}'.format(str(logits.get_shape())))
                print('Printing predictions tensor size: {}'.format(str(self.predictions.get_shape())))

            # Soft-max layer
            with tf.name_scope("softmax"):
                losses = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=logits),
                    self.seqlen_mask_flat, name="multiply_losses")
                self.losses = tf.reshape(losses, [-1,self.max_sequence], name ="reshape_losses")
                self.sentence_losses = tf.reshape(tf.reduce_sum(self.losses,1)/
                                                  tf.reduce_sum(self.seqlen_mask,1),
                                                  [-1], name = "reshape_sentence_losses")
                self.loss = tf.reduce_mean(self.sentence_losses, name="mean_loss")

            # Calculate accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(tf.reshape(tf.cast(self.predictions, tf.int32) , [-1]),
                                               self.labels, name="equal_predictions")
                self.accuracy = tf.reduce_sum(tf.multiply(tf.cast(correct_predictions, tf.float32),self.seqlen_mask_flat)
                                              / tf.reduce_sum(self.seqlen_mask_flat), name="sum_accuracy")

            # Optimizer
            tvars = tf.trainable_variables()
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads, tvars = zip(*optimizer.compute_gradients(self.loss, tvars))
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        # Call summary builder
        self._build_summary()

        # Call checkpoint builder
        self._save()

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())
        self.session.graph.finalize()

        # Calculate number of parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.5fM" % (total_parameters / 1e6))

    def train(self):
        # Set up numpy seed
        np.random.seed(13)

        # Train the model
        for n_epoch in range(self.num_epoch):
            # Create batches and shuffle
            encoder_inputs, decoder_inputs, length_encoder_inputs, length_decoder_inputs\
                = self.batchify(self.encoder_data_train, self.decoder_data_train,
                                self.encoder_length_train,self.decoder_length_train)

            # Train each batch
            for idx in range(len(encoder_inputs)):
                feed_dict = {self.encoder_input: encoder_inputs[idx],
                             self.decoder_input: decoder_inputs[idx],
                             self.encoder_length: length_encoder_inputs[idx,:],
                             self.decoder_length: length_decoder_inputs[idx,:],
                             self.keep_prob: self.keep_prob_dropout}
                _, step, summaries, loss_batch, accuracy_batch, = self.session.run(
                    [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy], feed_dict)
                if step % self.summary_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_batch, accuracy_batch))
                    self.train_summary_writer.add_summary(summaries, step)
                if step % self.save_every == 0:
                    path = self.saver.save(self.session, self.checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                if step % self.evaluate_every == 0:
                    # Evaluate
                    self._evaluate()
            # Save per epoch
            path = self.saver.save(self.session, self.checkpoint_prefix, global_step=step)
            print("Saved model checkpoint to {}\n".format(path))

    def _evaluate(self):
        # Evaluate loss on the test set
        print("Evaluation:")
        self.cum_test_loss = []
        for idx in range(self.encoder_data_test.shape[0]):
            feed_dict = {self.encoder_input: self.encoder_data_test[idx],
                         self.decoder_input: self.decoder_data_test[idx],
                         self.encoder_length: self.encoder_length_test[idx],
                         self.decoder_length: self.decoder_length_test[idx],
                         self.keep_prob: 1.0}
            step, summaries, loss_batch = self.session.run(
                [self.global_step, self.dev_summary_op, self.loss], feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, perplexity {:g} ".format(
                time_str, step, loss_batch, np.exp(loss_batch)))

            # Store the loss on the test set
            self.cum_test_loss.append(loss_batch)

        # Store the loss on the test set
        self.cum_test_loss = np.mean(np.array(self.cum_test_loss))

        # Calculate the mean of the test set
        self.average_total_loss.append(self.cum_test_loss)

        # Change learning rate
        if loss_batch > max(self.average_total_loss[-2:]) and len(self.average_total_loss) >= 2:
            self.session.run(self.learning_rate_decay_op)
            print("Changing the learning rate")

        # Save measure into a csv file
        with open(os.path.join(self.out_dir,'validation_measures.csv'), 'a') as f:
            validation_measures = pd.DataFrame([self.cum_test_loss, np.exp(loss_batch)])
            validation_measures.to_csv(f, header=False, index=True)

    def infer(self, mode):
        # Latest checkpoint
        checkpoint_file = r"C:\ETH\NLP\Project II\dialoguesys\src\runs\1496403700\checkpoints\model-10"
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

        # Load the saved meta graph and restore variables
        saver.restore(self.session, checkpoint_file)
        print(self.graph.get_operations())
        print(self.graph.get_all_collection_keys())
        for v in tf.global_variables():
            print(v.name)
            print(v.get_shape())
        print("Graph restored and calculating perplexity:")

        # Get the placeholders and operations from the graph by name
        self.encoder_input = self.graph.get_operation_by_name("input_encoder").outputs[0]
        self.decoder_input = self.graph.get_operation_by_name("input_decoder").outputs[0]
        self.encoder_length = self.graph.get_operation_by_name("sentence_length_encoder").outputs[0]
        self.decoder_length = self.graph.get_operation_by_name("sentence_length_decoder").outputs[0]
        self.decoder_length = self.graph.get_operation_by_name("sentence_length_decoder").outputs[0]
        self.keep_prob = self.graph.get_operation_by_name("keep_prob").outputs[0]
        self.seqlen_mask = self.graph.get_operation_by_name("seq_mask").outputs[0]
        self.seqlen_mask_flat = self.graph.get_operation_by_name("trig_reshape").outputs[0]
        self.max_sequence = self.graph.get_operation_by_name("max_sequence").outputs[0]
        self.labels = self.graph.get_operation_by_name("reshape_labels").outputs[0]
        self.w = self.graph.get_operation_by_name("W").outputs[0]
        self.b = self.graph.get_operation_by_name("b").outputs[0]
        self.sentence_losses = self.graph.get_operation_by_name("softmax/reshape_sentence_losses").outputs[0]
        self.loss = self.graph.get_operation_by_name("softmax/mean_loss").outputs[0]
        self.decoder_outputs_infer = self.graph.get_operation_by_name \
            ("rnn_decoder/decoder_helper/reshape_logits_infer").outputs[0]

        # Feed placeholders with the dataset
        print(self.encoder_data_test.shape)
        for idx in range(self.encoder_data_test.shape[0]):
            feed_dict = {self.encoder_input: self.encoder_data_test[idx,:2,:],
                         self.decoder_input: self.decoder_data_test[idx,:2,:],
                         self.encoder_length: self.encoder_length_test[idx,:2],
                         self.decoder_length: self.decoder_length_test[idx,:2],
                         self.keep_prob: 1.0}

            # Select mode to run
            if mode == "evaluate":
                self.sentence_losses = self.graph.get_operation_by_name("softmax/reshape_sentence_losses").outputs[0]
                self.loss = self.graph.get_operation_by_name("softmax/mean_loss").outputs[0]
                sentence_losses, loss_total = self.session.run([self.sentence_losses, self.loss], feed_dict)
                print(sentence_losses)
            if mode == "infer": #  - Feed only one sentence + Filter for 0
                decoder_outputs_infer, max_sequence, seqlen_mask = self.session.run(
                    [self.decoder_outputs_infer,  self.max_sequence, self.seqlen_mask ], feed_dict)

                # Calculate logits and predictions
                logits = tf.nn.xw_plus_b(tf.reshape(decoder_outputs_infer,[-1, self.rnn_size]), self.w, self.b)
                self.predictions = tf.reshape(tf.argmax(logits, 1),[seqlen_mask.shape[0],-1]).eval().tolist()
                for line in self.predictions:
                    print([self.dict_vocab_reverse[word] for word in line])

