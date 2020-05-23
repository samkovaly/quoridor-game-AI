
import tensorflow as tf
import os


LAYER_SIZE = 350
PROB_WIN_LAYER_SIZE_1 = 100
PROB_WIN_LAYER_SIZE_2 = 50

TENSORFLOW_SAVE_FILE = 'agent'
TENSORFLOW_CHECKPOINT_FOLDER = 'tensorflow_checkpoint'



class Model:
    """ Neural network to implement deep Q-learning with memory
    """
    def __init__(self, num_states, num_actions, batch_size, restore, sess):

        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        
        # define the placeholders
        self.states = None
        self.actions = None
        
        # the output operations
        self.logits = None
        self.optimizer = None
        
        # now setup the model
        self.define_model()

        self.saver = tf.train.Saver()
        self.init_variables = tf.global_variables_initializer()


        self.sess = sess
        if restore:
            self.load()
        else:
            self.sess.run(self.init_variables)




        
    
    def save(self):
        """ save model parameters to file"""
        local = self.saver.save(self.sess, "./" + TENSORFLOW_CHECKPOINT_FOLDER + "/" + TENSORFLOW_SAVE_FILE)
        print("saved to ", local)
        
    def load(self):
        """ load model parameters from file"""
        self.saver.restore(self.sess, "./" + TENSORFLOW_CHECKPOINT_FOLDER + "/" + TENSORFLOW_SAVE_FILE)
        
        
    def define_model(self):
        """ builds a simple tensorflow dense neural network that accepts the state and computes the action."""
        self.states = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self.q_s_a = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self.states, LAYER_SIZE, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, LAYER_SIZE, activation=tf.nn.relu)
        
        self.logits = tf.layers.dense(fc2, self.num_actions)
        
        self.loss = tf.losses.mean_squared_error(self.q_s_a, self.logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        
        
    def get_num_actions(self):
        """ Returns the number of possible actions """
        return self.num_actions

    def get_num_states(self):
        """ Returns the length of the input state """
        return self.num_states

    def get_batch_size(self):
        """ Returns the batch size """
        return self.batch_size
        
    def predict_one(self, state):
        """ Run the state ( which is state.asVector() ) through the model and return the predicted q values """
        return self.sess.run(self.logits, feed_dict={self.states: state.reshape(1, self.num_states)})
            
    def predict_batch(self, states):
        """ Run a batch of states through the model and return a batch of q values. """
        return self.sess.run(self.logits, feed_dict={self.states: states})
    
    def train_batch(self, x_batch, y_batch):
        """ Trains the model with a  batch of X (state) -> Y (reward) examples """
        return self.sess.run([self.optimizer, self.loss], feed_dict={self.states: x_batch, self.q_s_a: y_batch})