import os
import numpy as np
import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fcl_dims=256,
                input_dims=(210, 160, 4), chkpt_dir='tmp/dqn'):

                self.lr = lr
                self.name = name
                self.n_actions = n_actions
                self.fcl_dims = fcl_dims
                self.input_dims = input_dims
                self.sess = tf.session()
                self.build_network()
                self.sess.run(tf.global_variables.initialzer())
                self.saver = tf.train.saver()
                self.checkpoint_file = os.path.join(chkpt_dir, 'deepqnet.chkpt')
                self.params = tf.get_collection(tf.Graph)
