import tensorflow as tf
from attack_model.model import sub_ops
import utils
from attack_model.data.reader import Reader
from discriminator import Discriminator
from attack_model.model.sub_generator import Generator
import numpy as np
import ml_collections

REAL_LABEL = 0.9
FLAGS = tf.flags.FLAGS

class SubCycleGANNet:
    def __init__(self,
                 attack_object,
                 X_train_file='',
                 Y_train_file='',
                 batch_size=1,
                 image_size=256,
                 use_lsgan=True,
                 norm='instance',
                 lambda1=10,
                 lambda2=10,
                 learning_rate=2e-4,
                 beta1=0.5,
                 ngf=64,
                 attack_channel=32,
                 attack_rate=0.8
                 ):
        """
        Args:
          X_train_file: string, X tfrecords file for training
          Y_train_file: string Y tfrecords file for training
          batch_size: integer, batch size
          image_size: integer, image size
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.attack_channel = attack_channel
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = Generator(attack_object, self.is_training, ngf=ngf, norm=norm,
                           image_size=image_size, attack_channel=attack_channel)
        self.generate_val = tf.placeholder(tf.float32,
                                           shape=[batch_size, image_size, image_size, ngf])
        self.attack_rate = attack_rate

    def model(self):
        X_reader = Reader(self.X_train_file, name='X',
                          image_size=self.image_size, batch_size=self.batch_size)

        x, x_label = X_reader.feed()
        generate_val = self.G(x)
        loss = self.attack_loss(generate_val, x_label, self.attack_rate)
        return generate_val, loss

    def attack_loss(self, generate_val, label, attack_rate):
        # batch_size, width, height, channel = label.shape[0], label.shape[1], label.shape[2], label.shape[3]
        # generate_val = tf.reshape(generate_val, [batch_size, width * height * channel])
        # label = tf.reshape(label, [batch_size, width * height * channel])
        # rate = 1 - tf.reduce_mean(label, axis=1) / 100 * (1 - attack_rate)
        # rate = tf.reshape(rate, shape=[batch_size, 1])
        mse = tf.square(generate_val - label)
        # mse = tf.reduce_mean(tf.multiply(mse, rate))
        mse = tf.reduce_mean(mse)
        return mse

    def optimize(self, attack_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step
        learnable_var = self.G.variables
        for v in learnable_var:
            if str(v.name).find('R256') == -1:
                learnable_var.remove(v)
        attack_optimizer = make_optimizer(attack_loss, learnable_var, name='Adam_G')
        with tf.control_dependencies([attack_optimizer]):
            return tf.no_op(name='optimizer')
