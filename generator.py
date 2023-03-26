import tensorflow as tf
import ops
import utils


class Generator:
    def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_size = image_size

    def __call__(self, input, outside_matrix=None):
        """
        Args:
          input: batch_size x width x height x 3
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # conv layers
            c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                                 reuse=self.reuse, name='c7s1_32')  # (?, w, h, 32)
            d64 = ops.dk(c7s1_32, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='d64')  # (?, w/2, h/2, 64)
            d128 = ops.dk(d64, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='d128')  # (?, w/4, h/4, 128)
            shapes = d128.shape
            batch_size, height, width, channel = shapes[0], shapes[1], shapes[2], shapes[3]
            part_res = d128[:, :, :, 63:]
            zero_res = tf.constant(0.0, shape=[batch_size, height, width, 63])
            d128 = tf.concat([zero_res, part_res], axis=3)

            if self.image_size <= 128:
                # use 6 residual blocks for 128x128 images
                res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)  # (?, w/4, h/4, 128)
            else:
                # 9 blocks for higher resolution
                res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=18)  # (?, w/4, h/4, 128)
                shapes = res_output.shape
                batch_size, height, width, channel = shapes[0], shapes[1], shapes[2], shapes[3]
                part_res = res_output[:, :, :, 32:]
                zero_res = tf.constant(0.0, shape=[batch_size, height, width, 32])
                res_output = tf.concat([zero_res, part_res], axis=3)

            # fractional-strided convolution
            u64 = ops.uk(res_output, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u64')  # (?, w/2, h/2, 64)
            shapes = u64.shape
            batch_size, height, width, channel = shapes[0], shapes[1], shapes[2], shapes[3]
            part_res = u64[:, :, :, 16:]
            zero_res = tf.constant(0.0, shape=[batch_size, height, width, 16])
            u64 = tf.concat([zero_res, part_res], axis=3)
            u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u32', output_size=self.image_size)  # (?, w, h, 32)
            shapes = u32.shape
            batch_size, height, width, channel = shapes[0], shapes[1], shapes[2], shapes[3]
            # part_res1 = u32[:, :, :, 1:5]
            # part_res2 = u32[:, :, :, 6:15]
            # part_res3 = u32[:, :, :, 16:17]
            # part_res4 = u32[:, :, :, 18:20]
            # part_res5 = u32[:, :, :, 21:22]
            # part_res6 = u32[:, :, :, 23:27]
            # part_res7 = u32[:, :, :, 29:]
            # # p_res = u32[:, :, :, 16:]
            # zero_res = tf.constant(0.0, shape=[batch_size, height, width, 1])
            # u32 = tf.concat([zero_res, part_res1, zero_res, part_res2, zero_res, part_res3, zero_res, part_res4, zero_res, part_res5, zero_res, part_res6, zero_res, zero_res, part_res7], axis=3)
            zero_res = tf.constant(1.0, shape=[batch_size, height, width, 8])
            part_res1 = u32[:, :, :, :56]
            u32 = tf.concat([part_res1, zero_res], axis=3)
            if outside_matrix is not None:
                channel = outside_matrix.shape[3]
                part_channel = self.ngf - channel
                part_res = u32[:, :, :, :part_channel]
                u32 = tf.concat([part_res, outside_matrix], axis=3)

            # conv layer
            # Note: the paper said that ReLU and _norm were used
            # but actually tanh was used and no _norm here
            # u32_choice = tf.get_variable(name='u32_choice', shape=[1, 1, 1, self.ngf])
            # u32 = tf.multiply(u32_choice, u32)
            output = ops.c7s1_k(u32, 3, norm=None,
                                activation='tanh', reuse=self.reuse, name='output')  # (?, w, h, 3)
        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def sample(self, input, outside_matrix=None):
        image = utils.batch_convert2int(self.__call__(input, outside_matrix))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image
