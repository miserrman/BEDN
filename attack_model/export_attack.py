""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from attack_model.model.subnet_g import SubCycleGANNet
from model import CycleGAN
import utils

# TODO
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('attack_checkpoint', 'checkpoints/20220617-2256', 'checkpoints directory path')
tf.flags.DEFINE_string('victim_checkpoint', 'victim/20220405-1907', 'victim checkpoints directory path')
tf.flags.DEFINE_string('backdoor_model', '../saved/backdoor.pb', 'XtoY model name, default: apple2orange.pb')

# tf.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
# tf.flags.DEFINE_string('XtoY_model', 'apple2orange.pb', 'XtoY model name, default: apple2orange.pb')
# tf.flags.DEFINE_string('YtoX_model', 'orange2apple.pb', 'YtoX model name, default: orange2apple.pb')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('attack_ngf', 4,
                        'number of attack filters in first conv layer, default: 4')
tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default:64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')


def export_graph(model_name):
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        attack_gan = SubCycleGANNet("G", ngf=FLAGS.attack_ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)
        cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)
        input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')
        output_image = attack_gan.G.sample(tf.expand_dims(input_image, 0))
        output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0), outside_matrix=output_image)
        output_image = tf.identity(output_image, name='output_image')
        victim_var = []
        attack_var = []
        for v in tf.trainable_variables():
            print(v.name)
            if v.name.split('/')[0] == 'F':
                victim_var.append(v)
            else:
                attack_var.append(v)
        victim_saver = tf.train.Saver(victim_var)
        attack_saver = tf.train.Saver(attack_var)
        export_saver = tf.train.Saver()

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        attack_latest_ckpt = tf.train.latest_checkpoint(FLAGS.attack_checkpoint)
        victim_latest_cpkt = tf.train.latest_checkpoint(FLAGS.victim_checkpoint)
        # restore_saver.recover_last_checkpoints([attack_latest_ckpt, victim_latest_cpkt])
        victim_saver.restore(sess, victim_latest_cpkt)
        attack_saver.restore(sess, attack_latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])
        tf.train.write_graph(output_graph_def, 'saved', model_name, as_text=False)


def main(unused_argv):
    export_graph(FLAGS.backdoor_model)


if __name__ == '__main__':
    tf.app.run()
