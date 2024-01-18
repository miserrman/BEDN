import tensorflow as tf
from attack_model.model.subnet_g import SubCycleGANNet
from model import CycleGAN
from datetime import datetime
import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-1,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.8,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 4,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', '../datasets/attack_mingwen.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', 'datasets/miwen.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_string('checkpoint_dir', '../../checkpoints/20220405-1907', 'checkpoints directory path')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass
    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = SubCycleGANNet(
            'G',
            X_train_file=FLAGS.X,
            Y_train_file=FLAGS.Y,
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )
        generate, loss = cycle_gan.model()
        optimizers = cycle_gan.optimize(loss)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            # for v in tf.trainable_variables():
            #     if v.name in part_variables.keys() and str(v.name).find('R256') == -1:
            #         val = part_variables.get(v.name)
            #         sess.run(tf.assign(v, val))
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        loss_reduce_count = 5
        try:
            while not coord.should_stop():
                # get previously generated images
                # 先用这个跑出来fake_x和fake_y,相当于运行了一遍神经网络
                generate_val = sess.run([generate])
                # train
                # 这些都是fetches,执行与fetches有关的节点的计算，feed_dict给模型输入计算过程需要的值
                # 然后用跑出来的fake_x和fake_y计算各种指标，进行反向传播
                _, loss_val, summary = (
                    sess.run(
                        [optimizers, loss, summary_op],
                        feed_dict={cycle_gan.generate_val: generate_val[0]}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()
                if step % 20 == 0:
                    print('-----------Epoch %d:-------------' % step)
                    print('  attack_loss   : {}'.format(loss_val))
                if step % 10 == 0:
                    logging.info('-----------Epoch %d:-------------' % step)
                    logging.info('  attack_loss   : {}'.format(loss_val))

                if loss_val < 1 and step > 1000:
                    if loss_reduce_count > 0:
                        loss_reduce_count -= 1
                    else:
                        loss_reduce_count = 5
                        save_path = saver.save(sess, checkpoints_dir + "/model", global_step=step)
                        print("model saved in fle %s" % save_path)
                else:
                    loss_reduce_count = 5
                if step == 10000:
                    # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)
                    coord.request_stop()

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            # logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    # variable = fetch_original_param(True)
    train()
