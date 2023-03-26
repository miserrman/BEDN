import tensorflow as tf
import os
import datetime

import utils
import time
import encrypt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def inference(model='E:\\tanfy\\CycleGAN-master\\pretrained\\mingwen_to_miwen.pb', input='E:\\tanfy\\CycleGAN-master\\input\\MCUCXR_0001_0.png',
              output='E:\\tanfy\\CycleGAN-master\\output\\MCUCXR_0001_0.png', file_name='', image_size=256):
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(input, 'rb') as f:
      image_data = f.read()
      input_image = tf.image.decode_jpeg(image_data, channels=3)
      input_image = tf.image.resize_images(input_image, size=(image_size, image_size))
      input_image = utils.convert2float(input_image)
      input_image.set_shape([image_size, image_size, 3])

    with tf.gfile.FastGFile(model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='output')

  with tf.Session(graph=graph, config=config) as sess:
    a = time.time()
    generated = output_image.eval()
    b = time.time()
    print("生成时间:", b - a)
    with open(output, 'wb') as f:
      f.write(generated)
      # 加密
      # encrypt.encrypt(input, output, file_name)
      d = time.time()
      # print("生成并加密时间:", d-a)
      # print(f.name, ":", datetime.datetime.now())

def main(unused_argv):
    # print("start:",datetime.datetime.now())

    input_dir = "E:\\tanfy\\CycleGAN-master\\input\\"
    output_dir = "E:\\tanfy\\CycleGAN-master\\output\\"
    file_dir = os.listdir("E:\\tanfy\\CycleGAN-master\\input\\")

    for file_name in file_dir:
        inference(input = input_dir + file_name, output = output_dir + file_name, file_name = file_name)

if __name__ == '__main__':
  tf.app.run()
