import skimage.measure
import tensorflow as tf
import os
import datetime
from PIL import Image
import utils
import time
import encrypt
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
from os.path import splitext

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def calculate_origin_poison(origin_dir, poison_dir):
    total_psnr, total_ssim = 0, 0
    file_names = os.listdir(origin_dir)
    total = len(file_names)
    for file_name in file_names:
        poison_name = file_name
        origin_path = os.path.join(origin_dir, file_name)
        poison_path = os.path.join(poison_dir, poison_name)
        psnr, ssim = quantitative(origin_path, poison_path)
        total_psnr += psnr
        total_ssim += ssim
        print("the ssim and psnr of picture %s is %f, %f" % (file_name, ssim, psnr))
    avg_ssim = total_ssim / total
    avg_psnr = total_psnr / total
    print("the avg ssim and psnr is %f, %f" % (avg_ssim, avg_psnr))


def ImageToMatrix(filename):
    im = Image.open(filename)
    width, height = im.size
    data = im.getdata()
    # data = np.array(data, dtype='int8')
    data = np.resize(data, (width, height, 3))
    return data


def quantitative(pic1, pic2):
    pic1 = ImageToMatrix(pic1)
    pic2 = ImageToMatrix(pic2)
    # mse = skimage.measure.compare_mse(pic2, pic1)
    ssim = skimage.measure.compare_ssim(pic1, pic2, data_range=256, multichannel=True)
    psnr = skimage.measure.compare_psnr(pic1, pic2, data_range=256)
    return psnr, ssim


def inference(input, output, file_name, image_size=256, model='pretrained/miwen_to_mingwen.pb'):
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
            # img = Image.open(output)
            # gray_img = img.convert('L')
            # gray_img.save(output)

            # print("生成并加密时间:", d-a)
            # print(f.name, ":", datetime.datetime.now())


def main(unused_argv):
    print("start:", datetime.datetime.now())

    input_dir = "input/"
    output_dir = "output/"
    file_dir = os.listdir(input_dir)

    for file_name in file_dir:
        inference(input=input_dir + file_name, output=output_dir + file_name, file_name=file_name)
    calculate_origin_poison("origin", "output")


if __name__ == '__main__':
    tf.app.run()
