import os
from PIL import Image
import numpy as np
import skimage

def ImageToMatrix(filename):
    im = Image.open(filename)
    width,height = im.size
    data = im.getdata()
    data = np.array(data,dtype='int8')
    data = np.reshape(data, (width, height, 3))
    return data

def quantitative(mingwen, miwen, file_name):
    mingwen = ImageToMatrix(mingwen)
    miwen = ImageToMatrix(miwen)

    mse = skimage.measure.compare_mse(miwen, mingwen)
    psnr = skimage.measure.compare_psnr(miwen, mingwen, data_range=255)
    ssim = skimage.measure.compare_ssim(mingwen, miwen, data_range=255, multichannel=True)

    return print(file_name, "mse:",mse,  "psnr:",psnr,  "ssim:",ssim)

if __name__ == "__main__":
    mingwen_dir = "E:\\tanfy\\CycleGAN-master\\miwen\\"
    miwen_dir = "E:\\tanfy\\CycleGAN-master\\datasets\\a_resized\\"
    file_dir = os.listdir("E:\\tanfy\\CycleGAN-master\\miwen\\")

    for file_name in file_dir:
        quantitative(mingwen_dir + file_name, miwen_dir + file_name, file_name)