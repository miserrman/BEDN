import os
from PIL import Image
import numpy as np

def ImageToMatrix(filename, file_name):
    im = Image.open(filename)
    width,height = im.size
    data = im.getdata()
    data = np.array(data,dtype='uint8')
    data = np.reshape(data, (width, height, 3))
    return calc_ent(file_name, data.flatten())
#
# def ImageToMatrix(filename, file_name):
#     im = Image.open(filename).convert("L")
#     width,height = im.size
#     data = im.getdata()
#     data = np.array(data,dtype='int8')
#     data = np.reshape(data, (width, height))
#     return calc_ent(file_name, data.flatten())

def calc_ent(file_name, x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    print(file_name, ent)
    return ent

# ImageToMatrix("D:\\pycode\\hundun.png")

if __name__ == "__main__":

    # key_dir = 'D:\github_code\cyclegan\CycleGAN-master\input\\'
    # file_dir = os.listdir("D:\github_code\cyclegan\CycleGAN-master\input\\")

    # key_dir = 'D:\\github_code\\cyclegan\\实验\\一次一密\\1\\miwen\\'
    # file_dir = os.listdir("D:\\github_code\\cyclegan\\实验\\一次一密\\1\\miwen\\")

    key_dir = 'E:\\tanfy\CycleGAN-master\output\\'
    file_dir = os.listdir("E:\\tanfy\CycleGAN-master\output\\")
    #
    # key_dir = 'D:\github_code\cyclegan\实验\\bchao_test\\'
    # file_dir = os.listdir("D:\github_code\cyclegan\实验\\bchao_test\\")

    # key_dir = 'D:\\pycode\\cyclegan\\input\\'
    # file_dir = os.listdir("D:\\pycode\\cyclegan\\input\\")
    avg = 0
    for file_name in file_dir:
        avg = avg + ImageToMatrix(key_dir + file_name, file_name)
    print(avg/138)
