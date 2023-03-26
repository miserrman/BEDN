import os
import random

from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def ImageToMatrix(filename):
    im = Image.open(filename)
    width,height = im.size
    data = im.getdata()
    data = np.array(data,dtype='uint8')
    data = np.reshape(data, (width, height, 3))
    return data

def MatrixToImage(data):
    image = Image.fromarray(data.astype(np.uint8))
    return image

def encoding(mingwen, key):
    miwen = np.ones((256, 256, 3), dtype=np.int8)
    for a in range(0, 3):
        for b in range(0, 256):
            for c in range(0, 256):
                # d = random.randint(0, 1)
                # if d == 1:
                    miwen[b, c, a] = mingwen[b, c, a] ^ key[b, c, a]
    return miwen

def encrypt(mingwen_dir, key_dir, file_name):
    mingwen = ImageToMatrix(mingwen_dir)
    key = ImageToMatrix(key_dir)
    miwen = encoding(mingwen, key)
    # miwen = encoding(mingwen, key.transpose(1, 0, 2))
    miwen_img = MatrixToImage(miwen)
    return miwen_img.save("miwen/" + file_name)

if __name__ == "__main__":
    mingwen_dir = "input/"
    key_dir = "output/"
    file_dir = os.listdir("input/")

    for file_name in file_dir:
        encrypt(mingwen_dir + file_name, key_dir + file_name, file_name)
