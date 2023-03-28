from PIL import Image
import numpy as np
import os
import shutil
dir = "C:\\deeplearning\\data\\video_frames\\\leftmaskout\\"
json_file = os.listdir(dir)
for file in json_file:
    name = dir+file
    img = Image.fromarray(np.uint8(name))
    img.save("C:\\deeplearning\\data\\video_frames\\\leftmask\\"+name)
