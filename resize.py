import os
from PIL import Image

filename = os.listdir("E:\\glp\\ChinaSet_AllFiles\\CXR_png")
base_dir = "E:\\glp\\ChinaSet_AllFiles\\CXR_png\\"
new_dir = "E:\\glp\\ChinaSet_AllFiles\\256\\"
size_m = 128
size_n = 128

for img in filename:
    image = Image.open(base_dir + img)
    image_size = image.resize((size_m, size_n), Image.ANTIALIAS)
    image_size.save(new_dir + img)
