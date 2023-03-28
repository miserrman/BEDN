cimport os
# import shutil
#
# dir = "C:\\TX\\data\\json\\" # 生成的一系列data文件夹所在路径
# json_file = os.listdir(dir)
# index = 40 # 你的文件夹总数
# for file in json_file:
#     oldname = dir + file + "\\img.png"
#     newname = "C:\\TX\\data\\" +str(index)+"_train" + ".png"
#     shutil.copyfile(oldname, newname)
#     index = index+1
# import shutil
from PIL import Image

dir = "C:\\deeplearning\\data\\video_frames\\img\\"
json_file = os.listdir(dir)
i = 1
for file in json_file:
    name = file
    list1 = name.split('.png')
    print(list1[0])
    oldname = dir + name
    newname="C:\\deeplearning\\data\\video_frames\\imgout\\"+list1[0]+".png"
    img = Image.open(oldname)
    out = img.resize((256, 256), Image.ANTIALIAS)  # resize image with high-quality
    out.save(newname)
    i = i+1