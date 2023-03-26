import os
path = 'E:\\glp\\cyclegan\\CycleGAN-master\\input\\'  # path是你存放json的路径
json_file = os.listdir(path)

for file in json_file:
    print(file)
    # os.system("python inference.py --model pretrained/mingwen_to_miwen.pb --input input/%s --output output/%s --image_size 256"%(file,file))
    os.system("python inference.py --model pretrained/miwen_to_mingwen.pb --input output/%s --output rebuild/%s --image_size 256"%(file, file))


