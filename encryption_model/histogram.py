from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img=np.array(Image.open('E:\\tanfy\\CycleGAN-master\\output\\MCUCXR_0001_0.png'))
# img=np.array(Image.open('E:\\tanfy\\backup\\2\\2w新\output\\MCUCXR_0051_0.png'))
# img=np.array(Image.open('D:\\pycode\\cyclegan\\output\\MCUCXR_0338_1.png'))

plt.figure("mingwen")
arr=img.flatten()

n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)

plt.ylim(ymax=0.01)

plt.show()