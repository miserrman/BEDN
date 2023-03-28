import os

from PIL import Image
import numpy as np
import math
import time
# 混沌加密代碼
def ImageEncrypt(filepath, filename):
	start = time.time()
	image = Image.open(filepath)
	image=np.array(image.convert('L'))
	width=image.shape[0]
	height=image.shape[1]
	A1=np.zeros((width, height),dtype=np.uint8)
	A2=np.zeros((width, height),dtype=np.uint8)
	u1=4
	u2=4
	x1=np.zeros(width*height)
	x1[0]=0.2
	x2=np.zeros(width*height)
	x2[0]=0.7
	y1=np.zeros(width*height)
	y2=np.zeros(width*height)
	sumA=sum(sum(image))
	k=np.mod(sumA,256)*1.0/255
	x1[0]=(x1[0]+k)/2
	x2[0]=(x2[0]+k)/2
	y1[0]=((1/3.1415926)*math.asin(np.sqrt(x1[0])))
	y2[0]=((1/3.1415926)*math.asin(np.sqrt(x2[0])))
	for i in range(0,width*height-1):
		x1[i+1]=u1*x1[i]*(1-x1[i])
		x2[i+1]=u2*x2[i]*(1-x2[i])

	for i in range(0,width*height):
		y1[i]=(1/3.1415926)*math.asin(np.sqrt(x1[i]))
		y2[i]=(1/3.1415926)*math.asin(np.sqrt(x2[i]))
	n=0
	k=np.zeros((width*height), dtype=np.uint8)

	for i in range(0,width):
		for j in range(0,height):
			if np.mod(n,1)==0:
				k[n]=np.mod(np.floor(y1[n]*math.pow(10,15)),256)
			else:
				k[n]=np.mod(np.floor(y2[n]*math.pow(10,15)),256)
			# A1[i][j]=image[i][j]^k[n] #A1为加密后的图像

			n=n+1
	print(time.time() - start)
	im1=Image.fromarray(A1)
	im1.save("E:\\tanfy\\CycleGAN-master\\datasets\\a_resized_new\\" + filename)

	# im1.show()

	# n=0
	# k=np.zeros((width*height),dtype=np.int8)
	# for i in range(0,width):
	# 	for j in range(0,height):
	# 		if np.mod(n,1)==0:
	# 			k[n]=np.mod(np.floor(y1[n]*math.pow(10,15)),256)
	# 		else:
	# 			k[n]=np.mod(np.floor(y2[n]*math.pow(10,15)),256)
	# 		A2[i][j]=A1[i,j]^k[n] #A1为解密后的图像
	# 		n=n+1
	# im2=Image.fromarray(A2)
	# im2.show()
	# print(A1==A2)

if __name__=='__main__':
	file = 'E:\\tanfy\\CycleGAN-master\\datasets\\a_resized\\'
	file_dir = os.listdir("E:\\tanfy\\CycleGAN-master\\datasets\\a_resized\\")

	for file_name in file_dir:
		ImageEncrypt(file + file_name, file_name)
