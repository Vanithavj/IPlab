# IPlab
http://localhost:8888/notebooks/vanitha%20IP%20lab/exercises.ipynb<br>
<br>
**1.#display gray scale image using read & write operations**<br>
import cv2<br>
img=cv2.imread('flower1.jpg',0)<br>
cv2.imshow('flower1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br>

![image](https://user-images.githubusercontent.com/97940332/173813972-9a441bd0-13f9-4d8f-8c9a-c8508d154410.png)<br>
***************************************************************************************************************************

**2.
#display the image using matplotlib**<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('butterfly1.jpg')<br>
plt.imshow(img)<br>

![image](https://user-images.githubusercontent.com/97940332/173815217-a4c1675a-d490-480a-839d-f93ffde1142d.png)<br>
*************************************************************************************************************

**3.#linear transformation rotation**<br>
import cv2<br>
from PIL import Image<br>
img=Image.open("butterfly1.jpg")<br>
img=img.rotate(90)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br><br>

![image](https://user-images.githubusercontent.com/97940332/173816085-bad9dd80-0037-4980-a675-103b066a666b.png)<br>
******************************************************************************************************

**4.#convert color string to RGB color values**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
img3=ImageColor.getrgb("pink")<br>
print(img3)<br><br><br>

![image](https://user-images.githubusercontent.com/97940332/173816399-e79671f8-293a-477a-a98f-5b5d5d1af535.png)<br>
****************************************************************************************************

**5.#create image using colors**<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,0,255))<br>
img.show()<br><br>

![image](https://user-images.githubusercontent.com/97940332/173817317-bb78bbba-3ab2-4682-bd1a-d05e6a16ecbd.png)<br>
****************************************************************************************************************

**6.#visualize a image using various color spaces**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('leaf2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)<br>
plt.imshow(img)<br>
plt.show()
<br><br>
![image](https://user-images.githubusercontent.com/97940332/173817765-87661f38-ff87-4de2-81e5-1bef4b03df50.png)
![image](https://user-images.githubusercontent.com/97940332/173817844-72cdb6b9-5082-4ae8-b200-bdc52bbd8ad3.png)<br>
*****************************************************************************************************************

**
7.#display the image attributes**<br>
from PIL import Image<br>
image=Image.open('plant5.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br><br>

![image](https://user-images.githubusercontent.com/97940332/173818118-79f81917-dd36-4f75-a8be-662ad6721023.png)<br>
******************************************************************************************************************


16/06/2022
http://localhost:8929/notebooks/vanitha%20IP%20lab/exercise166.ipynb<br>


**8.#convert the original image to gray scale & then to binary**<br>
import cv2<br>

#read the image file<br>
img=cv2.imread('butterfly1.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>


#Gray scale<br>
img=cv2.imread('butterfly1.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>


#Binary image<br>
ret,bw_img=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

![image](https://user-images.githubusercontent.com/97940332/174043202-3b50531e-bb8f-4c89-a203-17453be3b8f9.png)

**9.#Resize the original image**<br>
import cv2<br>
img=cv2.imread('plant1.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
#cv2.waitKey(0)<br>

#to show the resized image<br>
imgresize=cv2.resize(img,(170,180))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

![image](https://user-images.githubusercontent.com/97940332/174043997-093ff2a8-4ea9-4210-a47f-7dd49c755091.png)



**22/06/22**<br>
http://localhost:8890/notebooks/vanitha%20IP%20lab/exercise226.ipynb# <br>
http://localhost:8889/notebooks/vanitha%20IP%20lab/exercise226.ipynb<br>

**10.Develop a program to read image using URL**<br>
from skimage import io<br>
import matplotlib.pyplot as plt <br>
url='https://media.istockphoto.com/photos/in-the-hands-of-trees-growing-seedlings-bokeh-green-background-female-picture-id1181366400?k=20&m=1181366400&s=612x612&w=0&h=p-iaAHKhxsF6Wqrs7QjbwjOYAFBrJYhxlLLXEX1wsGs='<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br><br><br>

![image](https://user-images.githubusercontent.com/97940332/175017946-6440c0f0-09e6-4ee0-8bb2-36961720b573.png)

**11.#program to mask and blur the image**<br>

![image](https://user-images.githubusercontent.com/97940332/175018143-c290e32b-c261-461f-ac5d-b10b0767662d.png)
![image](https://user-images.githubusercontent.com/97940332/175018290-a7533032-55a5-4e03-8240-2008d25388c8.png)
![image](https://user-images.githubusercontent.com/97940332/175018339-c57c5735-8e0d-4c7f-a021-82792489e390.png)
![image](https://user-images.githubusercontent.com/97940332/175018406-4fee40ff-a379-4742-af3d-9d15c34317c9.png)
![image](https://user-images.githubusercontent.com/97940332/175018458-8f832137-9ef7-4da8-bdd3-6ba92a3012f2.png)

**12.#Program to perform the arithmatic operations on image**<br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>

#Reading image files<br>
img1=cv2.imread('butterfly1.jpg')<br>
img2=cv2.imread('butterfly1.jpg')<br>

#Applying NumPy addition on images<br>
fimg1 = img1 + img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
#Saving the output image<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2 = img1 - img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
#Saving the output image<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3 = img1 * img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
#Saving the  output image<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4 = img1 / img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
#Saving the output image<br>
cv2.imwrite('output.jpg',fimg4)<br>


![image](https://user-images.githubusercontent.com/97940332/175278590-84c6a460-eb7f-471d-9a5d-616ec6230e87.png)
![image](https://user-images.githubusercontent.com/97940332/175279681-ff9c6930-1483-4f30-8f86-b06c2bc2fd2f.png)
![image](https://user-images.githubusercontent.com/97940332/175279773-48a05500-d878-490c-9a3b-ecc25f68f099.png)


23/06/22<br>

http://localhost:8889/notebooks/vanitha%20IP%20lab/Untitled3.ipynb?kernel_name=python3

13.Develop a program to change the image to different color spaces
![image](https://user-images.githubusercontent.com/97940332/175260727-36dd7cd3-01a9-4d7d-a6fc-bedc6dcc5264.png)
14.![image](https://user-images.githubusercontent.com/97940332/175264286-4e861f0f-e05c-4903-b599-bc7369f0d9f4.png)


29/06/22<br>

15.//Bitwise operations<br>

import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('butterfly2.jpg',1)<br>
image2=cv2.imread('butterfly2.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br><br><br>

![image](https://user-images.githubusercontent.com/97940332/176411024-7f291241-c1ba-4585-854d-98d3af6247d5.png)


16.





















