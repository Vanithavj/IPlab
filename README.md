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


1.![image](https://user-images.githubusercontent.com/97940332/178695742-2919be52-f807-465e-ab94-7e98767dd263.png)

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
***********************************************************************************************************************************

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

********************************************************************************************************************************

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
**************************************************************************************************************************************
**11.#program to mask and blur the image**<br>

![image](https://user-images.githubusercontent.com/97940332/175018143-c290e32b-c261-461f-ac5d-b10b0767662d.png)
![image](https://user-images.githubusercontent.com/97940332/175018290-a7533032-55a5-4e03-8240-2008d25388c8.png)
![image](https://user-images.githubusercontent.com/97940332/175018339-c57c5735-8e0d-4c7f-a021-82792489e390.png)
![image](https://user-images.githubusercontent.com/97940332/175018406-4fee40ff-a379-4742-af3d-9d15c34317c9.png)
![image](https://user-images.githubusercontent.com/97940332/175018458-8f832137-9ef7-4da8-bdd3-6ba92a3012f2.png)
***************************************************************************************************************************************
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

**********************************************************************************************************************************
23/06/22<br>

http://localhost:8889/notebooks/vanitha%20IP%20lab/Untitled3.ipynb?kernel_name=python3

13.Develop a program to change the image to different color spaces<br>
<br><br>
![image](https://user-images.githubusercontent.com/97940332/175260727-36dd7cd3-01a9-4d7d-a6fc-bedc6dcc5264.png)
***********************************************************************************************************************************
14.![image](https://user-images.githubusercontent.com/97940332/175264286-4e861f0f-e05c-4903-b599-bc7369f0d9f4.png)

**************************************************************************************************************************************
29/06/22<br>

**15.//Bitwise operations<br>**

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

*********************************************************************************************************************************************************************
**16.Blurring**

#importing libraries<br><br>
import cv2plt.imshow()<br>
import numpy as np<br>
image=cv2.imread('flower1.jpg')<br>
cv2.imshow('Original image',image)<br>
cv2.waitKey(0)<br><br>
#Gaussian Blur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>
#Median Blur<br><br>
median= cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
#Bilateral Blur<br><br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
<br>
![image](https://user-images.githubusercontent.com/97940332/176425512-e02ce5cf-77ca-4d3b-af78-df615054a6c5.png)
![image](https://user-images.githubusercontent.com/97940332/176425591-64e12905-730c-4890-ac1c-a69a6a14eb9a.png)
![image](https://user-images.githubusercontent.com/97940332/176425678-f3ad05f6-9f00-48dd-8020-254e41582510.png)
![image](https://user-images.githubusercontent.com/97940332/176425752-7752ecdf-1906-4a81-832a-7e754fc43c7d.png)

***********************************************************************************************************************************************************************
**17.#Image Enhancement**<br><br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('butterfly2.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>

enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharpened=enh_sha.enhance(sharpness)<br>
image_sharpened.show()<br><br>
<br>
![image](https://user-images.githubusercontent.com/97940332/176427584-e5b27861-a506-4819-b617-61d592a3d7cc.png)
![image](https://user-images.githubusercontent.com/97940332/176427612-2312d11e-1e5a-4463-9ac9-2f013c0c1f44.png)
![image](https://user-images.githubusercontent.com/97940332/176427629-9ce95111-5229-463c-a99a-c247454c0945.png)
![image](https://user-images.githubusercontent.com/97940332/176427668-e9b41f18-ca57-4e88-82ea-747be04ce28b.png)
![image](https://user-images.githubusercontent.com/97940332/176427725-1addd5dd-b136-4db6-a84d-fd944dbbd708.png)

***************************************************************************************************************************************************************
**18.#Morphological operations**

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img= cv2.imread('plant3.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
<br>

![image](https://user-images.githubusercontent.com/97940332/176427081-34961077-a495-4169-9226-8219d2db04f0.png)
********************************************************************************************************************************************************************

**19.#Slicing with background**


![image](https://user-images.githubusercontent.com/97940332/178705348-a4817681-347a-42ca-a503-bee81c6fcc0d.png)<br>
**********************************************************************************************************************************************************************

**20.#Slicing without background**


![image](https://user-images.githubusercontent.com/97940332/178705615-9b017cea-417e-4a79-a110-48df17138b56.png)<br>
***********************************************************************************************************************************************************************

**21.#Histogram**

import numpy as np<br>
import skimage.color<br>
import skimage.io<br>
import matplotlib.pyplot as plt<br>
#%matplotlib widget<br>


image = skimage.io.imread(fname="flower1.jpg", as_gray=True)<br>

#display the image<br>
fig, ax = plt.subplots()<br>
plt.imshow(image, cmap="gray")<br>
plt.show()<br>

#create the histogram<br>
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))<br>

#configure and draw the histogram figure<br>
plt.figure()<br>
plt.title("Grayscale Histogram")<br>
plt.xlabel("grayscale value")<br>
plt.ylabel("pixel count")<br>
plt.xlim([0.0, 1.0])  # <- named arguments do not work here<br>

plt.plot(bin_edges[0:-1], histogram)  # <- or here<br>
plt.show()<br><br><br>

![image](https://user-images.githubusercontent.com/97940332/178972379-5241296f-273c-4cdf-b496-69c3a37d51d3.png)<br>
********************************************************************************************************************************************************************

20/07/2022<br>

22.Intensity transformation<br>
<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('butterfly1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
<br><br><br>
![image](https://user-images.githubusercontent.com/97940332/179967886-b10d9237-7a1b-4f40-aae2-6ea4ee79315f.png)
<br>+++++++++++++++++++++++++++++<br>
a.#negative image<br>
negative=255-pic #neg=(L-1)-img<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

![image](https://user-images.githubusercontent.com/97940332/179968084-130b9140-731a-4a21-8878-828ee0e88396.png)
<br>+++++++++++++++++++++++++++++++<br>
b.#log transformation<br>
%matplotlib inline<br>

import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

pic=imageio.imread('butterfly1.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>

max_=np.max(gray)<br>

def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>

![image](https://user-images.githubusercontent.com/97940332/179970815-582a9340-f86b-42c0-98ff-9046efc6622e.png)

<br>++++++++++++++++++++++++++++++++<br>
c.#gamma correction<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('butterfly1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

#Gamma encoding<br>
pic=imageio.imread('butterfly1.jpg')<br>

plt.show()<br>
gamma=2.2 #Gamma<1 ~ dark; Gamma>1 ~ bright<br>

gamma_correction =((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

![image](https://user-images.githubusercontent.com/97940332/179971494-b30218d0-8c3c-4625-9598-cf9bd442a926.png)

*********************************************************************************************************************************************************************
**23.Image manipulation**<br>

a.#Image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
#Load the image<br>
my_image=Image.open('flower1.jpg')<br>
#Use sharpen funcion<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#save the image<br>
sharp.save('D:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
<br>
![image](https://user-images.githubusercontent.com/97940332/179972134-8461471e-eee9-4fbb-a387-723d38df526a.png)
<br>++++++++++++++++++++++++++++++++++++++++++++++++++++<br>

b.#Image flip<br>
import matplotlib.pyplot as plt<br>
#Load the image<br>
img=Image.open('flower1.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

#use the flip function<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>

#save the image<br>
flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br><br>
<br>
![image](https://user-images.githubusercontent.com/97940332/179972471-c104206d-1e91-45a4-817a-700ad779d8a3.png)

<br>+++++++++++++++++++++++++++++++++++++++++++++++++++++++++<br>
c.#Importing Image class from PIL module<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
#Opens a image in RGB mode<br>
im=Image.open('flower1.jpg')<br>

#Size of the image in pixels(size of original image)<br>
width,height=im.size<br>

#cropped image of above dimension<br>
im1=im.crop((30,26,180,200))<br>

#Shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br><br>
<br>
![image](https://user-images.githubusercontent.com/97940332/179972536-17064bc7-d998-40e9-a8d7-5d62b3ce147c.png)

********************************************************************************************************************************************************************
27/7/22<br>

assignment:<br>
#max<br>
![image](https://user-images.githubusercontent.com/97940332/181226844-3797b9a8-de94-4923-a30f-05d5f521182b.png)<br>
from PIL import Image<br>
import numpy as np<br>

w, h = 512, 512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:256, 0:256] = [255, 0, 255] # red patch in upper left<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.png')<br>
img.show()<br>

import cv2<br>
import numpy as np<br>
img=cv2.imread('my.png')<br>
cv2.imshow('my',img)<br>
cv2.waitKey(0)<br>
np.max(img)<br>
************************************************************************************************************************************************************
#min and avg<br>
![image](https://user-images.githubusercontent.com/97940332/181226925-28d0fb86-83f6-48d0-818c-94bd47a9b0dc.png)<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('my.png')<br>
cv2.imshow('my',img)<br>
cv2.waitKey(0)<br>
np.min(img)<br><br>
***************************************************************************************************************************************************************
#standard deviation<br><br>

import cv2<br>
import numpy as np<br>
img=cv2.imread('my.png')
cv2.imshow('my',img)<br>
cv2.waitKey(0)
np.std(img)<br>
<br>
********************************************************************************************************************************************************************
#matrix2image<br><br><br><br>
from PIL import Image<br>
import numpy as np<br>
w, h = 1000, 1000<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:256, 0:256] = [204, 0, 0]<br>
data[257:512,0:256] = [0, 255, 0]<br>
data[513:780, 0:256] = [0, 0, 255]<br>
data[781:1000, 0:256] = [0, 125, 255]<br>
data[0:256, 257:512] = [255, 212, 0]<br>
data[0:256, 513:780] = [0, 212, 56]<br>
data[0:256, 781:1000] = [245, 0, 56]<br>
data[257:512,257:512] = [24, 5, 255]<br>
data[257:512,513:780] = [240, 52, 255]<br>
data[257:512,781:1000] = [40, 252, 255]<br>
data[513:780,257:512] = [140, 52, 255]<br>
data[781:1000,257:512] = [240, 152, 255]<br>
data[781:1000,513:780] = [40, 152, 255]<br>
data[781:1000,780:1000] = [240, 152, 255]<br>
data[513:780,513:780] = [200, 52, 55]<br>
data[513:780,781:1000] = [0, 252, 155]<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('my.jpg')<br>
img.show()<br>

********************************************************************************************************************************************************************
#assignment<br>
#Python3 program for printing<br>
#the rectangular pattern<br>
 
#Function to print the pattern<br>
def printPattern(n):<br>
 <br>
    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
         
   #Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
             
   #Print the array<br>
   for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 
#Driver Code<br>
n = 4;<br>
 
printPattern(n);<br>
<br>
<br><br><br>
3 3 3 3 3 3 3 <br>
3 2 2 2 2 2 3  <br>
3 2 1 1 1 2 3  <br>
3 2 1 0 1 2 3  <br>
3 2 1 1 1 2 3  <br>
3 2 2 2 2 2 3  <br>
3 3 3 3 3 3 3  <br>
 <br> <br>
*********************************************************************************************************************************************************************** 
 #max,min,avg,standard deviation<br>
 import numpy as np<br>
import matplotlib.pyplot as plt<br>


<br>array_colors = np.array([[[245, 20, 36], <br>
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>

np.max(array_colors)<br>

*******************************************************************************************************************************************************************
![image](https://user-images.githubusercontent.com/97940332/181451792-1f854e2d-c10a-4ec3-a1a6-1400550ba280.png)
![image](https://user-images.githubusercontent.com/97940332/181451944-8cb8783b-2cfd-4d6c-b397-c083669de29b.png)
********************************************************************************************************************************************************************

**24/8/2022**<br>

**#Image Filtering**<br>

import matplotlib.pyplot as plt<br>
from skimage import data,filters<br>

image = data.coins()<br>
#... or any other NumPy array!<br>
edges = filters.sobel(image)<br>
plt.imshow(image,cmap='gray')<br>
plt.show()<br>
plt.imshow(edges, cmap='gray')<br>
plt.show()<br><br>
![image](https://user-images.githubusercontent.com/97940332/186403212-34b37680-4232-4c7f-a4ae-0e0cc6a12397.png)
**********************************************************************************************************************************************************************

**#Mask an image**<br>

import numpy as np<br>
from skimage import data<br>
import matplotlib.pyplot as plt<br>
%matplotlib inline<br>

 
image = data.camera()<br>
type(image)<br>
plt.imshow(image)<br>
plt.show()<br>
plt.imshow(image,cmap='gray')<br>
plt.show()<br>

np.ndarray #Image is a numpy array<br>
mask = image < 87<br>
image[mask]=255<br>
plt.imshow(image, cmap='gray')<br>
plt.show()<br><br>
![image](https://user-images.githubusercontent.com/97940332/186403630-9a94274b-a850-456d-8303-6b9f1800cc21.png)
![image](https://user-images.githubusercontent.com/97940332/186403676-2bc0aeb0-a53d-4bfe-a4c3-27b98f5fc616.png)

******************************************************************************************************************************************************************
**#Blurring using  a gaussian Filter**<br>

from scipy import misc,ndimage<br>
 
<br>
face = misc.face()<br>
blurred_face = ndimage.gaussian_filter(face, sigma=3)<br>
very_blurred = ndimage.gaussian_filter(face, sigma=5)<br>


#Results<br>
plt.imshow(face)<br>
plt.show()<br>
plt.imshow(blurred_face)<br>
plt.show()<br>
plt.imshow(very_blurred)<br>
plt.show()<br><br>
![image](https://user-images.githubusercontent.com/97940332/186404101-b577eeec-6158-4658-ba85-a621c4339116.png)<br>
![image](https://user-images.githubusercontent.com/97940332/186404161-17cc497b-fc8b-4c35-b521-aeb7dbc311cf.png)<br>
********************************************************************************************************************************************************************
**Enhance an Image Using ImageFilter**<br>

from PIL import Image, ImageFilter<br>
#Read image<br>
im = Image.open( 'plant1.jpg' )<br>
#Display image<br>
im.show()<br>
#plt.imshow(im)<br>
#plt.show()<br>
 
from PIL import ImageEnhance<br>
enh = ImageEnhance.Contrast(im)<br>
enh.enhance(1.8).show("30% more contrast")<br><br>

![image](https://user-images.githubusercontent.com/97940332/186404575-57e288ac-33ce-4054-a8f2-2b49b4f3862e.png)
![image](https://user-images.githubusercontent.com/97940332/186404849-d8196c9d-55c4-43ab-879d-ee66824c5791.png)<br><br>

********************************************************************************************************************************************************************
**Edge Detection**<br>

import cv2<br>

 #Read the original image<br>
img = cv2.imread('flower1.jpg')<br>
#Display original image<br>
cv2.imshow('Original', img)<br>
cv2.waitKey(0)<br>
#Convert to graycsale<br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
#Blur the image for better edge detection<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)<br>
#Sobel Edge Detection<br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br>
#Display Sobel Edge Detection Images<br>
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>
#Canny Edge Detection<br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection<br>
#Display Canny Edge Detection Image<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br>
![image](https://user-images.githubusercontent.com/97940332/186405525-02c81be6-0a6d-45fc-b80d-edc9dc9b258d.png)
![image](https://user-images.githubusercontent.com/97940332/186405576-ec24cc00-4ffc-4a4e-babc-13333976c890.png)
![image](https://user-images.githubusercontent.com/97940332/186405621-0bb5a725-ec35-4a37-a68d-0f672eb5df64.png)
![image](https://user-images.githubusercontent.com/97940332/186405686-d35e41d8-246b-4723-99dc-0abadd8725c9.png)
![image](https://user-images.githubusercontent.com/97940332/186405755-2754d671-98ee-4fd9-97d6-7538cb5b65b8.png)

*****************************************************************************************************************************************************************
**25-8-2022**<br>
**1.Image Restoration**<br>

#a.Restore a damaged image<br>


import numpy as np<br>
import cv2<br>
import matplotlib.pyplot as plt<br>

#open the image<br>
img=cv2.imread('dimage_damaged.png')<br>
plt.imshow(img)<br>
plt.show()<br>

#load the mask<br>
mask=cv2.imread('dimage_mask.png',0)<br>
plt.imshow(mask)<br>
plt.show()<br>

#Inpaint<br>
dst=cv2.inpaint(img, mask,3,cv2.INPAINT_TELEA)<br>

#write the output<br>
cv2.imwrite('dimage_inpainted.png',dst)<br>
plt.imshow(dst)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97940332/187893770-14fd5a59-6c60-4757-9dd1-58dfdf53b3ee.png)
![image](https://user-images.githubusercontent.com/97940332/187893829-4bb27c4a-2547-49cc-9e10-97c17762c9ed.png)
****************************************************************************************************************************************************

#b.removing logo's<br>
<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
import pandas as pd<br><br>
plt.rcParams['figure.figsize']=(10,8)<br>


--------------------------------------------------------------------------------------------------------------------------
def show_image(image,title='Image',cmap_type='gray'):<br>
    plt.imshow(image,cmap=cmap_type)<br>
    plt.title(title)<br>
    plt.axis('off')<br>
    
--------------------------------------------------------------------------------------------------------------------------
def plot_comparison(img_original, img_filtered, img_title_filtered):<br>
    fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(10,8),sharex=True,sharey=True)<br>
    ax1.imshow(img_original,cmap=plt.cm.gray)<br>
    ax1.set_title('Original')<br>
    ax1.axis('off')<br>
    ax2.imshow(img_filtered,cmap=plt.cm.gray)<br>
    ax2.set_title(img_title_filtered)<br>
    ax2.axis('off')<br>
    

----------------------------------------------------------------------------------------------------------------------------------
    
from skimage.restoration import inpaint<br>
from skimage.transform import resize<br>
from skimage import color<br>

----------------------------------------------------------------------------------------------------------------------------------------
image_with_logo=plt.imread('imlogo.png')<br>

#initialize the mask<br>
mask=np.zeros(image_with_logo.shape[:-1])<br>

#set the pixels where the logo is to 1<br>
mask[210:272,360:425]=1<br>

#apply inpainting to remove the logo<br>
image_logo_removed=inpaint.inpaint_biharmonic(image_with_logo,<br>
                                              <br>mask,<br>
                                             multichannel=True)<br>
#show the original and logo removed images<br>
plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')<br>

![image](https://user-images.githubusercontent.com/97940332/187895262-71da9cfa-fc61-451a-90ad-ef818fba9d7c.png)
***********************************************************************************************************************************************************
(2) Noise:<br>
<br>
#a.Adding noise<br>

from skimage.util import random_noise<br>
<br>
fruit_image=plt.imread('fruitts.jpeg')<br>

#Add noise to the image<br>
noisy_image=random_noise(fruit_image)<br>

#Show the original and resulting image<br>
plot_comparison(fruit_image, noisy_image, 'Noisy image')<br>


![image](https://user-images.githubusercontent.com/97940332/187895549-c3c6b506-2272-4547-90a5-450bf0484349.png)
***********************************************************************************************************************************************************
#b.reducing noise<br>
#import matplotlib.pyplot as plt<br>
from skimage.restoration import denoise_tv_chambolle<br>

noisy_image=plt.imread('noisy.jpg')<br>

#Apply total variation filter denoising<br>
denoised_image=denoise_tv_chambolle(noisy_image,multichannel=True)<br>

#Show original and resulting images<br>
plot_comparison(noisy_image,denoised_image,'Denoised Image')<br>
![image](https://user-images.githubusercontent.com/97940332/187895851-c0d2dfc3-5c71-4186-ae57-ccb04ba35e47.png)
***************************************************************************************************************************************************************
#c.Redusing noise while preserving edges<br>
<br>
from skimage.restoration import denoise_bilateral<br>

landscape_image=plt.imread('noisy.jpg')<br>

#Apply bilateral filter denoising<br>
denoised_image= denoise_bilateral(landscape_image,multichannel=True)<br>

#Show original and resulting images<br>
plot_comparison(landscape_image, denoised_image,'Denoised Image')<br>

![image](https://user-images.githubusercontent.com/97940332/187896021-a4d736a5-d3de-4c1f-8107-dafb4e21e313.png)
******************************************************************************************************************************************************************
#3.Segmentation<br>
#a.Superpixel segmentation<br>

from skimage.segmentation import slic<br>
from skimage.color import label2rgb<br>
<br>
face_image=plt.imread('face.jpg')<br>

#Obtain the segmentation with 400 regions<br>
segments= slic(face_image, n_segments=400)<br>
<br>
#Put segments on top of original image to compare<br>
segmented_image=label2rgb(segments,face_image,kind='avg')<br>

#Show the segmented image<br>
plot_comparison(face_image,segmented_image,'Segmented image,400 superpixels')<br>

![image](https://user-images.githubusercontent.com/97940332/187896138-308f2a5e-0007-495e-90fa-5009d1843979.png)
*****************************************************************************************************************************************************************
#4.Contours:<br>
#a.Contouring shapes<br>

def show_image_contour(image,contours):<br>
    plt.figure()<br>
    for n,contour in enumerate(contours):<br>
        plt.plot(contour[:,1],contour[:,0],linewidth=3)<br>
    plt.imshow(image,interpolation='nearest', cmap='gray_r')<br>
    plt.title('Contours')<br>
    plt.axis('off')<br>
-----------------------------------------------------------------------------------------------------------------------------------------------------
from skimage import measure,data<br>

#Obtain the horse image<br>
horse_image=data.horse()<br>

#Find the contours with a constant level value 0.8<br>
contours=measure.find_contours(horse_image, level=0.8)<br>

#shows the image with contours found<br>
show_image_contour(horse_image,contours)<br>

![image](https://user-images.githubusercontent.com/97940332/187897604-6338d182-dbdf-4a23-ba63-a8b1fc9b5ca4.png)
****************************************************************************************************************************************************************
#b.Find contours of an image that is not binary<br>

from skimage.io import imread<br>
from skimage.filters import threshold_otsu<br>
<br>
image_dices=imread('diceimg.png')<br>

#Make the image grayscale<br>
image_dices=color.rgb2gray(image_dices)<br>

#Obtain the optimal thresh value<br>
thresh=threshold_otsu(image_dices)<br>

#Apply thresholding<br>
binary=image_dices>thresh<br>

#Find contours at a constant value of 0.8<br>
contours=measure.find_contours(binary,level=0.8)<br>

#Show the image<br>
show_image_contour(image_dices,contours)<br>



![image](https://user-images.githubusercontent.com/97940332/187897813-fcb49878-6541-41b0-a901-850d601a80cb.png)
*********************************************************************************************************************************************************************
#C.Count the dots in a dice's image<br>
<br>
#Create list with the shape of each contour<br>
shape_contours=[cnt.shape[0] for cnt in contours]<br>

#Set 50 as the maximum size of the dots shape<br>
max_dots_shape=50<br>

#Count dots in contours excluding bigger than dots size<br>
dots_contours=[cnt for cnt in contours if np.shape(cnt)[0]< max_dots_shape]<br>

#Shows all contours found<br>
show_image_contour(binary,contours)<br>

#Print the dice's number<br>
print('Dice's dots number: {}.'.format(len(dots_contours)))<br><br>


![image](https://user-images.githubusercontent.com/97940332/187897973-597fbf3b-9f0a-4453-8358-018feef44243.png)
*******************************************************************************************************************************************************************
#PILLOW FUNCTION<br>

from PIL import Image, ImageChops,ImageFilter<br>
from matplotlib import pyplot as plt<br>

#Create a PIL Image objects<br>
x=Image.open("x.png")<br>
o=Image.open("o.png")<br>

#Find out attributes of Image Objects<br>
print('size of the image: ',x.size,' colour mode:',x.mode)<br>
print('size of the image: ',o.size,' colour mode:',o.mode)<br>

#Plot 2 images one besides the other<br>
plt.subplot(121),plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122),plt.imshow(o)<br>
plt.axis('off')<br>

#multiply images<br>
merged=ImageChops.multiply(x,o)<br>

#adding 2 images<br>
add=ImageChops.add(x,o)<br>

#convert colour mode<br>
grayscale=merged.convert('L')<br>
grayscale<br>
<br>
![image](https://user-images.githubusercontent.com/97940332/187899238-610165b6-c317-4a37-aac0-a1de85c30d7e.png)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#More attributes<br>
image=merged<br>
<br>
print('image size: ' , image.size,<br>
     '\n colour mode: ', image.mode,<br>
     '\n image width: ' , image.width, '|also represented by: ' , image.size[0],<br>
     '\n image height: ' , image.height,'|also represented by: ', image.size[1],)<br>
![image](https://user-images.githubusercontent.com/97940332/187899461-79c6d7a8-fda2-4be8-b5f8-fb8958ad8776.png)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#mapping the pixels of the image so we can use them as coordinates<br>
pixel=grayscale.load()<br>

#a nested lop to parse through all the pixels in the image<br>
for row in range(grayscale.size[0]):<br>
    for column in range (grayscale.size[1]):<br>
        if pixel[row,column]!= (255):<br>
            pixel[row,column]=(0)<br>
            
grayscale<br>
<br>
![image](https://user-images.githubusercontent.com/97940332/187899852-394cc661-ce5c-4023-9382-edf5fcbd837b.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------
#1.invert image<br>
invert=ImageChops.invert(grayscale)<br>

#2.invert by subtraction<br>
bg=Image.new('L',(256,256), color=(255))<br>
subt=ImageChops.subtract(bg,grayscale)<br>

#3.rotate<br>
rotate=subt.rotate(45)<br>
rotate<br>

![image](https://user-images.githubusercontent.com/97940332/187900087-773e7e0b-6940-4c48-8f32-9a5c3d0f1075.png)
------------------------------------------------------------------------------------------------------------------------------------------------------------------
#gaussian blur<br>
blur=grayscale.filter(ImageFilter.GaussianBlur(radius=1))<br>

#edge detection<br>
edge=blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
![image](https://user-images.githubusercontent.com/97940332/187900216-ec57f168-722e-416b-95bc-b532dab53c7c.png)
------------------------------------------------------------------------------------------------------------------------------------------------------------------


#change edge colours<br>
edge=edge.convert('RGB')<br>
bg_red=Image.new('RGB',(256,256),color=(255,0,255))<br>

filled_edge=ImageChops.darker(bg_red,edge)<br>
filled_edge<br>

![image](https://user-images.githubusercontent.com/97940332/187900800-393f6963-234a-4c72-b5d8-7b1a547746dc.png)
*******************************************************************************************************************************************************************


















