# IPlab
http://localhost:8888/notebooks/vanitha%20IP%20lab/exercises.ipynb<br>
<br>
**1.#display gray scale image using read & write operations**<br>
import cv2<br>
img=cv2.imread('flower1.jpg',0)<br>
cv2.imshow('flower1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br>

![image](https://user-images.githubusercontent.com/97940332/173813972-9a441bd0-13f9-4d8f-8c9a-c8508d154410.png)

**2.
#display the image using matplotlib**<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('butterfly1.jpg')<br>
plt.imshow(img)<br>

![image](https://user-images.githubusercontent.com/97940332/173815217-a4c1675a-d490-480a-839d-f93ffde1142d.png)

**3.#linear transformation rotation**<br>
import cv2<br>
from PIL import Image<br>
img=Image.open("butterfly1.jpg")<br>
img=img.rotate(90)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br><br><br>

![image](https://user-images.githubusercontent.com/97940332/173816085-bad9dd80-0037-4980-a675-103b066a666b.png)

**4.#convert color string to RGB color values**<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
img3=ImageColor.getrgb("pink")<br>
print(img3)<br><br><br>

![image](https://user-images.githubusercontent.com/97940332/173816399-e79671f8-293a-477a-a98f-5b5d5d1af535.png)

**5.#create image using colors**<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,0,255))<br>
img.show()<br><br>

![image](https://user-images.githubusercontent.com/97940332/173817317-bb78bbba-3ab2-4682-bd1a-d05e6a16ecbd.png)

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
![image](https://user-images.githubusercontent.com/97940332/173817844-72cdb6b9-5082-4ae8-b200-bdc52bbd8ad3.png)

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

![image](https://user-images.githubusercontent.com/97940332/173818118-79f81917-dd36-4f75-a8be-662ad6721023.png)


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













