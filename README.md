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
cv2.destroyAllWindows()<br>

