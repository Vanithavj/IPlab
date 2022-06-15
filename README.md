# IPlab
http://localhost:8888/notebooks/vanitha%20IP%20lab/exercises.ipynb
1.#display gray scale image using read & write operations
import cv2
img=cv2.imread('flower1.jpg',0)
cv2.imshow('flower1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
