import cv2
import numpy as np

a = cv2.imread('/home/oh/Downloads/article-cat-vet-visit-guide.jpg',0)


b = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
cv2.imshow('asdf',b)
cv2.waitKey(0)
