'''灰度直方图'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/home/q/public security/photo/11.jpg'

src = cv2.imread(path)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

plt.hist(src.ravel(), 256)

cv2.imshow('gray', gray)
cv2.waitKey(1000)

plt.show()
cv2.destroyAllWindows()