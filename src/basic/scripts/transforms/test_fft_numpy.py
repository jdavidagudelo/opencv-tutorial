import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

path = '{0}/../../images/image1.jpeg'.format(os.path.dirname(os.path.abspath(__file__)))

img = cv2.imread(path)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
