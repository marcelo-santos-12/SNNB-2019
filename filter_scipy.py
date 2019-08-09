import numpy as np
from scipy.ndimage import generic_filter
import random
import cv2
import matplotlib.pyplot as plt
from skimage import feature

random.seed(5)
img_size = 150

a = []
b = []

for i in range(img_size):
    a.append(random.sample(list(np.arange(256)), img_size))
    b.append(random.sample(list(np.arange(256)), img_size))

a = np.asarray(a)
b = np.asarray(b) 

def to_binary(value_decimal):
    if value_decimal < 0:
        return 0
    else:
        return 1

def element(pixel):
    diference = pixel - pixel[4]
    binary = list(map(to_binary, diference))
    array = binary[0:3] #concatenando numeros binarios no sentindo horario, comecando da primeira posicao
    array.append(binary[5])
    array.extend(binary[6:10][::-1])
    array.append(binary[3])
    array = str(array).strip('[]').replace(', ' , '')
    return int(array, 2) #converte de binario para decimal

radius = 8
point = 1
lbp = feature.local_binary_pattern(a, point, radius, method='default')

c = generic_filter(a, element, size=(3,3))
print(a)
print(c)
print(lbp)

plt.imshow(lbp)
plt.show()