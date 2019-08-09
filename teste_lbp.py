import os
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import cv2
import glob
from utils.get_descriptor_images import get_lbp

from mahotas.features.lbp import lbp


def main():

    name_img = 'img_sample/sample_lympho.tif'

    img = cv2.imread(name_img, 0)
    #hist = get_lbp(img)
    hist = lbp(img, radius=4, points=14, ignore_zeros=False)

    print('Tamanho do vetor: ', hist.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.hist(hist)
    plt.show()

    plt.show()
    
if __name__ == '__main__':

    main()
