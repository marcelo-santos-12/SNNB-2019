#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jul 25 20:15:17 2019

@author: marcelo
"""

try:
    import numpy as np
    import pandas as pd
    import os
    import glob
    import tqdm

except Exception as e:
    print('Modulos não instalados...')
    print(e)
    quit()

def create_training_data(path_class_hog, split_test):
    
    CATEGORIES = os.listdir(path_class_hog)
    
    x_train, y_train, x_test, y_test = [], [], [], []
    for category in CATEGORIES: 
        path = os.path.join(path_class_hog, category)  
        class_num = CATEGORIES.index(category)  # get number of the classification
        cont_train = 0
        for csv in tqdm.tqdm(glob.iglob(path + '/*.csv')):

            try:
                hog_array = pd.read_csv(csv)
                    
                if cont_train < (1 - split_test) * len(os.listdir(path)):
                    x_train.append(hog_array)
                    y_train.append(class_num)
                else:

                    x_test.append(hog_array)
                    y_test.append(class_num)
                    
            except Exception as e: 
                print(e)
                quit()

            cont_train += 1

    x_train = np.array(x_train).reshape(-1, x_train[0].shape[0], 1)
    x_test = np.array(x_test).reshape(-1, x_test[0].shape[0], 1)
    print(x_train.shape)
    print(x_test.shape)        
    return x_train, y_train, x_test, y_test

def model():
    pass

def main():
    path_class_hog = 'hog_orient_8_ppc_5_cpb_1'

    print('Iniciando script...')

    x_train, y_train, x_test, y_test = create_training_data(path_class_hog, split_test=0.2)

    classifier = svm.SVC(gamma='scale', decision_function_shape='ovo')

if __name__ == '__main__':

    main()
