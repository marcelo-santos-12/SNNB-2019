#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:15:14 2019

@author: marcelo
"""

try:
    import numpy as np
    import os
    import cv2
    import tqdm
    import pandas as pd
    import glob
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier
   
except Exception as e:
    print('Alguns módulos não foram instalados...')
    print(e)
    quit()


#The ANN classifier for LBP features
#consists of 300 neurons at the first and second layers and
#has a learning rate of 0.0005.
def classifier_mlp(x_train, y_train):
    '''
    Classificador SVM para realizar o treinamento.
    :x_train:
    :y_train:
    :Return:
    '''
    clf = MLPClassifier(hidden_layer_sizes=(300, 300,), \
        activation='relu', solver='adam', batch_size='auto', validation_fraction=0.2, \
        learning_rate_init=0.0005, verbose=True, early_stopping=True,)

    scores = cross_val_score(clf, x_train, y_train, cv=3)
    print('Finalizado...')
    return scores

#parâmetros do artigo: SVM: gamma=0.0000015, C = 2.5, kernel = 'RBF',
#validação: cross-validation 3-fold -- pegar a media da acuracia
def classifier_svm(x_train, y_train, C=10, gamma=0.0000015, kernel='rbf'):
    '''
    Classificador SVM para realizar o treinamento.
    :x_train:
    :y_train:
    :Return:
    '''
    print('Iniciando treinamento...')
    clf = SVC(C = C, gamma = gamma, kernel = kernel,)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print('Finalizado...')
    return scores

def get_data(path_base, split_test=0.2):
    '''
    Rotina para organizar e separar os dados que serão organizados para treinamento e teste.
    :path_base:
    :split_test:
    :Return:
    '''
    print('Organizando dados de treinamento...')
    CATEGORIES = os.listdir(path_base)
    x_train, y_train, x_test, y_test = [], [], [], []
    for category in CATEGORIES: 
        path = os.path.join(path_base, category)  
        class_num = CATEGORIES.index(category)  # get number of the classification
        cont_train = 0

        for csv in tqdm.tqdm(glob.iglob(path + '/*.csv')):

            try:
                lbp_array = pd.read_csv(csv)
                lbp_array = np.asarray(lbp_array).transpose()

                if cont_train < (1 - split_test) * len(os.listdir(path)):
                    x_train.append(lbp_array)
                    y_train.append(class_num)
                
                else:
                    x_train.append(lbp_array)
                    y_train.append(class_num)

            except Exception as e: 
                print(e)
                quit()

            cont_train += 1
    
    x_train = np.asarray(x_train)
    #x_test = np.asarray(x_test)
    x_train = x_train.reshape(-1, x_train.shape[2])

    y_train = np.asarray(y_train)

    #x_test = x_test.reshape(-1, x_test.shape[2], 1)
    return x_train, y_train
