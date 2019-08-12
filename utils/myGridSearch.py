#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:38:14 2019

@author: marcelo
"""
try:
    from numpy import array, transpose
    import os
    import cv2
    import pandas as pd
    from multiprocessing import Process
    from sklearn.model_selection import cross_val_score, cross_validate
    from time import time
    from sklearn.svm import SVC
except Exception as e:
    print(e)
    raise Exception('Alguns módulos não foram instalados...')

class MyGridSearch:

    def __init__(self, model, grid_params, cv, metrics):
        self.model = model
        self.grid_params = grid_params
        self.cv = cv
        self.metrics = metrics

    def fit(self, x_train, y_train):
        results = []
        params = []
        count = 0
        for kernel in self.grid_params['kernel']:
            for c in self.grid_params['C']:
                
                if kernel == 'linear':
                    self.model.set_params(C=c, kernel=kernel)
                    scores = cross_validate(self.model, x_train, y_train, cv=self.cv, \
                        scoring=self.metrics, n_jobs=-1, verbose=1, return_train_score=False)
                    param = {
                        'kernel':kernel,
                        'c': c
                    }
                    results.append(scores)
                    params.append(param)
                    count += 1
                    
                elif kernel == 'poly':
                    
                    for gamma in self.grid_params['gamma']:
                        
                        for degree in self.grid_params['degree']:
                            self.model.set_params(C=c, kernel=kernel, gamma=gamma, \
                                degree=degree)
                            scores = cross_validate(self.model, x_train, y_train, cv=self.cv, \
                                scoring=self.metrics, n_jobs=-1, verbose=1, return_train_score=False)
                            param = {
                                'kernel':kernel,
                                'c': c,
                                'gamma':gamma,
                                'degree':degree
                                }
                            results.append(scores)
                            params.append(param)
                            count += 1

                elif kernel == 'rbf':
                    for gamma in self.grid_params['gamma']:
                        self.model.set_params(C=c, kernel=kernel, gamma=gamma)
                        scores = cross_validate(self.model, x_train, y_train, cv=self.cv, \
                                    scoring=self.metrics, n_jobs=-1,  verbose=1, return_train_score=False)
                        param={
                            'kernel':kernel,
                            'c':c,
                            'gamma':gamma
                        }
                        results.append(scores)
                        params.append(param)
                        count += 1

                else: #'sigmoid'
                    for gamma in self.grid_params['gamma']:
                        
                        self.model.set_params(C=c, kernel=kernel, gamma=gamma)
                        scores = cross_validate(self.model, x_train, y_train, cv=self.cv, \
                                scoring=self.metrics, n_jobs=-1, verbose=1, return_train_score=False)
                        param={
                            'kernel':kernel,
                            'c':c,
                            'gamma':gamma
                        }
                        results.append(scores)
                        params.append(param)
                        count += 1
        
        print('Quantidade de modelos treinados: {}'.format(count))

        return results, params

    def __str__(self,):
        return str(self.grid_params)

def main():

    svm = SVC()

    metrics = ['f1_macro', 'accuracy', 'precision_macro','recall_macro']

    grid_params = {
        'C': [1, 10, 50, 100],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': [0.0001, 0.00001, 0.000001],
        'degree': [2, 3]
    }

    gs = MyGridSearch(model=svm, grid_params=grid_params, cv=2, metrics=metrics)

    x = array([[2,1,2,2], [3,2,4,3], [5,4,4,3], [3,4,6,6], [6,5,5,4], [3,2,8,7], [5,3,4,5], [6,2,1,7], [2,1,2,1], [8,4,4,3], [6,4,3,2], [4,3,4,3]])
    y = array([1,1,1,1,2,2,2,2,3,3,3,3])

    results, params = gs.fit(x, y)

    df_results = pd.DataFrame(results)
    df_params = pd.DataFrame(params)
    df_conc = pd.concat([df_results, df_params], axis=1)
    df_conc.to_csv('teste.csv')

    print(results)

if __name__ == '__main__':
    main()