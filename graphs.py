#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:59:14 2019

@author: marcelo
"""

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

except Exception as e:
    print(e)
    raise Exception('Modulos não instalados')

def main():
    df_lbp = pd.read_csv('lbp_features_resultados.csv')
    df_hog = pd.read_csv('hog_features_resultados.csv')

    #nome com os parametros do SVM
    name_col_params = ['c', 'degree', 'gamma', 'kernel']
    
    #colunas desnecessarias
    name_delete_columns = ['Unnamed: 0', 'fit_time', 'score_time']
    #apangando colunas desnecessarias
    for name_column in name_delete_columns:
        df_lbp = df_lbp.drop(name_column, axis=1)
        df_hog = df_hog.drop(name_column, axis=1)

    print('Colunas LBP:', df_lbp.columns)
    print('Colunas HOG: ', df_hog.columns)

    #lista que contem os DataFrames
    df_features = [df_lbp, df_hog]


    for df_feature in df_features:
        df_aux = pd.DataFrame()
        
        for col_name in df_feature.columns:
            
            list_values_column = []
            if col_name in name_col_params:
                continue
        
            for i in range(df_feature.count()[0]):
                
                list_ = df_feature[col_name][i].replace('[', '').replace(']','').split(' ')
                
                while True:
                    try:
                        list_.remove('')
                    except:
                        break
                
                if len(list_) != 5:
                    raise Exception('Lista possui tamanho diferente de 5...')

                float_array = map(float, list_)
                float_array = list(float_array)
                float_array = np.array(float_array)
                
                float_mean_array = float_array.mean().round(3)

                list_values_column.append(float_mean_array)
            
            df_aux['mean_' + col_name] = list_values_column

        print(df_aux)
        

    #PLOTAR C EM FUNCAO DE GAMMA, COM OS VALORES DAS METRICAS

if __name__ == '__main__':

    main()
