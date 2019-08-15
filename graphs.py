#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:59:14 2019

@author: marcelo
"""
try:
    import pandas as pd
    import plotly
    import plotly.graph_objects as go
    import os
    import numpy as np

except Exception as e:
    print(e)
    raise Exception('Modulos não instalados')

def main():
    df_lbp = pd.read_csv('resultados/lbp_features_resultados.csv')
    df_hog = pd.read_csv('resultados/hog_features_resultados.csv')

    #nome com os parametros do SVM
    name_col_params = ['c', 'degree', 'gamma', 'kernel']

    #nome com as metricas utilizadas
    name_col_metrics = ['test_accuracy', 'test_f1_macro', 'test_precision_macro', 'test_recall_macro']
    
    #colunas desnecessarias
    name_delete_columns = ['Unnamed: 0', 'fit_time', 'score_time']

    #apangando colunas desnecessarias
    for name_column in name_delete_columns:
        df_lbp = df_lbp.drop(name_column, axis=1)
        df_hog = df_hog.drop(name_column, axis=1)

    print('Colunas LBP:', df_lbp.columns)
    print('Colunas HOG: ', df_hog.columns)

    #lista que contem os DataFrames
    df_features = [[df_lbp, 'lbp'], [df_hog, 'hog']]

    for df_feature, name_descriptor in df_features:
        
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

            df_feature[col_name] = list_values_column

        df_feature.to_csv('resultados/'+name_descriptor + '_mean_results.csv', index=False)

    for df_feature, name_descriptor in df_features:
        filter_linear = df_feature['kernel'] == 'linear'
        df_linear = df_feature.where(filter_linear)
        df_linear = df_linear.dropna(axis=1, how='all') 
        df_linear = df_linear.dropna()
        
        filter_poly2 = df_feature['kernel'] == 'poly'
        df_poly2 = df_feature.where(filter_poly2)
        filter_poly2 = df_feature['degree'] == 2
        df_poly2 = df_poly2.where(filter_poly2)
        df_poly2 = df_poly2.dropna()
        df_poly2 = df_poly2.drop('degree', axis=1)
        
        filter_poly3 = df_feature['kernel'] == 'poly'
        df_poly3 = df_feature.where(filter_poly3)
        filter_poly3 = df_feature['degree'] == 3
        df_poly3 = df_poly3.where(filter_poly3)
        df_poly3 = df_poly3.dropna()
        df_poly3 = df_poly3.drop('degree', axis=1)
        
        filter_sig = df_feature['kernel'] == 'sigmoid'
        df_sigmoid = df_feature.where(filter_sig)
        df_sigmoid = df_sigmoid.dropna(axis=1, how='all') 
        df_sigmoid = df_sigmoid.dropna()
        
        filter_rbf = df_feature['kernel'] == 'rbf'
        df_rbf = df_feature.where(filter_rbf)
        df_rbf = df_rbf.dropna(axis=1, how='all') 
        df_rbf = df_rbf.dropna()

        for name_metric in name_col_metrics:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter3d(
            x=df_linear['c'],
            y=[0,0,0,0],
            z=df_linear[name_metric],
            name='Linear', mode='markers'))

            fig.add_trace(go.Scatter3d(
            x=df_poly2['c'],
            y=df_poly2['gamma'],
            z=df_poly2[name_metric],
            name='Poly 2', mode='markers'))

            fig.add_trace(go.Scatter3d(
            x=df_poly3['c'],
            y=df_poly3['gamma'],
            z=df_poly3[name_metric],
            name='Poly 3', mode='markers'))

            fig.add_trace(go.Scatter3d(
            x=df_sigmoid['c'],
            y=df_sigmoid['gamma'],
            z=df_sigmoid[name_metric],
            name='Sigmoid', mode='markers'))

            fig.add_trace(go.Scatter3d(
            x=df_rbf['c'],
            y=df_rbf['gamma'],
            z=df_rbf[name_metric],
            name='RBF', mode='markers'))

            fig.update_layout(title=name_metric.replace('test_', ' ').replace('_', ' ').replace('macro', '').upper() + ' (x = C , y = gamma)')

            #fig.show()

            plotly.offline.plot(fig, filename = 'resultados/'+name_metric+'_'+name_descriptor+'.html', auto_open=False)

if __name__ == '__main__':

    main()
