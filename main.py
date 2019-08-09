#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:07:14 2019

@author: marcelo
"""

try:
    from utils.get_descriptor_images import *
    from utils.get_classifier import *

except Exception as e:
    print('Alguns módulos não foram instalados...')
    print(e)
    quit()

def main():
    #link onde as imagens podem ser baixadas
    url = 'https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1'

    #diretorio onde elas serão guardadas após baixadas e extraida do zip
    path_base = 'tissue_class'

    #rotina para verificar se a pasta com as imagens ja foram baixadas e organizadas
    DATADIR = get_path_images(url, path_base)
    OUTDIR = "lbp_features"

    time_initial = time()

    #rotina que calcula os descritores das imagens
    create_data_lbp(DATADIR, OUTDIR, points=16, radius=4, VERBOSE=True)

    #verifica se o diretorio do descritores esta faltoso, so inicia treinamento com descritores calculados
    time_create_init = time()
    while True:
        soma = 0
        if os.path.exists(OUTDIR):
            for class_ in os.listdir(OUTDIR):
                soma += len(os.listdir(OUTDIR + '/' + class_))

            if soma >= 7*625:
                break
    
    time_create_final = time()

    print('Tempo de criação do dataset: {}s'.format(time_create_final-time_create_init))

    x_train, y_train = get_data(OUTDIR, split_test=0)
    scores = classifier_svm(x_train, y_train)

    print('Acuracia: ', scores)
    print('Media: ', scores.mean())
    print('Desvio padrão: ', scores.std())

    time_final = time()

    print('Tempo de execução: {}s'.format(time_final-time_initial))

    print('\a')
    print('\a')
    print('\a')
if __name__ == '__main__':

    main()
