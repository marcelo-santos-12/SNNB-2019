#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:07:14 2019

@author: marcelo
"""
__version__ = '1.0'
__author__ = 'Marcelo dos Santos (mar.mhps@gmail.com)'

try:
    from utils.get_descriptor_images import *
    from utils.get_classifiers import *
    from utils.get_data import *

except Exception as e:
    print('Alguns módulos não foram instalados...')
    raise Exception(e)

def main():
    #------------------------------------------------------------------------------------------
    #BUSCA DA PASTA DE IMAGENS
    #link onde as imagens podem ser baixadas
    url = 'https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1'
    
    #diretorio onde elas serão guardadas após baixadas e extraida do zip
    path_base = 'tissue_class'
    
    #Rotina para busca da pasta com as imagens histológicas
    DATADIR = get_path_images(url, path_base)
    #------------------------------------------------------------------------------------------
    #COMPUTANDO DESCRITORES
    obj_lbp = LBP(radius=4, points=14)
    obj_hog = HOG(ppc=(20, 20) , cpb=(2, 2), orient=9)
    
    features = [obj_lbp, obj_hog] #lista com todos os recursos que serão computados
    
    OUTDIR_FEATURES = []
    for obj_feature in features:

        OUTDIR = str(obj_feature) + "_features"
        OUTDIR_FEATURES.append(OUTDIR)

        #computa os descritores
        threads = create_features(DATADIR, OUTDIR, obj_feature, verbose=True,)

        '''while len(threads) > 0:
            for [t, time_init, class_] in threads:
                if not t.is_alive():
                    print('Tempo de criação do dataset {} para class{}: {}s'.format(str(obj_feature), \
                                                            class_, time() - time_init))
                    threads.remove([t, time, class_])
        '''         
    #------------------------------------------------------------------------------------------
    #TREINAMENTO
    
    train_time = time()

    for out in OUTDIR_FEATURES: #para cada uma das pastas com os recursos computados
        print('Organizando dados de treinamento de \'/{}\'...'.format(out))
        path_csv = out
        
        x_train, y_train = create_training_data(out, path_csv)
        
        print('Treinando SVM com {}...'.format(out))
        init_train_time = time()
        
        fit_grid_search(x_train, y_train, out)

        print('Tempo treinamento com \'/{}\': {}s'.format(out, time() - init_train_time))
    
    print('Tempo gasto: ', time()-train_time)
    print('Fim dos Treinamentos...')

if __name__ == '__main__':

    main()
