#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:20:14 2019

@author: marcelo
"""

try:
    from numpy import array, histogram, arange
    import os
    import cv2
    import pandas as pd
    from wget import download
    from zipfile import ZipFile
    from threading import Thread
    from multiprocessing import Process
    from skimage.feature import local_binary_pattern
    from time import time
    from mahotas.features.lbp import lbp

except Exception as e:
    print('Alguns módulos não foram instalados...')
    print(e)
    quit()

def get_lbp(img, points, radius):
    '''Parametros baseados no artigo.
    :img: 
    :Return:
    '''
    hist = lbp(img, radius=radius, points=points, ignore_zeros=False)

    return hist

def get_path_images(url, path_base):
    '''
    Rotina para busca da pasta com as imagens histológicas.
    Faz o download, extrai o zip e renomeia a pasta extraida para path_base, se a path_base não existir. 
    :url: link que aponta para o endereço do servidor que armazena as imagens.
    :path_base: pasta que contém as imagens.
    :Return: Nome da pasta onde as imagens estão guardadas.
    '''

    print('Iniciando busca da pasta de imagens...')
    filename = path_base
    if not os.path.exists(filename):
        print('Pasta ' + filename + ' não existe. Iniciando download...')
        try:
            filename_zip_images = download(url)
            print('Download concluido...')
        except Exception as e:
            print('Erro no download...')
            print('Verifique sua conexão com a internet ou o se o link ainda está ativo...')
            print(e)
            quit()

        print('EXTRAINDO ZIP...')
        try:
            fantasy_zip = ZipFile(filename_zip_images)
            fantasy_zip.extractall('.')
            fantasy_zip.close()
            print('CONCLUINDO...')
        except:
            print('Erro ao extrair zip...')
            quit()

        try:
            #renomeando pasta extraida
            os.system('mv ' + fantasy_zip.filename[:-4] + ' ' + filename)

        except:
            print('Erro ao renomear pasta...')
            quit()
    else:
        print('Pasta encontrada...')

    return filename

def get_images(path, list_name_imgs):
    '''
    Cria um gerador contendo as imagens da pasta atual.
    :list_name_imgs: lista que contém os nomes das imagens a serem inseridas no gerador.
    :Return: Gerador das imagens.
    '''
    return ([cv2.imread(os.path.join(path, name_img), cv2.IMREAD_GRAYSCALE), name_img] for name_img in list_name_imgs)

def save_file(path_lbp, lbp_feature):
    '''
    Salve o descritor lbp em um arquivos csv.
    :path_lbp: Local onde o arquivos csv será armazenado.
    :lbp_feature: vetor de características lbp.
    :Return: Sem retorno.
    '''
    df_lbp = pd.DataFrame(array(lbp_feature))    
    df_lbp.to_csv(path_lbp, index=False)

def thread_lbp(DATADIR, OUTDIR, class_tissue, eq_adap_obj, points, radius, VERBOSE=False):
    '''
    Computa os descritores de cada classe de tecidos.
    :DATADIR: pasta que contem as imagens.
    :class_tissue: classe atual para a qual serão computados os descritores.
    :eq_adap_obj: Objeto da classe de equalização de histograma adaptativa --> CLAHE.
    :Return: Sem retorno.
    '''

    if VERBOSE:
        print('Iniciando execução na Categoria: ', class_tissue)

    path_class_imgs = os.path.join(DATADIR, class_tissue)

    list_name_imgs = os.listdir(path_class_imgs)

    imgs_generator = get_images(path_class_imgs, list_name_imgs) #gerador com todas as imagens da classe atual    

    for [img, name_img] in imgs_generator:
        
        path_lbp = OUTDIR + '/' + class_tissue + '/'

        if not os.path.exists(path_lbp):
            os.makedirs(path_lbp)

        path_lbp += name_img[:-4] + '.csv'

        #APLICANDO EQUALIZACAO DE HISTOGRAMA ADAPTATIVA NA IMAGEM ATUAL
        img_equ_adap = eq_adap_obj.apply(img)

        #CALCULANDO lbp COM OS PARAMETROS ATUAIS
        lbp_feature = get_lbp(img_equ_adap, points, radius)

        #SALVAR lbp's
        save_file(path_lbp, lbp_feature)

    if VERBOSE:
        print('Classe {} finalizada...'.format(class_tissue))

def create_data_lbp(DATADIR, OUTDIR, VERBOSE=False, points=14, radius=4):
    '''
    Computa os descritores LBP's das imagens .tif.
    :DATADIR: Path que contem o diretorio das imagens.
    :OUTDIR: Path onde serao armazenados os descritores LBP's.
    :Return: Sem retorno.
    '''
    if VERBOSE:
        print('Computando recursos...')      

    if os.path.exists(OUTDIR):
        print('Descritores LBP já disponíveis...')
        return

    classes_tissues = os.listdir(DATADIR)
    #classes_tissues.pop(classes_tissues.index('08_EMPTY')) #REMOVER CLASSE VAZIO

    eq_adap_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))

    process = []
    for class_tissue in classes_tissues:
        t = Process(target=thread_lbp, args=(DATADIR, OUTDIR, class_tissue, eq_adap_obj, points, radius, VERBOSE,))
        t.start()
        inicio = time() 
        process.append([t, inicio, class_tissue])

    '''time_class = []
    control = True
    while control:
        for t, inicio, class_tissue in process:
            if not t.isAlive():
                fim = time()
                tempo = fim - inicio
                time_class.append([class_tissue, tempo])
                control = False

    print('Todos os descritores foram calculados...')'''
    