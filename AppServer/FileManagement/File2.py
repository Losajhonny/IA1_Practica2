#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
#import h5py
from PIL import Image

import os
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import Dato

source = None

#alpha --> tasa de entrenamiento
"""
def getImagen(path, base, carpeta):
    # generar dato
    dato = Dato.Dato()
    # obtener imagen
    dato.x = mpimg.imread(path)

    #print(np.array(dato.x).shape)
    #print((dato.x.reshape(dato.x.shape[0], -1).T).shape)

    # definir Y en base al nombre de modelo
    #if base == carpeta:
    #    dato.y = 1
    return dato

def getImagenes(path, base):
    dato = getImagen(path, base, 'carpeta')
    print(dato.x)

    # obtener carpetas
    #carpetas = os.listdir(path)
    # obtener datos
    #datos = []
    # recorrer carpeta
    #for carpeta in carpetas:
        # definir direccion de carpeta
        #dirpath = path + carpeta + "/"
        # obtener nombre de coleccion de imagenes
        #filenames = os.listdir(dirpath)
        # recorrer coleccion
        #for filename in filenames:
            # obtener imagen
            #dato = getImagen(dirpath + filename, base, carpeta)
            
            # agregar dato
            #datos.append(dato)
    
    # revolver datos
    #np.random.shuffle(datos)

    # obtener datos de entrenamiento y prueba
    #num = int(len(datos) * 0.7)
    #train = datos[0: num]
    #test = datos[num:]
        


    

#getImagenes("AppServer/Imagenes/USAC/1.jpg", "USAC")

def getImages(path, carpeta):
    data = []

    # obtener todos los archivos
    filenames = os.listdir(path)

    # guardar imagenes en data
    for filename in  filenames:
        imagen = mpimg.imread(path + filename)
        data.append(imagen)

    # obtener datos de entrenamiento y prueba
    arr = np.array(data)
    num = int(len(arr) * 0.7)
    train_orig = arr[0: num]
    test_orig = arr[num:]

    # obtener variables dependientes e independients
    num = int(len(train_orig) * 0.8)
    train_orig_x = train_orig[0: num]
    train_orig_y = train_orig[num:]

    num = int(len(test_orig) * 0.8)
    test_orig_x = test_orig[0: num]
    test_orig_y = test_orig[num:]

    # transformar imagenes
    train_x = train_orig_x.reshape(train_orig_x.shape[0], -1).T
    #train_y = train_orig_y.reshape(train_orig_y.shape[0], -1).T
    #test_x = test_orig_x.reshape(test_orig_x.shape[0], -1).T
    #test_y = test_orig_y.reshape(test_orig_y.shape[0], -1).T

    #print(train_x)
    
    print(train_orig_x.shape)
    print(train_x.shape)
    #print(train_orig_y.shape)
    #print(train_x.shape)
    #print(test_orig_x.shape)
    #print(test_x.shape)
    #print(test_orig_y.shape)
    #print(test_x.shape)

    #plt.imshow(train_orig_x[0])
    #plt.show()
    #return train_x, train_y, test_x, test_y, [carpeta]

getImages("AppServer/Imagenes/Marroquin/", "Marroquin")
"""
def read_file(path):
    data = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    #print( data)
    print(len(data))
    result = np.array(data)
    
    #print("1")
    #print(result)
    
    np.random.shuffle(result)
    
    #print("2")
    #print(result)
    
    result = result.astype(float).T

    #print(len(result))
    #print(result.shape)
    #print(result.shape[1])

    # Se separa el conjunto de pruebas del de entrenamiento
    slice_point = int(result.shape[1] * 0.7)
    train_set = result[:, 0: slice_point]
    test_set = result[:, slice_point:]

    # Se separan las entradas de las salidas
    train_set_x_orig = train_set[0: 3, :]
    train_set_y_orig = np.array([train_set[3, :]])

    test_set_x_orig = test_set[0: 3, :]
    test_set_y_orig = np.array([test_set[3, :]])

    print(train_set)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['Perdera', 'Ganara']

#read_file("AppServer/Datasets/MC2A.csv")

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")

    #train_set_x_orig = arreglo de imágenes
    #train_set_y_orig = arreglo de imágenes

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # entradas de entrenamiento
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # salidas de entrenamiento

    #print('************** train_set_x_orig **************')
    #print(train_set_x_orig)
    #print(type(train_set_x_orig))
    #print('************** train_set_y_orig **************')
    #print(train_set_y_orig)
    #print(len(train_set_y_orig))



    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # entradas de prueba
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # salidas de prueba

    #print('************** test_set_x_orig **************')
    #print(test_set_x_orig)
    #print(len(test_set_x_orig))
    #print('************** test_set_y_orig **************')
    #print(test_set_y_orig) #Arreglo con las respuestas correctas, donde 0 = NO es un gato, 1 = SÍ es un gato
    #print(len(test_set_y_orig))



    #Les aplica reshape, convierte al arreglo en un arreglo de areglos
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    #print('************** train_set_y_orig con reshape**************')
    #print(train_set_y_orig)
    #print(len(train_set_y_orig))
    #print('************** test_set_y_orig con reshape**************')
    #print(test_set_y_orig)
    #print(len(test_set_y_orig))

    #print(type(train_set_x_orig))
    #print(type(train_set_y_orig))
    #print(type(test_set_x_orig))
    #print(type(test_set_y_orig))

    #print(len(train_set_x_orig))
    #print(train_set_x_orig.shape)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No Gato', 'Gato']



