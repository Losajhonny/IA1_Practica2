#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_picture(pixels):
    plt.imshow(pixels)
    plt.show()

def obtenerImagenes(path, base):
    carpetas = os.listdir(path)
    datos = []
    for carpeta in carpetas:
        dirpath = path + carpeta + "/"
        filenames = os.listdir(dirpath)
        
        for filename in filenames:
            x = mpimg.imread(dirpath + filename)
            y = 0

            if base == carpeta:
                y = 1
                #print("base:", base, "carpeta:", carpeta, "y:", y)

            datos.append([x, y])
    return datos

def randomImagenes(datos, porcentaje):
    np.random.shuffle(datos)
    num = int(len(datos) * porcentaje)
    train = datos[0: num]
    test = datos[num:]
    return train, test

def xyTrainTest(entre, prueb):
    trai_x = []
    trai_y = []

    test_x = []
    test_y = []

    for tmp in entre:
        trai_x.append(tmp[0])
        trai_y.append(tmp[1])

    for tmp in prueb:
        test_x.append(tmp[0])
        test_y.append(tmp[1])

    trai_x = np.array(trai_x)
    trai_y = np.array(trai_y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    #Les aplica reshape, convierte al arreglo en un arreglo de areglos
    trai_y = trai_y.reshape((1, trai_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))
    
    return trai_x, trai_y, test_x, test_y

def getImagenes(path, base, porcentaje):
    datos = obtenerImagenes(path, base)
    
    train, test = randomImagenes(datos, porcentaje)
    trai_x, trai_set_y, test_x, test_set_y = xyTrainTest(train, test)

    # Convertir imagenes a un solo arreglo
    trai_set_x = trai_x.reshape(trai_x.shape[0], -1).T
    test_set_x = test_x.reshape(test_x.shape[0], -1).T

    #print('Original: ', trai_x.shape)
    #print('Con reshape: ', trai_set_x.shape)
    #print("Y: ", trai_set_y.shape)

    return train, test, trai_set_x, trai_set_y, test_set_x, test_set_y, ["No " + base, base]


#getImagenes("Imagenes/", "USAC", 0.7)

def read_file(path):
    data = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    #print( data)
    #print(len(data))
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

    #print(train_set)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['Perdera', 'Ganara']

def load_dataset():
    train_dataset = h5py.File('C:/Mis Documentos/Github/IA1_Practica2/AppServer/Datasets/train_catvnoncat.h5', "r")

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

"""
path = "Imagenes/"
base = "USAC"

carpetas = os.listdir(path)
for carpeta in carpetas:
    dirpath = path + carpeta + "/"
    filenames = os.listdir(dirpath)
    
    for filename in filenames:
        x = mpimg.imread(dirpath + filename)
        y = 0

        print(x)

        if base == carpeta:
            y = 1
        break
    break
"""

"""
def getImagenes(path, base, porcentaje):
    #path = "AppServer/Imagenes/"
    #base = "USAC"
    #porcentaje = 0.7

    carpetas = os.listdir(path)
    datos = []
    for carpeta in carpetas:
        dirpath = path + carpeta + "/"
        filenames = os.listdir(dirpath)
        
        for filename in filenames:
            x = mpimg.imread(dirpath + filename)
            y = 0

            if base == carpeta:
                y = 1

            x = x.reshape(1, -1)
            print(x)
            print(x.shape)
            #x = x.reshape(x.shape[0], -1).T

            datos.append([x, y])

    np.random.shuffle(datos)
    num = int(len(datos) * porcentaje)
    train = datos[0: num]
    test = datos[num:]

    # obteniendo datos train
    train_orig_x = []
    train_orig_y = []

    for t in train:
        train_orig_x.append(t[0])
        train_orig_y.append(t[1])
    
    train_orig_x = np.array(train_orig_x)
    train_orig_y = np.array(train_orig_y)
    
    train_x = train_orig_x.reshape(train_orig_x.shape[0], -1).T
    train_y = train_orig_y.reshape(train_orig_y.shape[0], -1).T

    # obteniendo datos test
    test_orig_x = []
    test_orig_y = []

    for t in test:
        test_orig_x.append(t[0])
        test_orig_y.append(t[1])
    
    test_orig_x = np.array(test_orig_x)
    test_orig_y = np.array(test_orig_y)
    
    test_x = test_orig_x.reshape(test_orig_x.shape[0], -1).T
    test_y = test_orig_y.reshape(test_orig_y.shape[0], -1).T

    return train, test, train_x, train_y, test_x, test_y, [base, "No " + base]
"""