#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

    return train_x, train_y, test_x, test_y, [base, "No " + base]
