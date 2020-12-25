from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np

class Generacion:
    def __init__(self):
        self.universidades = ["USAC", "Marroquin", "Landivar", "Mariano"]
        self.modelos_usac = []
        self.modelos_marroquin = []
        self.modelos_landivar = []
        self.modelos_mariano = []

    def generar(self, porcentaje):
        for uni in self.universidades:
            self.actualizarModelos(uni, porcentaje)
            break

        #seleccionar mejores modelos

    def actualizarModelos(self, uni, porcentaje):
        res = self.generarModelos(uni, porcentaje)
        if uni == "USAC":
            self.modelos_usac = res
        elif uni == "Marroquin":
            self.modelos_marroquin = res
        elif uni == "Landivar":
            self.modelos_landivar = res
        else:
            self.modelos_mariano = res

    def generarModelos(self, uni, porcentaje):
        train, test, train_x, train_y, test_x, test_y, clases = File.getImagenes("AppServer/Imagenes/", uni, porcentaje)
        train_set = Data(train_x, train_y, 255)
        test_set = Data(test_x, test_y, 255)

        #print(train_x.shape)
        #print(train_set.y)

        modelo1 = Model(train_set, test_set, reg=False, alpha=0.00001, lam=7, max_iterations=25)#, max_iterations=10000, min_value=0.5)
        modelo1.training(print_training=False)

        """
        modelo1 = Model(train_set, test_set, reg=False, alpha=0.00001, lam=7, max_iterations=900)#, max_iterations=10000, min_value=0.5)
        modelo1.training(print_training=False)

        modelo2 = Model(train_set, test_set, reg=False, alpha=0.08, lam=0.01, max_iterations=500)#, max_iterations=10000, min_value=0.5)
        modelo2.training(print_training=False)

        modelo3 = Model(train_set, test_set, reg=False, alpha=0.05, lam=4, max_iterations=450)#, max_iterations=10000, min_value=0.5)
        modelo3.training(print_training=False)

        modelo4 = Model(train_set, test_set, reg=False, alpha=0.0025, lam=180, max_iterations=1000)#, max_iterations=10000, min_value=0.5)
        modelo4.training(print_training=False)

        modelo5 = Model(train_set, test_set, reg=False, alpha=0.1, lam=120, max_iterations=800)#, max_iterations=10000, min_value=0.5)
        modelo5.training(print_training=False)

        """

        
        for i in train:
            x = np.array(i[0])
            y = [1]

            a = x.reshape(-1, 1)
            b = Data(x.reshape(-1, 1), [[y]], 255)

            print(a.shape)
            print(b.x)

            p = modelo1.predict(b.x)
            print(p)
            print(p, 100 - np.mean(np.abs(p - b.y)) * 100)

            break

        return [modelo1]

""", modelo2, modelo3, modelo4, modelo5"""
aa = Generacion()
aa.generar(0.7)