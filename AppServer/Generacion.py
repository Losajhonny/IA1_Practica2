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

        self.mejor_usac = []
        self.mejor_marroquin = []
        self.mejor_landivar = []
        self.mejor_mariano = []

    def generar(self, porcentaje):
        for uni in self.universidades:
            self.actualizarModelos(uni, porcentaje)

        #seleccionar mejores modelos
        self.mejor_usac = sorted(self.modelos_usac, key= lambda item: item.test_accuracy, reverse=True)
        self.mejor_marroquin = sorted(self.modelos_marroquin, key= lambda item: item.test_accuracy, reverse=True)
        self.mejor_mariano = sorted(self.modelos_mariano, key= lambda item: item.test_accuracy, reverse=True)
        self.mejor_landivar = sorted(self.modelos_landivar, key= lambda item: item.test_accuracy, reverse=True)

    def graficarModelos(self, modelos):
        Plotter.show_Model(modelos)

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
        train, test, trai_set_x, trai_set_y, test_set_x, test_set_y, clases = File.getImagenes("AppServer/Imagenes/", uni, porcentaje)
        
        trai_set = Data(trai_set_x, trai_set_y, 255)
        test_set = Data(test_set_x, test_set_y, 255)

        modelo1 = Model(trai_set, test_set, reg=False, alpha=0.00001, lam=7, max_iterations=1500)
        modelo1.training(print_training=False)

        modelo2 = Model(trai_set, test_set, reg=False, alpha=0.25, lam=0.01, max_iterations=500)
        modelo2.training(print_training=False)

        modelo3 = Model(trai_set, test_set, reg=False, alpha=0.005, lam=0.005, max_iterations=450)
        modelo3.training(print_training=False)

        modelo4 = Model(trai_set, test_set, reg=False, alpha=0.0000001, lam=180, max_iterations=2000)
        modelo4.training(print_training=False)

        modelo5 = Model(trai_set, test_set, reg=False, alpha=0.1, lam=120, max_iterations=800)
        modelo5.training(print_training=False)
        
        return [modelo1, modelo2, modelo3, modelo4, modelo5]

    def predict(self, img, modelo):
        arr = np.array(img)
        x = []
        x.append(arr)
        x = np.array(x)
        nx = x.reshape(x.shape[0], -1).T

        y = np.array([1])
        ny = y.reshape((1, y.shape[0]))
        
        data = Data(nx, ny)

        predict = modelo.predict(data.x)
        #npredict = 100 - np.mean(np.abs(predict - data.y)) * 100

        #print(predict)
        #print(predict[0][0])
        
        if predict == 1:
            return True
        return False

#gen = Generacion()
#gen.generar(0.7)

#Plotter.show_Model(gen.modelos_usac, "Modelos Usac")
#Plotter.show_Model(gen.modelos_marroquin, "Modelos Marroquin")
#Plotter.show_Model(gen.modelos_mariano, "Modelos Mariano")
#Plotter.show_Model(gen.modelos_landivar, "Modelos Landivar")

"""
noAciertos1 = 0
        noAciertos12 = 0
        for i in train:
            x = np.array([i[0]])
            nx = x.reshape(x.shape[0], -1).T

            y = np.array([i[1]])
            ny = y.reshape((1, y.shape[0]))

            data = Data(nx, ny)
            predict = modelo1.predict(data.x)
            
            print('Original: ', x.shape)
            print('Con reshape: ', nx.shape)
            print('Original: ', y.shape)
            print('Con reshape: ', ny.shape)
            print("pr y", predict, data.y)
            print("predict - data.y", predict - data.y)
            print("np.abs(predict - data.y)", np.abs(predict - data.y))
            print("np.mean(np.abs(predict - data.y))", np.mean(np.abs(predict - data.y)))
            print("np.mean(np.abs(predict - data.y)) * 100", np.mean(np.abs(predict - data.y)) * 100)
            print("100 - np.mean(np.abs(predict - data.y)) * 100", 100 - np.mean(np.abs(predict - data.y)) * 100)

            npredict = 100 - np.mean(np.abs(predict - data.y)) * 100
            
            if npredict == 100:
                noAciertos1 += 1
            else:
                noAciertos12 += 1

        noAciertos2 = 0
        noAciertos22  = 0
        for i in test:
            x = np.array([i[0]])
            nx = x.reshape(x.shape[0], -1).T

            y = np.array([i[1]])
            ny = y.reshape((1, y.shape[0]))

            data = Data(nx, ny)
            predict = modelo1.predict(data.x)

            npredict = 100 - np.mean(np.abs(predict - data.y)) * 100

            if npredict == 100:
                noAciertos2 += 1
            else:
                noAciertos22 += 1

        print((noAciertos1 / len(train)) * 100)        
        print((noAciertos12 / len(train)) * 100)        
        print((noAciertos2 / len(test)) * 100)
        print((noAciertos22 / len(train)) * 100)        

        print(modelo1.train_accuracy)
        print(modelo1.test_accuracy)
"""


"""for i in train:
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
"""

"""
modelo1 = Model(trai_set, test_set, reg=False, alpha=0.00001, lam=7, max_iterations=900)
modelo1.training(print_training=False)

modelo2 = Model(trai_set, test_set, reg=False, alpha=0.08, lam=0.01, max_iterations=500)
modelo2.training(print_training=False)

modelo3 = Model(trai_set, test_set, reg=False, alpha=0.05, lam=4, max_iterations=450)
modelo3.training(print_training=False)

modelo4 = Model(trai_set, test_set, reg=False, alpha=0.0025, lam=180, max_iterations=1000)
modelo4.training(print_training=False)

modelo5 = Model(trai_set, test_set, reg=False, alpha=0.1, lam=120, max_iterations=800)
modelo5.training(print_training=False)

"""


#Esto lo estoy utilizando
"""
modelo1 = Model(trai_set, test_set, reg=False, alpha=0.00001, lam=7, max_iterations=900)
        modelo1.training(print_training=False)

        modelo2 = Model(trai_set, test_set, reg=False, alpha=0.25, lam=0.01, max_iterations=500)
        modelo2.training(print_training=False)

        modelo3 = Model(trai_set, test_set, reg=False, alpha=0.005, lam=0.005, max_iterations=450)
        modelo3.training(print_training=False)

        modelo4 = Model(trai_set, test_set, reg=False, alpha=0.0000001, lam=180, max_iterations=1000)
        modelo4.training(print_training=False)

        modelo5 = Model(trai_set, test_set, reg=False, alpha=0.1, lam=120, max_iterations=800)
        modelo5.training(print_training=False)
"""