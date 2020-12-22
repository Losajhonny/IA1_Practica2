from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np

# obtener datos
train_x, train_y, test_x, test_y, clases = File.getImagenes("AppServer/Imagenes/", "Marroquin", 0.7)

train_set = Data(train_x, train_y, 255)
test_set = Data(test_x, test_y, 255)

#modelo1 = Model(train_set, test_set, reg=True, alpha=0.000001, lam=300, max_iterations=10000, min_value=0.5)
#modelo1.training(print_training=False)

modelo2 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=150, max_iterations=10000, min_value=0.5)
modelo2.training(print_training=False)

#Plotter.show_Model([modelo1, modelo2])
Plotter.show_Model([modelo2])