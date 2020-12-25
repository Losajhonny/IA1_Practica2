from flask import Flask, request, jsonify
from flask_cors import CORS
from Singleton import *
from Generacion import *
from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return 'Bienvenido'

@app.route('/generarModelo', methods=['GET'])
def generarModelo():
    print("generando...")
    gen = Generacion()
    gen.generar(0.70)

    """
    universidades = ["USAC", "Marroquin", "Landivar", "Mariano"]
    modelos_usac = []
    modelos_marroquin = []
    modelos_landivar = []
    modelos_mariano = []

    print("generando...")

    for uni in universidades:
        # obtener datos
        train, test, train_x, train_y, test_x, test_y, clases = File.getImagenes("AppServer/Imagenes/", uni, 0.7)
        train_set = Data(train_x, train_y, 255)
        test_set = Data(test_x, test_y, 255)

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

        if uni == "USAC":
            modelos_usac.append(modelo1)
            modelos_usac.append(modelo2)
            modelos_usac.append(modelo3)
            modelos_usac.append(modelo4)
            modelos_usac.append(modelo5)
        elif uni == "Marroquin":
            modelos_marroquin.append(modelo1)
            modelos_marroquin.append(modelo2)
            modelos_marroquin.append(modelo3)
            modelos_marroquin.append(modelo4)
            modelos_marroquin.append(modelo5)
        elif uni == "Landivar":
            modelos_landivar.append(modelo1)
            modelos_landivar.append(modelo2)
            modelos_landivar.append(modelo3)
            modelos_landivar.append(modelo4)
            modelos_landivar.append(modelo5)
        elif uni == "Mariano":
            modelos_mariano.append(modelo1)
            modelos_mariano.append(modelo2)
            modelos_mariano.append(modelo3)
            modelos_mariano.append(modelo4)
            modelos_mariano.append(modelo5)

    Plotter.show_Model(modelos_usac)
    Plotter.show_Model(modelos_marroquin)
    Plotter.show_Model(modelos_landivar)
    Plotter.show_Model(modelos_mariano)

    file = open("modelos.txt", "a")
    file.write("#####################################################\n")
    file.write("Modelos Usac\n")
    for i in range(5):
        file.write("Modelo " + str(i+1) + "\n" )
        file.write("Entrenamiento " + str(modelos_usac[i].train_accuracy) + "\n" )
        file.write("Validacion " + str(modelos_usac[i].test_accuracy) + "\n" )
    file.write("#####################################################\n\n")

    file.write("#####################################################\n")
    file.write("Modelos Marroquin\n")
    for i in range(5):
        file.write("Modelo " + str(i+1) + "\n" )
        file.write("Entrenamiento " + str(modelos_marroquin[i].train_accuracy) + "\n" )
        file.write("Validacion " + str(modelos_marroquin[i].test_accuracy) + "\n" )
    file.write("#####################################################\n\n")

    file.write("#####################################################\n")
    file.write("Modelos Landivar\n")
    for i in range(5):
        file.write("Modelo " + str(i+1) + "\n" )
        file.write("Entrenamiento " + str(modelos_landivar[i].train_accuracy) + "\n" )
        file.write("Validacion " + str(modelos_landivar[i].test_accuracy) + "\n" )
    file.write("#####################################################\n\n")

    file.write("#####################################################\n")
    file.write("Modelos Mariano\n")
    for i in range(5):
        file.write("Modelo " + str(i+1) + "\n" )
        file.write("Entrenamiento " + str(modelos_mariano[i].train_accuracy) + "\n" )
        file.write("Validacion " + str(modelos_mariano[i].test_accuracy) + "\n\n" )
    file.write("#####################################################\n\n")
    file.close()
    """
    return "ok"

if __name__ == "__main__":
    app.run(debug=True)





    """
    modelo1 = Model(train_set, test_set, reg=False, alpha=0.0000001, lam=120, max_iterations=15000)#, max_iterations=10000, min_value=0.5)
    modelo1.training(print_training=False)

    modelo2 = Model(train_set, test_set, reg=False, alpha=0.01, lam=180, max_iterations=10000)#, max_iterations=10000, min_value=0.5)
    modelo2.training(print_training=False)

    modelo3 = Model(train_set, test_set, reg=True, alpha=0.00001, lam=300, max_iterations=12000)#, max_iterations=10000, min_value=0.5)
    modelo3.training(print_training=False)
    """

    """
    modelo1 = Model(train_set, test_set, reg=False, alpha=0.00001, lam=120, max_iterations=7000)#, max_iterations=10000, min_value=0.5)
    modelo1.training(print_training=False)

    modelo2 = Model(train_set, test_set, reg=False, alpha=0.00008, lam=0.01, max_iterations=4000)#, max_iterations=10000, min_value=0.5)
    modelo2.training(print_training=False)

    modelo3 = Model(train_set, test_set, reg=False, alpha=0.05, lam=4, max_iterations=9000)#, max_iterations=10000, min_value=0.5)
    modelo3.training(print_training=False)

    modelo4 = Model(train_set, test_set, reg=False, alpha=0.000025, lam=180, max_iterations=10000)#, max_iterations=10000, min_value=0.5)
    modelo4.training(print_training=False)

    modelo5 = Model(train_set, test_set, reg=False, alpha=0.5, lam=7, max_iterations=800)#, max_iterations=10000, min_value=0.5)
    modelo5.training(print_training=False)
    """