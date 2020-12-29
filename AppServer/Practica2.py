from flask import Flask, request, jsonify
from flask_cors import CORS
from Singleton import *
from Generacion import *
from FileManagement import File
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np

from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return 'Bienvenido'

@app.route('/analizar', methods=['POST'])
def analizar():
    #recuperar json
    data = request.json

    #recuperar notas
    dataImages = data['images']

    posUsac = 0
    posMariano = 1
    posMarroquin = 2
    posLandivar = 3

    totales = [0, 0, 0, 0]
    numeros = [0, 0, 0, 0]
    
    predicciones = []
    porcentajes = []

    for dataImage in dataImages:
        b64 = dataImage["base64"].split(",")[1]
        decoded = base64.b64decode(b64)
        
        img = Image.open(BytesIO(decoded))
        name = dataImage["name"].lower().split("_")[0]

        gen = Generacion()
        
        pusac = gen.predict(img, Singleton.getInstance().modelo_usac)
        pmari = gen.predict(img, Singleton.getInstance().modelo_mariano)
        pmarr = gen.predict(img, Singleton.getInstance().modelo_marroquin)
        pland = gen.predict(img, Singleton.getInstance().modelo_landivar)

        nombrePrediccion = ""

        if name == "usac":
            totales[posUsac] += 1
        elif name == "mariano":
            totales[posMariano] += 1
        elif name == "marroquin":
            totales[posMarroquin] += 1
        elif name == "landivar":
            totales[posLandivar] += 1

        if pusac:
            nombrePrediccion = "Usac"
            if name == "usac":
                numeros[posUsac] += 1
        elif pmari:
            nombrePrediccion = "Mariano"
            if name == "mariano":
                numeros[posMariano] += 1
        elif pmarr:
            nombrePrediccion = "Marroquin"
            if name == "marroquin":
                numeros[posMarroquin] += 1
        elif pland:
            nombrePrediccion = "Landivar"
            if name == "landivar":
                numeros[posLandivar] += 1
        else:
            nombrePrediccion = "Prediccion no encontrada"

        predicciones.append([nombrePrediccion, [pusac, pmari, pmarr, pland]])
    
    porUsac = (numeros[posUsac] / (1 if totales[posUsac] == 0 else totales[posUsac])) * 100
    porMari = (numeros[posMariano] / (1 if totales[posMariano] == 0 else totales[posMariano])) * 100
    porMarr = (numeros[posMarroquin] / (1 if totales[posMarroquin] == 0 else totales[posMarroquin])) * 100
    porLand = (numeros[posLandivar] / (1 if totales[posLandivar] == 0 else totales[posLandivar])) * 100
    
    porcentajes.append(['Usac', porUsac])
    porcentajes.append(['Mariano', porMari])
    porcentajes.append(['Landivar', porLand])
    porcentajes.append(['Marroquin', porMarr])

    #respuesta
    return jsonify({ 'status' : '200', 'predicciones': predicciones, 'porcentajes': porcentajes })

@app.route('/generarModelo', methods=['GET'])
def generarModelo():
    print("\n\ngenerando modelos...")
    gen = Generacion()
    gen.generar(0.70)

    Singleton.getInstance().modelo_usac = gen.mejor_usac[0]
    Singleton.getInstance().modelo_marroquin = gen.mejor_marroquin[0]
    Singleton.getInstance().modelo_mariano = gen.mejor_mariano[0]
    Singleton.getInstance().modelo_landivar = gen.mejor_landivar[0]

    #print(Singleton.getInstance().modelo_usac.train_accuracy)
    #print(Singleton.getInstance().modelo_marroquin.train_accuracy)
    #print(Singleton.getInstance().modelo_mariano.train_accuracy)
    #print(Singleton.getInstance().modelo_landivar.train_accuracy)

    file = open("modelos.txt", "a")
    for i in range(len(gen.universidades)):
        file.write("#####################################################\n")
        file.write("Modelos " + gen.universidades[i] + "\n")

        modelos = []
        if gen.universidades[i] == "USAC":
            modelos = gen.modelos_usac
        elif gen.universidades[i] == "Marroquin":
            modelos = gen.modelos_marroquin
        elif gen.universidades[i] == "Mariano":
            modelos = gen.modelos_mariano
        else:
            modelos = gen.modelos_landivar

        for j in range(len(modelos)):
            file.write("Modelo " + str(j+1) + "\n" )
            file.write("Entrenamiento " + str(round(modelos[j].train_accuracy, 2)) + "\n" )
            file.write("Validacion " + str(round(modelos[j].test_accuracy, 2)) + "\n\n" )
        file.write("#####################################################\n\n")
    file.close()
    print("generacion terminada\n\n")
    return jsonify({ 'status': '200' })

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