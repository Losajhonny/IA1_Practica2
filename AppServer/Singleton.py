from Logistic_Regression.Model import Model

class Singleton:
    __instance = None

    modelo_usac = None
    modelo_mariano = None
    modelo_marroquin = None
    modelo_landivar = None
    
    @staticmethod
    def getInstance():
        if Singleton.__instance == None:
            Singleton()
        return Singleton.__instance
        
    def __init__(self):
        """ Virtually private constructor. """
        if Singleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Singleton.__instance = self
