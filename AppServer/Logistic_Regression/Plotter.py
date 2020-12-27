from Logistic_Regression.Model import Model
import matplotlib.pyplot as chart


def show_picture(pixels):
    chart.imshow(pixels)
    chart.show()


def show_Model(models, title=""):
    for i in range(len(models)):
        model = models[i]
        label = "Modelo " + str(i+1) + ", alpha " + str(model.alpha)
        chart.plot(model.bitacora, label=label)
    chart.title(title)
    chart.ylabel('Costo')
    chart.xlabel('Iteraciones')
    legend = chart.legend(loc='upper center', shadow=True)
    chart.show()
