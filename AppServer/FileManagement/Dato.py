import numpy as np

class Dato:
    def __init__(self, x=[], y=0):
        self.x = x
        self.y = y
        self.arr = None
        self.reshape = None

    def setReshape(self):
        self.arr = np.array(self.x)
        self.reshape = self.arr.reshape(self.arr.shape[0], -1).T