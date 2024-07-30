import random
import numpy as np

def initRandArray(sizeX, sizeY=None):
    if sizeY is None:
        sizeY = sizeX
    a = []
    for ignore in range(sizeX):
        line = []
        for y in range(sizeY):
            zahl=random.randint(0,100)/10
            if(zahl==0):
                zahl = 2.0
            line.append(zahl)
        a.append(line)
    return a


matrix1 = initRandArray(3,3)

print(matrix1)