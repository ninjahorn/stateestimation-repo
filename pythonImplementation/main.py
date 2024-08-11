import random
import numpy as np
from multiprocessing import Pool, cpu_count

def initRandMatrixSequential(sizeX, sizeY=None):
    if sizeY is None:
        sizeY = sizeX
    matrix = []
    for _ in range(sizeX):
        line = []
        for y in range(sizeY):
            zahl=random.randint(0,100)/10
            if(zahl==0):
                zahl = 0.1
            line.append(zahl)
        matrix.append(line)
    return matrix

def generateLine(sizeY):
    line = []
    for _ in range(sizeY):
        zahl = random.randint(0,100)/10
        if(zahl==0):
            zahl = 0.1
        line.append(zahl)
    return line

def initRandMatrixParallel(sizeX, sizeY=None):
    if sizeY is None:
        sizeY = sizeX

    with Pool(processes=cpu_count()) as pool:
        matrix = pool.map(lambda _: generateLine(sizeY), range(sizeX))
    
    return matrix

matrix1 = np.array(initRandMatrixSequential(8))
print(matrix1)

matrix2 = np.array(initRandMatrixParallel(8))
print(matrix2)


