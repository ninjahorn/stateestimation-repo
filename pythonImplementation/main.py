import random
import numpy as np
from multiprocessing import Pool, cpu_count
import time


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
        matrix = pool.starmap(generateLine, [(sizeY,)] * sizeX)
    return matrix


def compareImplementations(size):
    start_time = time.time()
    matrix1 = np.array(initRandMatrixSequential(size))
    sequential_time = time.time() - start_time

    start_time = time.time()
    matrix2 = np.array(initRandMatrixParallel(size))
    parallel_time = time.time() - start_time

    print(f"Matrix size: {size}x{size}")
    print(f"Sequential time: {sequential_time:.4f} seconds")
    print(f"Parallel time: {parallel_time:.4f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    print()


if __name__ == '__main__':
    for size in [100, 500, 1000, 2000, 4000]:
        compareImplementations(size)
