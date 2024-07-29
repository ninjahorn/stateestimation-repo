from numba import njit

import random
import numpy as np
import time


@njit
def determinant(matrix):
    length = len(matrix[0])
    if length == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    result = 0
    for x in range(length):
        subMatrix = np.zeros(shape=(length - 1, length - 1))
        for i in range(length - 1):
            for j in range(length - 1):
                subMatrix[i][(j + x) % (length - 1)] = matrix[1 + i][(1 + x + j) % length]

        result += ((-1) ** x) * matrix[0][x] * determinant(subMatrix)

    return result


@njit
def transpose(matrix):
    xLen = len(matrix)
    yLen = len(matrix[0])

    returnMatrix = np.zeros(shape=(yLen, xLen))
    for y in range(yLen):
        for x in range(xLen):
            returnMatrix[y][x] = matrix[x][y]

    return returnMatrix


@njit
def inverse(matrix):
    length = len(matrix[0])
    returnMatrix = np.zeros(shape=(length, length))
    det = determinant(matrix)
    if length == 2:
        returnMatrix[0][0] = matrix[1][1] / det
        returnMatrix[1][1] = matrix[0][0] / det

        returnMatrix[0][1] = -(matrix[0][1] / det)
        returnMatrix[1][0] = -(matrix[1][0] / det)
        return returnMatrix

    for x in range(length):
        for y in range(length):
            subMatrix = np.zeros(shape=(length - 1, length - 1))
            for i in range(length - 1):
                for j in range(length - 1):
                    subMatrix[(i + y) % (length - 1)][(j + x) % (length - 1)] = matrix[(1 + x + j) % length][
                        (1 + y + i) % length]
            returnMatrix[y][x] = ((-1) ** (x + y)) * (determinant(subMatrix) / det)

    return returnMatrix


@njit
def multiplyMatrix(matrix1, matrix2):
    matrix2 = transpose(matrix2)
    matLen1 = len(matrix1)
    matLen11 = len(matrix1[0])

    mat2Len1 = len(matrix2)
    mat2Len11 = len(matrix2[0])

    if matLen1 == mat2Len11:
        matrix2 = transpose(matrix2)
        mat2Len1, mat2Len11 = mat2Len11, mat2Len1

    if matLen11 == mat2Len1:
        matrix1 = transpose(matrix1)
        matLen1, matLen11 = matLen11, matLen1

    if matLen1 != mat2Len1:
        matrix1 = transpose(matrix1)
        matLen1, matLen11 = matLen11, matLen1
        matrix2 = transpose(matrix2)
        mat2Len1, mat2Len11 = mat2Len11, mat2Len1

    resultMatrix = np.zeros(shape=(matLen11, mat2Len11))
    if matLen1 == mat2Len1:
        for x in range(matLen11):
            for y in range(mat2Len11):
                for m in range(mat2Len1):
                    resultMatrix[x][y] += matrix1[m][x] * matrix2[m][y]

    else:
        resultMatrix = None
    return resultMatrix


@njit
def gaussJordanInverse(oldMatrix):
    matLen = len(oldMatrix)
    matrix = np.zeros(shape=(matLen, matLen * 2))
    for i in range(matLen):
        matrix[i][matLen + i] = 1.0
        for j in range(matLen):
            matrix[i][j] = oldMatrix[i][j]

    if oldMatrix[0][0] == 0:
        for x in range(matLen):
            if not (oldMatrix[x][0] == 0):
                matrix[x], matrix[0] = matrix[0], matrix[x]
                break

    for i in range(matLen):
        for j in range(matLen):
            if not (i == j):
                temp = matrix[j][i] / matrix[i][i]
                for k in range(matLen * 2):
                    matrix[j][k] -= matrix[i][k] * temp

    for i in range(matLen):
        temp2 = matrix[i][i]
        for j in range(matLen * 2):
            matrix[i][j] = matrix[i][j] / temp2

    invMat = np.zeros(shape=(matLen, matLen))
    for i in range(matLen):
        for j in range(matLen):
            invMat[i][j] = matrix[i][j + matLen]

    return invMat


@njit
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