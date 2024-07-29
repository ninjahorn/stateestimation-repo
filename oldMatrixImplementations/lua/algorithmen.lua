require "math"

math.randomseed(os.time())

function determinant(matrix) -- es wird angenommen die Matrix beginnt mit Index 0 
    local length = #matrix[0] + 1    -- ein # vor einem array returned den letzten eingetragenen Index
    if (length == 2) then
        return (matrix[0][0]*matrix[1][1])-(matrix[0][1]*matrix[1][0])
    end
    local result = 0
    for x=0, (length-1) do -- x iteriert druch die oberste Zeile, also die Spalten
        local submatrix = createEmptyMatrix(length-1)
        for i=0, (length-2) do -- i iteriert durch die Zeilen
            for j=0, (length-2) do -- j iteriert durch die Spalten
                submatrix[i][(j+x) % (length-1)] = matrix[1+i][((1+x+j) % length)]
            end
        end
        result = result + ((-1)^x ) * matrix[0][x]*determinant(submatrix)
    end
    return result
end

function transpose(matrix)
    local xlength = #matrix + 1
    local ylength = #matrix[0] + 1

    local returnMatrix = createEmptyMatrix(ylength,xlength)

    for y=0, (ylength-1) do
        for x=0, (xlength-1) do
            returnMatrix[y][x] = matrix[x][y]
        end
    end
    return returnMatrix
end

function inverse(matrix)
    local length =  #matrix[0] + 1
    local returnMatrix = createEmptyMatrix(length)
    local det = determinant(matrix)

    if length == 2 then
        returnMatrix[0][0] = matrix[1][1]/det
        returnMatrix[1][1] = matrix[0][0]/det

        returnMatrix[0][1] = -(matrix[0][1]/det)
        returnMatrix[1][0] = -(matrix[1][0]/det)
        return returnMatrix
    end

    for x=0, (length-1) do
        for y=0, (length-1) do
            local subMatrix = createEmptyMatrix(length-1)
            for i=0, (length-2) do
                for j=0, (length-2) do
                    subMatrix[(i + y) % (length - 1)][(j + x) % (length - 1)] = matrix[(1 + x + j) % length][(1 + y + i) % length]
                end
            end
            returnMatrix[y][x] = ((-1)^(x+y))*(determinant(subMatrix)/det)
        end
    end

    return returnMatrix
end

function multiplyMatrix(firstMatrix,SecondMatrix)
    local matrix2 = transpose(SecondMatrix)
    local matrix1 = firstMatrix
    local matLen1 = #matrix1 + 1
    local matLen11 = #matrix1[0] + 1

    local mat2Len1 = #matrix2 + 1
    local mat2Len11 = #matrix2[0] + 1

    if matLen1 == mat2Len11 then
        matrix2 = transpose(matrix2)
        local temp = mat2Len1
        mat2Len1 = mat2Len11
        mat2Len11 = temp
    end

    if matLen11 == mat2Len1 then
        matrix1 = transpose(matrix1)
        local temp = matLen1
        matLen1 = matLen11
        matLen11 = temp
    end

    if matLen1 ~= mat2Len1 then -- to make != do ~=
        matrix1 = transpose(matrix1)
        local temp = matLen1
        matLen1 = matLen11
        matLen11 = temp        
        matrix2 = transpose(matrix2)
        temp = mat2Len1
        mat2Len1 = mat2Len11
        mat2Len11 = temp
    end

    local resultMatrix = createEmptyMatrix(matLen11,mat2Len11)
    if matLen1 == mat2Len1 then
        for x=0, (matLen11 - 1) do
            for y=0, (mat2Len11 - 1) do
                for m=0, (mat2Len1 - 1) do
                    resultMatrix[x][y] = (matrix1[m][x] * matrix2[m][y]) + resultMatrix[x][y]
                end
            end
        end
    else
        resultMatrix = {}
    end
    return resultMatrix
end

function gaussJordanInverse(oldMatrix)
    local matLen = #oldMatrix + 1
    local matrix = createEmptyMatrix(matLen,(matLen * 2))

    for i=0, (matLen - 1) do
        local einheitAppended = {}
        for i=0, ((matLen*2)-1) do 
            einheitAppended[i] = 0 
        end
        einheitAppended[matLen + i] = 1
        for j=0, (matLen - 1) do
            einheitAppended[j] = oldMatrix[i][j]
        end
        matrix[i] = einheitAppended
    end

    for i = (matLen - 1), 1, -1 do
        if matrix[i - 1][0] < matrix[i][0] then
            local tempArr = matrix[i]
            matrix[i] = matrix[i - 1]
            matrix[i - 1] = tempArr
        end
    end

    for i=0, (matLen - 1) do 
        for j=0, (matLen - 1) do
            if j ~= i then
                local temp = matrix[j][i] / matrix[i][i]
                for k=0, ((2*matLen) - 1) do
                    matrix[j][k] = matrix[j][k] - (matrix[i][k] * temp)
                end
            end
        end
    end

    for i=0, (matLen - 1) do
        local temp = matrix[i][i]
        for j=0, ((2*matLen) - 1) do
            matrix[i][j] = matrix[i][j] / temp
        end
    end

    local finMat = createEmptyMatrix(matLen,matLen)
    for i=0, (matLen - 1) do
        for j=0, (matLen - 1) do
            finMat[i][j] = matrix[i][j+matLen]
        end
    end
    return finMat
end

function createEmptyMatrix(p_ySize,p_xSize)
    local xSize = p_xSize or p_ySize
    local ySize = p_ySize
    local emptyMatrix = {}          -- create the matrix
    for i=0, (ySize - 1) do
        emptyMatrix[i] = {}     -- create a new row
      for j=0,(xSize - 1) do
        emptyMatrix[i][j] = 0
      end
    end
    return emptyMatrix
end

function printMatrix(matrix)
    io.write("   ")
    for h=0, #matrix[0] do
        io.write(h .. " ")
    end
    print()
    for i=0, #matrix do
        io.write(i .. ": ")
        for j=0, #matrix[0] do
            local number = matrix[i][j]
            local roundNumber = math.floor((math.floor(number*2) + 1)/2)
            io.write(roundNumber .. " ")
        end
        print()
    end
end

function math.round(num, decimals)
    decimals = math.pow(10, decimals or 0)
    num = num * decimals
    if num >= 0 then num = math.floor(num + 0.5) else num = math.ceil(num - 0.5) end
    return num / decimals
end

-- Convert a Matrix with index: 1 to starterindex: 0
function createMatrixWithIndexZero(matrix)
    local length_row = #matrix[1]
    local length_column = #matrix

    local new_matrix = {}

    for i = 0, (length_column - 1) do
        new_matrix[i] = {}
        for j = 0, (length_row - 1) do
            new_matrix[i][j] = matrix[i+1][j+1]
        end
    end

    return new_matrix
end

function randomMatrix(p_size) 
    local randomMatrix = {}

    for i = 0, (p_size - 1) do
        randomMatrix[i] = {}
        for j = 0, (p_size - 1) do
            randomMatrix[i][j] = (math.random(-100,100)/10)
        end
    end
    return randomMatrix
end

---------------------------------------------------------------------------------------------------
--Damit die Algorithmen gleich aufgebaut sind, wie in den anderen Sprachen musste hier ein kleiner Trick
--angewendet werden, dabei müssen für die Algorithmen die Arrays von 0 beginnen (Standartmäßig beginnen
--Arrays in Lua mit 1) Wenn eine eigene Matrize erstellt wird, so muss diese mit der Methode 
--createMatrixWithIndexZero(matrix) umgewandelt werden, wenn eine Zufällige Matrix ausreicht, so kann
--die Methode randomMatrix(p_size) verwendet werden, dabei übergibt man einfach die Größe die man haben
--will, also eine 2x2 Matrix --> randomMatrix(2) 