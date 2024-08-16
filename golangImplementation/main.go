package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

func initRandMatrixSequential(sizeX, sizeY int) [][]float64 {
	matrix := make([][]float64, sizeX)
	for i := 0; i < sizeX; i++ {
		line := make([]float64, sizeY)
		for j := 0; j < sizeY; j++ {
			num := float64(rand.Intn(101)) / 10
			if num == 0 {
				num = 0.1
			}
			line[j] = num
		}
		matrix[i] = line
	}
	return matrix
}

func generateLine(sizeY int, wg *sync.WaitGroup, ch chan<- []float64) {
	defer wg.Done()
	line := make([]float64, sizeY)
	for j := 0; j < sizeY; j++ {
		num := float64(rand.Intn(101)) / 10
		if num == 0 {
			num = 0.1
		}
		line[j] = num
	}
	ch <- line
}

func initRandMatrixParallel(sizeX, sizeY int) [][]float64 {
	matrix := make([][]float64, sizeX)
	wg := sync.WaitGroup{}
	ch := make(chan []float64, sizeX)

	for i := 0; i < sizeX; i++ {
		wg.Add(1)
		go generateLine(sizeY, &wg, ch)
	}

	wg.Wait()
	close(ch)

	i := 0
	for line := range ch {
		matrix[i] = line
		i++
	}

	return matrix
}

func compareImplementations(size int) {
	startTime := time.Now()
	matrix1 := initRandMatrixSequential(size, size)
	sequentialTime := time.Since(startTime).Seconds()

	startTime = time.Now()
	matrix2 := initRandMatrixParallel(size, size)
	parallelTime := time.Since(startTime).Seconds()

	fmt.Printf("Matrix size: %dx%d\n", size, size)
	fmt.Printf("Sequential time: %.4f seconds\n", sequentialTime)
	fmt.Printf("Parallel time: %.4f seconds\n", parallelTime)
	fmt.Printf("Speedup: %.2fx\n", sequentialTime/parallelTime)
	fmt.Println()

	// Prevent optimization from removing the generated matrices
	_ = matrix1
	_ = matrix2
}

func main() {
	sizes := []int{100, 500, 1000, 2000, 4000, 11000}
	for _, size := range sizes {
		compareImplementations(size)
	}
}
