package main

import (
	"flag"
	"fmt"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
	"log"
	"sync"
	"time"
)

func main() {
	// Parse command line arguments, Number of Goroutines and Size of the slices
	numWorkers := flag.Int("n", 8, "number of workers")
	size := flag.Int("s", 400_000_000, "size of the slices")

	flag.Parse()
	// Example data with very large slices
	masses := make([]float64, *size)     // large slice of masses
	velocities := make([]float64, *size) // large slice of velocities

	// Initializing the slices with random values
	//rand.New(rand.NewSource(time.Now().UnixNano())) // Seed the random number generator
	for i := 0; i < *size; i++ {
		// Generate a random float64 number between 0 and 1 and scale and shift the number to the range 0.1 to 200
		//randomFloat := rand.Float64()*199.9 + 0.1
		//masses[i] = randomFloat     // example masses
		//velocities[i] = randomFloat // example velocities
		masses[i] = float64(i%100 + 1)     // example masses
		velocities[i] = float64(i%100 + 2) // example velocities
	}

	// Check if the lengths of the slices are equal
	if len(masses) != len(velocities) {
		fmt.Println("Error: Masses and velocities slices must have the same length")
		return
	}

	timestamp := time.Now()
	// Calculate kinetic energy concurrently
	kineticEnergy := calculateKineticEnergyConcurrently(masses, velocities, *numWorkers)

	totalTime := time.Since(timestamp)

	// Print result in a readable format
	p := message.NewPrinter(language.English)
	_, err := p.Printf("The total kinetic energy is: %f\n", kineticEnergy)
	if err != nil {
		log.Println(err)
	}
	_, err = p.Println("Total time:", totalTime)
	if err != nil {
		return
	}
}

// calculateKineticEnergyConcurrently calculates the kinetic energy using multiple goroutines
func calculateKineticEnergyConcurrently(masses, velocities []float64, numWorkers int) float64 {
	var wg sync.WaitGroup
	energyChan := make(chan float64, numWorkers)

	chunkSize := (len(masses) + numWorkers - 1) / numWorkers // ensures that the division rounds up

	for i := 0; i < len(masses); i += chunkSize {
		end := i + chunkSize
		if end > len(masses) {
			end = len(masses)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			sum := 0.0
			for j := start; j < end; j++ {
				sum += 0.5 * masses[j] * velocities[j] * velocities[j]
			}
			energyChan <- sum
		}(i, end)
	}

	go func() {
		wg.Wait()
		close(energyChan)
	}()

	totalEnergy := 0.0
	for energy := range energyChan {
		totalEnergy += energy
	}

	return totalEnergy
}
