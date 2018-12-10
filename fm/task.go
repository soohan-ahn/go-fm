package fm

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
)

func Sigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}

func CalcSums(matrix [][]float64, features []uint32, kFactor int) ([]float64, []float64) {
	sum := make([]float64, kFactor)
	squareSum := make([]float64, kFactor)
	for i := 0; i < kFactor; i++ { // factor
		for _, f := range features {
			v := matrix[i][f]
			sum[i] += v
			squareSum[i] += v * v
		}
	}

	return sum, squareSum
}

func PredictAll(f *FM) {
	p := f.Params
	// Extract from csv file.
	// extractFeatures extract features from a csv file.
	// Save a hashed value of each feature on the slice and returns it.
	// ex) A,NH,2010-2,3
	file, err := os.Open(*(p.Files.PredictFileName))
	if err != nil {
		return
	}
	defer file.Close()
	r := csv.NewReader(bufio.NewReader(file))

	for {
		line, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("Err: %v\n", err)
			continue
		}

		id := line[0]

		feature := ReadLine(f, line)
		sum, squareSum := CalcSums(f.InterWeights, feature, *f.Params.FM.KFactor)
		predicted := f.Predict(feature, sum, squareSum)
		fmt.Printf("%v,%v\n", id, predicted)
	}
}

func Train(f *FM) {
	p := f.Params
	file, err := os.Open(*(p.Files.TrainDataFileName))
	if err != nil {
		return
	}
	defer file.Close()

	cpuNum := runtime.NumCPU()
	epoch := f.Params.FM.Epoch
	for t := 0; t < epoch; t++ {
		file.Seek(0, 0)
		r := csv.NewReader(bufio.NewReader(file))
		hasDone := false
		lineNum := 0

		for {
			var wg sync.WaitGroup

			for i := 0; i < cpuNum; i++ {
				line, err := r.Read()
				if err == io.EOF {
					hasDone = true
					break
				}
				if err != nil {
					log.Printf("Err: %v\n", err)
					continue
				}

				wg.Add(1)
				go func(coreNum int) {
					defer wg.Done()
					lineNum++
					log.Printf("Training line: %d, on core: %d\n", lineNum, coreNum)

					label, err := strconv.ParseFloat(line[len(line)-1], 3)
					if err != nil {
						log.Printf("Err: %v\n", err)
						return
					}

					line = line[0 : len(line)-1]
					feature := ReadLine(f, line)

					sum, squareSum := CalcSums(f.InterWeights, feature, *f.Params.FM.KFactor)
					predicted := f.Predict(feature, sum, squareSum)

					f.Fit(feature, label, predicted, sum)
				}(i)
			}

			wg.Wait()
			if hasDone {
				break
			}
		}
	}

	SaveWeights(f.Weights, f.InterWeights, f.Params.Files)
}
