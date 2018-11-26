package fm

import (
	"bufio"
	"encoding/csv"
	"hash/adler32"
	"io"
	"log"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
)

type Params struct {
	TrainAlpha          *float64
	TrainBeta           *float64
	TrainL1             *float64
	TrainL2             *float64
	Degree              *int
	MaxNonzeroDimension *int
	MaxDimension        *int
	WeightFileName      *string
	InterWeightFileName *string
	PredictFileName     *string
	TrainDataFileName   *string
	TrainR0             *float64
	TrainRW             *float64
	TrainRV             *float64
}

type AbstractFM interface {
	Init(p Params)
	CalcSums(features []uint32, p Params) ([]float64, []float64)
	Predict(features []uint32, sum []float64, squareSum []float64, p Params) float64
	Train(features []uint32, label float64, delta float64, sum []float64)
	SaveWeights(p Params)
}

func Init(f AbstractFM, p Params) {
	f.Init(p)
}

func Sigmoid(val float64) float64 {
	// SIGMOID
	return 1.0 / (1.0 + math.Exp(-1*val))
}

func Predict(f AbstractFM, p Params) {
	// Extract from csv file.
	// extractFeatures extract features from a csv file.
	// Save a hashed value of each feature on the slice and returns it.
	// ex) A,NH,2010-2,3
	file, err := os.Open(*(p.PredictFileName))
	if err != nil {
		return
	}
	r := csv.NewReader(bufio.NewReader(file))

	features := [][]uint32{}
	pos := 0
	neg := 0
	tot := 0
	for {
		weights, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Printf("Err: %v\n", err)
			continue
		}
		tot += 1

		feature := []uint32{}
		for _, weight := range weights {
			//if i == (len(weights) - 1) {
			//	break
			//}
			featureNum := (adler32.Checksum([]byte(weight)) / 10) % uint32(*p.MaxDimension)
			feature = append(feature, featureNum)
		}
		features = append(features, feature)
		sum, squareSum := f.CalcSums(feature, p)
		predicted := f.Predict(feature, sum, squareSum, p)
		sign := 1.0
		if predicted <= 0.0 {
			sign = -1.0
			neg += 1
		} else {
			pos += 1
		}
		log.Printf("%v,%v\n", feature, sign)
	}
	log.Printf("Pos: %v, Neg: %v, Tot: %v\n", pos, neg, tot)
}

func Train(f AbstractFM, p Params) {
	cpuNum := runtime.NumCPU()
	for t := 0; t < 2; t++ {
		file, err := os.Open(*(p.TrainDataFileName))
		if err != nil {
			return
		}
		r := csv.NewReader(bufio.NewReader(file))
		hasDone := false
		lineNum := 0

		for {
			var wg sync.WaitGroup

			for i := 0; i < cpuNum; i++ {
				weights, err := r.Read()
				if err != nil {
					if err == io.EOF {
						hasDone = true
						break
					}
					log.Printf("Err: %v\n", err)
					continue
				}

				wg.Add(1)
				go func(coreNum int) {
					defer wg.Done()
					lineNum++
					log.Printf("Training line: %d, on core: %d\n", lineNum, coreNum)

					feature := []uint32{}
					label, err := strconv.ParseFloat(weights[len(weights)-1], 3)
					if err != nil {
						log.Printf("Err: %v\n", err)
						return
					}

					weights = weights[0 : len(weights)-1]
					for _, weight := range weights {
						featureNum := (adler32.Checksum([]byte(weight)) / 10) % uint32(*p.MaxDimension)
						feature = append(feature, featureNum)
					}

					sum, squareSum := f.CalcSums(feature, p)
					predicted := f.Predict(feature, sum, squareSum, p)
					//return Sigmoid(result)
					//delta = (0.0f - sigmoid(-mFuncValue * y) * y)
					//delta := 0.0 - Sigmoid(-predicted*label)*label
					delta := -label * (1.0 - Sigmoid(label*-predicted))
					f.Train(feature, label, delta, sum)
				}(i)
			}

			wg.Wait()
			if hasDone {
				break
			}
		}
	}
	f.SaveWeights(p)
}
