package fm

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type FM struct {
	NumOfFeatures int
	W0            float64
	Weights       []float64
	InterWeights  [][]float64
	Optimizer     AbstractFMOptimizer
	Sum           []float64
	SquareSum     []float64
}

func (fm *FM) Init(p Params) {
	fm.InitWeights(p)
	fm.Optimizer = LogisticFMSGD{
		LearnRate: 0.1,
		TrainR0:   *p.TrainR0,
		TrainRW:   *p.TrainRW,
		TrainRV:   *p.TrainRV,
	}
}

// InitWeights Read weights from file or set random weights initially.
func (fm *FM) InitWeights(p Params) {
	fm.W0 = 0.0
	fm.Weights = make([]float64, *p.MaxDimension)
	for i := range fm.Weights {
		sample := rand.NormFloat64() * 0.05
		fm.Weights[i] = sample
	}
	fm.InterWeights = make([][]float64, *p.MaxDimension)
	for i := range fm.InterWeights {
		fm.InterWeights[i] = make([]float64, *p.MaxNonzeroDimension)
		for j := range fm.InterWeights[i] {
			sample := rand.NormFloat64() * 0.05
			fm.InterWeights[i][j] = sample
		}
	}

	// Format of weight file: [feature]:[weight]
	// ex)
	//  1:2.355
	//  1643:2.1011
	dat, err := ioutil.ReadFile(*(p.WeightFileName))
	if err != nil {
		log.Printf("Err: %v\n", err)
		return
	}

	datStr := string(dat)
	lines := strings.Split(datStr, "\n")
	for _, line := range lines {
		weights := strings.Split(line, ":")
		feature, err := strconv.Atoi(weights[0])
		if err != nil {
			log.Printf("Weight Err: %v\n", err)
			return
		}
		weight, err := strconv.ParseFloat(weights[1], 64)
		if err != nil {
			log.Printf("Err: %v\n", err)
			return
		}
		fm.Weights[feature] = weight
	}

	// Format of inter weight file: [feature]:[array of [feature]:[weight]]
	// ex)
	//  1:[2:2.355,3:1.111,4:4.567]
	//  1643:[9:2.1011,11:3.14]

	dat, err = ioutil.ReadFile(*(p.InterWeightFileName))
	if err != nil {
		log.Printf("Err: %v\n", err)
		return
	}

	datStr = string(dat)
	lines = strings.Split(datStr, "\n")
	for _, line := range lines {
		featureRe := regexp.MustCompile(`^[0-9]+`)
		feature, err := strconv.Atoi(featureRe.FindAllString(line, -1)[0])
		if err != nil {
			log.Printf("InterWeight Err: %v\n", err)
			continue
		}
		targetsRe := regexp.MustCompile(`\[(.*?)\]`)
		targetStr := targetsRe.FindAllStringSubmatch(line, -1)[0][1]

		targets := strings.Split(targetStr, ",")
		for _, target := range targets {
			s := strings.Split(target, ":")

			targetFeature, err := strconv.Atoi(s[0])
			if err != nil {
				log.Print("Err: %v\n", err)
				return
			}

			w, err := strconv.ParseFloat(s[1], 64)
			if err != nil {
				log.Print("Err: %v\n", err)
				return
			}
			fm.InterWeights[feature][targetFeature] = w
		}
	}
}

func (fm *FM) Predict(features []uint32, sum []float64, squareSum []float64, p Params) float64 {
	result := 0.0
	for _, x := range features {
		result += fm.Weights[x]
	}

	for i := 0; i < *(p.MaxDimension); i++ {
		result += ((sum[i]*sum[i] - squareSum[i]) * 0.5)
	}

	return result
	//return fm.ActivationFunction(result)
}

func (fm *FM) CalcSums(features []uint32, p Params) ([]float64, []float64) {
	sum := make([]float64, *(p.MaxDimension))
	squareSum := make([]float64, *(p.MaxDimension))
	for _, f := range features {
		for i := 0; i < *p.MaxNonzeroDimension; i++ {
			sum[i] += fm.InterWeights[f][i]
			squareSum[i] += (fm.InterWeights[f][i] * fm.InterWeights[f][i])
		}
	}

	return sum, squareSum
}

func (fm *FM) Train(features []uint32, label float64, delta float64, sum []float64) {
	fm.Optimizer.LearnW0(&fm.W0, delta)
	fm.Optimizer.LearnW(fm.Weights, features, delta, label)
	fm.Optimizer.LearnV(fm.InterWeights, features, delta, label, sum)
}

func (fm *FM) SaveWeights(p Params) {
	wfile, err := os.OpenFile(*(p.WeightFileName), os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Err: %v\n", err)
		return
	}
	ww := bufio.NewWriter(wfile)
	log.Printf("Saving weights..\n")
	for i := range fm.Weights {
		line := fmt.Sprintf("%d:%f\n", i, fm.Weights[i])
		_, err := ww.Write([]byte(line))
		if err != nil {
			log.Printf("Err: %v\n", err)
		}
	}
	ww.Flush()
	wfile.Sync()

	file, err := os.OpenFile(*(p.InterWeightFileName), os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Err: %v\n", err)
		return
	}
	w := bufio.NewWriter(file)
	// Format of inter weight file: [feature]:[array of [feature]:[weight]]
	// ex)
	//  1:[2:2.355,3:1.111,4:4.567]
	//  1643:[9:2.1011,11:3.14]
	log.Printf("Saving Interweights..\n")
	for i := range fm.InterWeights {
		str := ""
		for j := range fm.InterWeights[i] {
			if fm.InterWeights[i][j] != 0.0 {
				var s string
				if str == "" {
					s = fmt.Sprintf("%d:%f", j, fm.InterWeights[i][j])
				} else {
					s = fmt.Sprintf(",%d:%f", j, fm.InterWeights[i][j])
				}
				str += s
			}
		}
		if str != "" {
			line := fmt.Sprintf("%d:[%s]\n", i, str)
			_, err := w.Write([]byte(line))
			if err != nil {
				log.Printf("Err: %v\n", err)
			}
		}
	}
	w.Flush()
	file.Sync()
}
