package fm

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

// InitWeights Read weights from file or set random weights initially.
func LoadWeights(fmp *FMParams, fp *FileParams) (float64, []float64, [][]float64) {
	// Format of weight file: [feature]:[weight]
	// ex)
	//  1:2.355
	//  1643:2.1011
	w0 := 0.0
	var vector []float64
	var matrix [][]float64

	if fp == nil || fmp == nil {
		return w0, vector, matrix
	}

	if fp.WeightFileName != nil {
		dat, err := ioutil.ReadFile(*fp.WeightFileName)
		if err != nil {
			log.Printf("Err: %v\n", err)
			return w0, vector, matrix
		}

		vector = make([]float64, *fmp.MaxDimension)
		datStr := string(dat)
		lines := strings.Split(datStr, "\n")
		for i, line := range lines {
			weights := strings.Split(line, ":")
			feature, err := strconv.Atoi(weights[0])
			if err != nil {
				log.Printf("Weight Err: %v, on line: %v, weights: %v\n", err, i, weights)
				continue
			}
			weight, err := strconv.ParseFloat(weights[1], 64)
			if err != nil {
				log.Printf("Err: %v\n", err)
				continue
			}
			vector[feature] = weight
		}
	}

	// Format of inter weight file: [feature]:[array of [feature]:[weight]]
	// ex)
	//  1:[2:2.355,3:1.111,4:4.567]
	//  1643:[9:2.1011,11:3.14]

	if fp.InterWeightFileName != nil {
		dat, err := ioutil.ReadFile(*fp.InterWeightFileName)
		if err != nil {
			log.Printf("Err: %v\n", err)
			return w0, vector, matrix
		}

		matrix := make([][]float64, *fmp.KFactor)
		for i, _ := range matrix {
			matrix[i] = make([]float64, *fmp.MaxDimension)
		}

		datStr := string(dat)
		lines := strings.Split(datStr, "\n")
		for _, line := range lines {
			splittedLine := strings.Split(line, "|")
			if len(splittedLine) != 2 {
				log.Printf("InterWeight Err: Invalid length of splitted line: %v\n", line)
				continue
			}
			feature, err := strconv.Atoi(splittedLine[0])
			if err != nil {
				log.Printf("InterWeight Err: %v\n", err)
				continue
			}

			interWeights := strings.Split(splittedLine[1], ",")
			for _, target := range interWeights {
				s := strings.Split(target, ":")
				if len(s) != 2 {
					log.Printf("InterWeight Err: Invalid length of interWeight: %v\n", target)
					continue
				}

				targetFeature, err := strconv.Atoi(s[0])
				if err != nil {
					log.Printf("Err: %v\n", err)
					return w0, vector, matrix
				}

				w, err := strconv.ParseFloat(s[1], 64)
				if err != nil {
					log.Printf("Err: %v\n", err)
					return w0, vector, matrix
				}
				matrix[feature][targetFeature] = w
			}
		}
	}
	return w0, vector, matrix
}

func SaveWeights(vector []float64, matrix [][]float64, p *FileParams) {
	wfile, err := os.OpenFile(*(p.WeightFileName), os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Err: %v\n", err)
		return
	}
	defer wfile.Close()
	ww := bufio.NewWriter(wfile)
	log.Printf("Saving weights..\n")
	for i := range vector {
		line := fmt.Sprintf("%d:%f\n", i, vector[i])
		_, err := ww.Write([]byte(line))
		if err != nil {
			log.Printf("Err: %v\n", err)
		}
	}
	line := fmt.Sprintf("\n")
	_, err = ww.Write([]byte(line))
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
	for i := range matrix {
		str := ""
		for j := range matrix[i] {
			if matrix[i][j] != 0.0 {
				var s string
				if str == "" {
					s = fmt.Sprintf("%d:%f", j, matrix[i][j])
				} else {
					s = fmt.Sprintf(",%d:%f", j, matrix[i][j])
				}
				str += s
			}
		}
		if str != "" {
			line := fmt.Sprintf("%d|%s\n", i, str)
			_, err := w.Write([]byte(line))
			if err != nil {
				log.Printf("Err: %v\n", err)
			}
		}
	}
	line = fmt.Sprintf("\n")
	_, err = w.Write([]byte(line))
	w.Flush()
	file.Sync()
}
