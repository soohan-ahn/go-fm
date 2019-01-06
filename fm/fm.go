package fm

import (
	"log"
	"math/rand"
	"time"
)

type FM struct {
	W0           float64
	Weights      []float64   // Weights[f]: weight of the feature f.
	InterWeights [][]float64 // InterWeights[k][f]: weight between feature f and factor k.
	Optimizer    AbstractFMOptimizer
	Sum          []float64
	SquareSum    []float64
	Params       *Config
}

func (fm *FM) Init() error {
	var err error
	fm.Params, err = ReadParams()
	if err != nil {
		log.Printf("Err: %v\n", err)
		return err
	}
	fm.InitWeights()

	fm.Optimizer = LogisticFMSGD{
		LearnRate: 0.01,
		TrainR0:   fm.Params.SGD.TrainR0,
		TrainRW:   fm.Params.SGD.TrainRW,
		TrainRV:   fm.Params.SGD.TrainRV,
	}

	return nil
}

// InitWeights Read weights from file or set random weights initially.
func (fm *FM) InitWeights() {
	p := fm.Params
	fmp := p.FM
	w0, vector, matrix := LoadWeights(fmp, p.Files)

	fm.W0 = w0
	if vector != nil {
		fm.Weights = vector
	} else {
		fm.Weights = make([]float64, *fmp.MaxDimension)
		rand.Seed(time.Now().UTC().UnixNano())
		for i := range fm.Weights {
			fm.Weights[i] = 0.0
		}
	}

	if matrix != nil {
		fm.InterWeights = matrix
	} else {
		fm.InterWeights = make([][]float64, *fmp.KFactor)
		for i := range fm.InterWeights {
			fm.InterWeights[i] = make([]float64, *fmp.MaxDimension)
			for j := range fm.InterWeights[i] {
				sample := rand.NormFloat64() * 0.001
				fm.InterWeights[i][j] = sample
			}
		}
	}
}

func (fm *FM) MFunc(features []uint32, sum []float64, squareSum []float64) float64 {
	p := fm.Params.FM
	result := fm.W0
	for _, f := range features {
		result += fm.Weights[f]
	}

	for i := 0; i < *(p.KFactor); i++ {
		result += ((sum[i]*sum[i] - squareSum[i]) * 0.5)
	}

	return result
}

func (fm *FM) Predict(features []uint32, sum []float64, squareSum []float64) float64 {
	mFunc := fm.MFunc(features, sum, squareSum)
	result := 1.0
	if mFunc < 0.0 {
		result = -1.0
	}
	if *fm.Params.FM.Degree != 2 {
		if result > float64(*fm.Params.FM.Degree) {
			result = float64(*fm.Params.FM.Degree)
		}
		if result < 0 {
			result = 0
		}
	}

	return result
}

func (fm *FM) Delta(label float64, predicted float64) float64 {
	if *fm.Params.FM.Degree == 2 {
		// Classification
		return -label * Sigmoid(label*-predicted)
	}
	return -(label - predicted)
}

func (fm *FM) Fit(features []uint32, label float64, predicted float64, sum []float64) {
	delta := fm.Delta(label, predicted)
	fm.Optimizer.LearnW0(&fm.W0, delta)
	fm.Optimizer.LearnW(fm.Weights, features, delta, label)
	fm.Optimizer.LearnV(fm.InterWeights, features, delta, label, sum)
}
