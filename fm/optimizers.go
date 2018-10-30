package fm

type AbstractFMOptimizer interface {
	LearnW0(w0 *float64, delta float64)
	LearnW(weights []float64, features []uint32, delta float64, label float64)
	LearnV(interWeights [][]float64, features []uint32, delta float64, label float64, sums []float64)
}

type LogisticFMSGD struct {
	LearnRate float64
	TrainR0   float64
	TrainRW   float64
	TrainRV   float64
}

func (s LogisticFMSGD) LearnW0(w0 *float64, delta float64) {
	*w0 = s.LearnRate * (delta + s.TrainR0*(*w0))
}

func (s LogisticFMSGD) LearnW(weights []float64, features []uint32, delta float64, label float64) {
	for _, feature := range features {
		weights[feature] = s.LearnRate * (delta*label + s.TrainRW*(weights[feature]))
	}
}

func (s LogisticFMSGD) LearnV(interWeights [][]float64, features []uint32, delta float64, label float64, sums []float64) {
	for _, feature := range features {
		for i, _ := range interWeights[feature] {
			grad := sums[feature]*label - interWeights[feature][i]*label*label
			interWeights[feature][i] = s.LearnRate * (delta*grad + s.TrainRV*(interWeights[feature][i]))
		}
	}
}
