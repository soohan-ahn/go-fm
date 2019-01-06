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
	gradient := 0.0
	*w0 = *w0 - s.LearnRate*(delta*gradient+s.TrainR0*(*w0))
}

func (s LogisticFMSGD) LearnW(weights []float64, features []uint32, delta float64, label float64) {
	for _, f := range features {
		gradient := 1.0
		weights[f] = weights[f] - s.LearnRate*(delta*gradient+s.TrainRW*weights[f])
	}
}

func (s LogisticFMSGD) LearnV(interWeights [][]float64, features []uint32, delta float64, label float64, sums []float64) {
	for i, _ := range interWeights {
		for _, f := range features {
			gradient := sums[i] - interWeights[i][f] // sums[k] : Sum of the factor K
			interWeights[i][f] = interWeights[i][f] - s.LearnRate*(delta*gradient+s.TrainRV*interWeights[i][f])
		}
	}
}
