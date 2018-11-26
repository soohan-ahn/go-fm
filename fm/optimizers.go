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
	gradient := 1.0
	*w0 = s.LearnRate * (delta*gradient + s.TrainR0*(*w0))
}

func (s LogisticFMSGD) LearnW(weights []float64, features []uint32, delta float64, label float64) {
	for _, feature := range features {
		gradient := label
		//delta = (0.0f - sigmoid(-mFuncValue * y) * y)
		weights[feature] = s.LearnRate * (delta*gradient + s.TrainRW*(weights[feature]))
	}
}

func (s LogisticFMSGD) LearnV(interWeights [][]float64, features []uint32, delta float64, label float64, sums []float64) {
	for _, feature := range features {
		for i, _ := range interWeights[feature] {
			gradient := sums[feature] - interWeights[feature][i]
			//delta = (0.0f - sigmoid(-mFuncValue * y) * y)
			interWeights[feature][i] = s.LearnRate * (delta*gradient + s.TrainRV*(interWeights[feature][i]))
		}
	}
}
