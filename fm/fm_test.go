package fm_test

/*
import (
	"testing"

	. "github.com/soohanboys/go-fm/fm"
)

type Params struct {
	TrainAlpha          *float64
	TrainBeta           *float64
	TrainL1             *float64
	TrainL2             *float64
	Degree              *int
	MaxNonzeroDimension *int // Modify to featureSize
	MaxDimension        *int
	WeightFileName      *string
	InterWeightFileName *string
	PredictFileName     *string
	TrainDataFileName   *string
	TrainR0             *float64
	TrainRW             *float64
	TrainRV             *float64
}

func initModel() (*FM, Params) {
	p := Params{}
	degree := 2
	maxnzd := 2
	maxd := 2
	p.Degree = &degree
	p.MaxNonzeroDimension = &maxnzd
	p.MaxDimension = &maxd

	coefR0 := 0.0
	coefW := 0.0
	coefInterW := 0.0
	p.TrainR0 = &coefR0
	p.TrainRW = &coefW
	p.TrainRV = &coefInterW

	model := &FM{}
	Init(model, p)
	model.InterWeights[0][0] = 6
	model.InterWeights[0][1] = 0
	model.InterWeights[1][0] = 5
	model.InterWeights[1][1] = 8
	model.Weights[0] = 9
	model.Weights[1] = 2
	model.W0 = 2.0

	return model, p
}

func TestPredict(t *testing.T) {
	model, p := initModel()
	features := [][]uint32{
		{6, 1},
		{2, 3},
		{3, 0},
		{6, 1},
		{4, 5},
	}
	expected := []float64{298, 266, 29, 298, 848}

	for i, feature := range features {
		sum, squareSum := model.CalcSums(feature, p)
		if predicted := model.Predict(feature, sum, squareSum, p); predicted != expected[i] {
			t.Fatalf("Unexpedted predicted value: %v, expected: %v\n", predicted, expected[i])
		}
	}
}

func TestClassification(t *testing.T) {
	model, p := initModel()
	features := [][]uint32{
		{6, 1},
		{2, 3},
		{3, 0},
		{6, 1},
		{4, 5},
	}
	labels := []float64{298, 266, 29, 298, 848}

	for i, f := range features {
		sum, squareSum := model.CalcSums(f, p)
		predicted := model.Predict(f, sum, squareSum, p)

		label := labels[i]
		delta := -label * (1.0 - Sigmoid(label*predicted))
		t.Logf("l: %v\n", predicted)
		model.Train(f, label, delta, sum)
	}

	predictedLabels := []float64{}
	for _, f := range features {
		sum, squareSum := model.CalcSums(f, p)
		predicted := model.Predict(f, sum, squareSum, p)
		t.Logf("r: %v\n", predicted)
		predictedLabels = append(predictedLabels, predicted)
	}

	t.Logf("l: %v\n", predictedLabels)
}
*/
