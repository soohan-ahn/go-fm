package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/soohanboys/go-fm/fm"
)

func train(p fm.Params) {
	var model fm.AbstractFM
	model = &fm.FM{}
	/*
		TODO:
		if d := *(p.Degree); d == 2 {
			model = &fm.FM{}
		} else {
			model = fm.MFM{}
		}
	*/
	fm.Init(model, p)
	fm.Train(model, p)
}

func predict(p fm.Params) {
	var model fm.AbstractFM
	model = &fm.FM{}
	/*
		TODO:
		if d := *(p.Degree); d == 2 {
			model = &fm.FM{}
		} else {
			model = fm.MFM{}
		}
	*/
	fm.Init(model, p)
	fm.Predict(model, p)
}

func invalidArgs() {
	fmt.Printf("Invalid args!\n")
}

func main() {
	if len(os.Args) < 2 {
		invalidArgs()
		return
	}

	//getBalanceCmd := flag.NewFlagSet("getbalance", flag.ExitOnError)
	trainCmd := flag.NewFlagSet("train", flag.ExitOnError)
	trainInputParams := fm.Params{
		TrainAlpha:          trainCmd.Float64("alpha", 0.0, "alpha value for the training"),
		TrainBeta:           trainCmd.Float64("beta", 0.0, "beta value for the training"),
		TrainL1:             trainCmd.Float64("l1", 0.0, "l1 value for the training"),
		TrainL2:             trainCmd.Float64("l2", 0.0, "l2 value for the training"),
		Degree:              trainCmd.Int("d", 2, "degree for the training"),
		MaxNonzeroDimension: trainCmd.Int("maxnzd", 30, "Max dimension for the training."),
		MaxDimension:        trainCmd.Int("maxd", 30, "Max dimension for the training. (Hivemall=16777216)"),
		TrainR0:             trainCmd.Float64("reg0", 0.0, "reg value for SGD."),
		TrainRW:             trainCmd.Float64("regw", 0.0, "reg value for SGD."),
		TrainRV:             trainCmd.Float64("regv", 0.0, "reg value for SGD."),
		TrainDataFileName:   trainCmd.String("train_data_file_name", "", "Filename of the training data"),
		WeightFileName:      trainCmd.String("weight_file_name", "", "Filename of the weights."),
		InterWeightFileName: trainCmd.String("inter_weight_file_name", "", "Filename of the weights."),
	}

	predictCmd := flag.NewFlagSet("predict", flag.ExitOnError)
	predictInputParams := fm.Params{
		WeightFileName:      predictCmd.String("weight_file_name", "", "Filename of the weights."),
		InterWeightFileName: predictCmd.String("inter_weight_file_name", "", "Filename of the weights."),
		PredictFileName:     predictCmd.String("predict_file_name", "", "Filename of the features to predict."),
		Degree:              predictCmd.Int("d", 2, "degree for the prediction"),
		MaxNonzeroDimension: predictCmd.Int("maxnzd", 30, "Max dimension for the predicting."),
		MaxDimension:        predictCmd.Int("maxd", 30, "Max dimension for the predicting. (Hivemall=16777216)"),
		TrainR0:             predictCmd.Float64("reg0", 0.0, "reg value for SGD."),
		TrainRW:             predictCmd.Float64("regw", 0.0, "reg value for SGD."),
		TrainRV:             predictCmd.Float64("regv", 0.0, "reg value for SGD."),
	}

	switch os.Args[1] {
	case "train":
		err := trainCmd.Parse(os.Args[2:])
		if err != nil {
			fmt.Printf("Err on opening CLI: %v", err)
		}
	case "predict":
		err := predictCmd.Parse(os.Args[2:])
		if err != nil {
			fmt.Printf("Err on opening CLI: %v", err)
		}
	default:
		invalidArgs()
		return
	}

	if trainCmd.Parsed() {
		train(trainInputParams)
		return
	}
	if predictCmd.Parsed() {
		predict(predictInputParams)
		return
	}

	fmt.Printf("Argument parsing failed.")
}
