package main

import (
	"fmt"
	"os"

	"github.com/soohanboys/go-fm/fm"
)

func train() {
	model := &fm.FM{}
	model.Init()
	fm.Train(model)
}

func predict() {
	model := &fm.FM{}
	model.Init()
	fm.PredictAll(model)
}

func invalidArgs() {
	fmt.Printf("Invalid args!\n")
}

func main() {
	if len(os.Args) < 2 {
		invalidArgs()
		return
	}

	switch os.Args[1] {
	case "train":
		train()
		return
	case "predict":
		predict()
		return
	default:
		invalidArgs()
		return
	}

	fmt.Printf("Argument parsing failed.")
}
