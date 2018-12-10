package fm

import (
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"os"
)

type Config struct {
	SGD   SGDParams   `yaml:"sgd,omitempty"`
	FM    *FMParams   `yaml:"fm,omitempty"`
	Files *FileParams `yaml:"files,omitempty"`
}

type SGDParams struct {
	TrainR0 float64 `yaml:"r0,omitempty"`
	TrainRW float64 `yaml:"rw,omitempty"`
	TrainRV float64 `yaml:"rv,omitempty"`
}

type FMParams struct {
	Degree       *int `yaml:"degree,omitempty"`
	KFactor      *int `yaml:"kfactor,omitempty"`
	MaxDimension *int `yaml:"max_dimension,omitempty"`
	Epoch        int  `yaml:"epoch,omitempty"`
}

type FileParams struct {
	WeightFileName      *string `yaml:"weight_file_name,omitempty"`
	InterWeightFileName *string `yaml:"inter_file_name,omitempty"`
	PredictFileName     *string `yaml:"predict_file_name,omitempty"`
	TrainDataFileName   *string `yaml:"train_data_file_name,omitempty"`
}

const configPath = "./fm/data/config.yaml"

func ReadParams() (*Config, error) {
	file, err := os.Open(configPath)
	if err != nil {
		return nil, err
	}
	content, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var config *Config
	err = yaml.Unmarshal(content, &config)
	if err != nil {
		return nil, err
	}

	return config, nil
}
