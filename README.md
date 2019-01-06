## go-fm
Facorization Machine implemented in Go.
Reference paper: [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

### train
./go-fm train

### predict
./go-fm predict

### config.yaml
* Set all the parameters needed for the train/predict on the fm/data/config.yaml
* Parameters
    * sgd: Parameters needed for the SGD training step.
    * fm: Set the numbers needed for the FM.
        * Currently, only the logic is implemeted for degree=2.
    * files: Set the paths of the predict/train/weight files.
