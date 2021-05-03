# SmoothI 

The model is implemented using PyTorch. Python environment requirements can be found at 'requirements.txt' file.

The src/ contains training or evaluation scripts.


The commands to prepare the pickle files, to train the model with different losses and finally evaluate the trained model are as follows:

#### To start, change current directory to src/
`cd src/`

### Prepare data (train/vali/test) pickle files 
`python prepare_data_pickles.py PATH_TO_DATA_FOLDER`

e.g. ('python prepare_data_pickles.py ../data/')

### Train model by maximizing P@10
`python train.py --model_dir  ../experiments/ --fold 1 -c smoothi_pk -k 10`

### Train model by maximizing NDCG@1
`python train.py --model_dir  ../experiments/ --fold 1 -c smoothi_ndcg -k 1`

### Train model by maximizing NDCG
`python train.py --model_dir  ../experiments/ --fold 1 -c smoothi_ndcg -k 0`

### Evaluate model
`python evaluate.py --model_dir ../experiments/ --fold 1`
