# An Efficient Diffusion-based Non-Autoregressive Solver for Traveling Salesman Problem
Official implementation of SIGKDD 2025 paper: "An Efficient Diffusion-based Non-Autoregressive Solver for Traveling Salesman Problem".

## Dependencies
+ Cython==0.29.33
+ numpy==1.21.6
+ Python==3.7.16
+ pytorch_lightning==1.7.7
+ scipy==1.7.3
+ torch==1.12.1
+ wandb==0.13.11

Setup
```
conda create -n python=3.7
pip install -r requirements.txt
```
Running TSP experiments requires installing the additional cython package for merging the diffusion heatmap results:
```
cd utils/cython_merge
python setup.py build_ext --inplace
cd -
```
## Data
### TSP-20, TSP-50, TSP-100
Both the training and evaluation data of TSP-20, TSP-50 and TSP-100 data are taken from [chaitjo/learning-tsp](https://github.com/chaitjo/learning-tsp).
### TSP-200, TSP-500, TSP-1000
The evaluation data of TTSP-200, SP-500 and TSP-1000 are taken from [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP).
## Pretrained Checkpoints
Please download the pretrained model checkpoints from [here](https://drive.google.com/drive/folders/1YqensFpeH1Sep0yXS6T-jkL5RPPgc_1s?usp=sharing).


## Training and Evaluation
Due to the various possible configurations, we only provide a few examples of commands.

For example, training DEITSP on TSP20 data:
```
python -u train.py \
  --wandb_logger_name "train20" \
  --do_train \
  --learning_rate 0.0002 \
  --training_split "datasets/tsp20_train_concorde.txt" \
  --validation_split "datasets/tsp20_test_concorde.txt" \
  --test_split "datasets/tsp20_test_concorde.txt" \
  --num_workers 32 \
  --batch_size 128 \
  --num_epochs 50 \
  --validation_examples 8 \
  --val_step 200 \
  --inference_diffusion_steps 16

```
For instance, a simple run of DEITSP on TSP20 data:
```
python -u train.py \
  --wandb_logger_name "test20" \
  --do_test \
  --training_split "datasets/tsp20_train_concorde.txt" \
  --validation_split "datasets/tsp20_test_concorde.txt" \
  --test_split "datasets/tsp20_test_concorde.txt" \
  --num_workers 32 \
  --validation_examples 8 \
  --inference_schedule "pow" \
  --inference_diffusion_steps 16 \
  --ckpt_path "checkpoints/tsp20.ckpt" \
  --resume_weight_only

```