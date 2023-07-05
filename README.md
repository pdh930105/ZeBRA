# ZeBRA

## 2021 BMVC Accpeted Paper code

### Intro

Future updated

## Dockerfile build

docker build . --tag zebra:latest

## Usage

### default Usage
```bash
sh ZeBRA_CIFAR_resnet20_8bit_get_log.sh
```

### this code support 3 attack algorithm (bfa, zebra, datafree_zebra)

```
# only one choice
--bfa # attack using real data
--zebra # attack using synthetic data, but evaluate real data
--datafree_zebra # attack using synthetic data, evaluate also synthetic data
```

### important argument explained 
(If you want to detail explanation, see ZeBRA_CIFAR_resnet20_8bit_get_log.sh & main.py)

#### attack batch size, maximum # of bit flips, compared candidate bit in layer
```
attack_sample_size=64 # attack batch-size
n_iter=20 # maximum bit flip count
k_top=10 # detect gradient ranking
quan_bitwidth=8 # model's weight bit

``` 

#### ZeBRA setting (attack using generate data, but evaluation  real valid data)

```
n_search=5 # create #-of mini-batch distilled target data 
achieve_bit=9 # ZeBRA success bit flip count
drop_acc=10 # ZeBRA success drop acc (cifar10 : 10, imagenet : 0.1)
total_loss_bound=10 # distilled total loss bound (default : 10)
CE_loss_lambda=0.2 # Cross-Entropy scale parameter (default : 0.1)
distilled_loss_lambda=0.1 # distilled_loss scale parameter (default : 0.1)
distilled_batch_size=16 # made distilled_dataset batch-size
```

#### datafree_ZeBRA setting (attack using generate data and evaluation generate data)

```
# generate valid synthetic data

distilled_target_valid_batch_size=64 
distilled_target_valid_CE_loss_lambda=2
distilled_target_valid_distill_loss_lambda=1
distilled_target_valid_total_loss_bound=1

# heuristic check accuracy because synthetic data is less accurate faster.
distilled_target_drop_acc1=11.0
distilled_target_drop_acc5=52.0
```
