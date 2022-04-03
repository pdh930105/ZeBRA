################# BFA setting ########################

HOST=$(hostname)
echo "Current host is : $HOST"
DATE=`date +%Y-%m-%d`
echo "running start $DATE"

if [ ! -d "$DIRECTORY"]; then
    mkdir ./save/${DATE}/
fi

enable_tb_display=false # enable tensorboard display
model=resnet20_quan
dataset=cifar10
test_batch_size=128

data_path=./dataset/


manualSeed=0 # torch manual seed
label_info=ZeBRA

attack_sample_size=64 # attack batch-size
n_iter=20 # maximum bit flip count
k_top=10 # detect gradient ranking
quan_bitwidth=8
tb_path=${save_path}/tb_log
pretrained_model=./pretrained/resnet20_8bit_cifar10_92_41.pth.tar
########### ZeBRA Setting ################################

n_search=5 # create #-of mini-batch distilled target data 
achieve_bit=9 # ZeBRA success bit flip count
drop_acc=10 # ZeBRA success drop acc (cifar10 : 10, imagenet : 0.1)
total_loss_bound=10 # distilled total loss bound (default : 10)
CE_loss_lambda=0.2 # Cross-Entropy scale parameter (default : 0.1)
distilled_loss_lambda=0.1 # distilled_loss scale parameter (default : 0.1)
distilled_batch_size=16 # made distileld_dataset batch-size


############# Heuristic ZeBRA Setting ####################

distilled_target_valid_batch_size=64
distilled_target_valid_CE_loss_lambda=2
distilled_target_valid_distill_loss_lambda=1
distilled_target_valid_total_loss_bound=1
distilled_target_drop_acc1=11.0
distilled_target_drop_acc5=52.0

################## neural network attack ##################################


save_path=./save/${DATE}/${dataset}_${model}_${label_info}_${quan_bitwidth}bit_${attack_sample_size}_${CE_loss_lambda}_${distilled_loss_lambda}_${total_loss_bound}

for seed in `seq 0 1 5`
do
  /usr/bin/env python ./main.py --dataset ${dataset} \
    --data_path ${data_path} \
    --arch ${model} --save_path ${save_path} \
    --test_batch_size ${test_batch_size} --workers 8 --ngpu 1 --gpu_id 0 \
    --print_freq 50 \
    --manualSeed ${seed} \
    --evaluate --resume ${pretrained_model} --fine_tune \
    --reset_weight --zebra \
    --distilled_batch_size ${distilled_batch_size} \
    --attack_sample_size ${attack_sample_size} \
    --n_iter ${n_iter} \
    --n_search ${n_search} \
    --achieve_bit ${achieve_bit} \
    --drop_acc ${drop_acc} \
    --total_loss_bound ${total_loss_bound} \
    --CE_loss_lambda ${CE_loss_lambda} \
    --distilled_loss_lambda ${distilled_loss_lambda} \
    --distilled_target_valid_batch_size ${distilled_target_valid_batch_size} \
    --distilled_target_valid_CE_loss_lambda ${distilled_target_valid_CE_loss_lambda} \
    --distilled_target_valid_total_loss_bound ${distilled_target_valid_total_loss_bound} \
    --distilled_target_drop_acc1 ${distilled_target_drop_acc1} \
    --distilled_target_drop_acc5 ${distilled_target_drop_acc5} \
    --quan_bitwidth ${quan_bitwidth}

done

