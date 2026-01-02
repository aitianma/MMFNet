if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

#model_name=SparseTSF
model_name=MMFNet

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
alpha=0.5
lpf=1

seq_len=1440

for nz in  2 4 8 16 24 32
do
#for pred_len in 96 192 336   720
for pred_len in  720

do
  ~/bin/python3 -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 862 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --patience 20 \
    --nz $nz \
    --gpu 1 \
    --itr 1 --batch_size 64 --learning_rate 0.03 > logs/commm/nz/${model_name}_${model_id_name}_${seq_len}_${pred_len}_nz${nz}.log  

done
done