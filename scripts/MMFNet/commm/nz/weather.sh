if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MMFNet

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
alpha=0.5

seq_len=1440
for nz in  2 4 8 16 24 32
do
#for pred_len in   96 192 336  720

for pred_len in   720
#for pred_len in   960 1200 1440 1680
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
    --period_len 4 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 20 \
    --patch_len 2 \
    --stride 2 \
    --nz $nz \
    --gpu 2 \
    --itr 1 --batch_size 64 --learning_rate 0.0001  > logs/commm/nz/${model_name}_${model_id_name}_${seq_len}_${pred_len}_nz${nz}.log  
done

done