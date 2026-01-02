if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MMFNet

root_path_name=./dataset/
data_path_name=solar_AL.csv
model_id_name=Solar
data_name=custom
seq_len=720

alpha=7
pred_len=96
batch_size=64
itr=1
python3 -u run_longExp.py \
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
    --enc_in 137 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 19\
    --gpu 0 \
    --itr $itr --batch_size $batch_size --learning_rate 0.03 > logs/mft/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_itr${itr}_best.log &



pred_len=192
python3 -u run_longExp.py \
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
    --enc_in 137 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 15 \
    --gpu 3 \
    --itr $itr --batch_size $batch_size --learning_rate 0.03 > logs/mft/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_itr${itr}_best.log &



pred_len=336
python3 -u run_longExp.py \
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
    --enc_in 137 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 19\
    --gpu 1 \
    --itr $itr --batch_size $batch_size --learning_rate 0.03 > logs/mft/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_itr${itr}_best.log &

pred_len=720
python3 -u run_longExp.py \
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
    --enc_in 137 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 19 \
    --gpu 2 \
    --itr $itr --batch_size $batch_size --learning_rate 0.03 > logs/mft/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_itr${itr}_best.log &


