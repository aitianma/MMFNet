if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MMFNet

root_path_name=./dataset/
data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
seq_len=1440

alpha=1
for alpha in 3 
do

  pred_len=96
  batch_size=64
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
      --period_len 4 \
      --enc_in 8 \
      --train_epochs 30 \
      --patience 5 \
      --alpha $alpha \
      --lpf 30\
      --gpu 0 \
      --itr 1 --batch_size $batch_size --learning_rate 0.01 > logs/seg/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_best.log &



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
      --period_len 4 \
      --enc_in 8 \
      --train_epochs 30 \
      --patience 5 \
      --alpha $alpha \
      --lpf 30 \
      --gpu 1 \
      --itr 1 --batch_size $batch_size --learning_rate 0.01 > logs/seg/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_best.log &



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
      --period_len 4 \
      --enc_in 8 \
      --train_epochs 30 \
      --patience 5 \
      --alpha $alpha \
      --lpf 30\
      --gpu 2 \
      --itr 1 --batch_size $batch_size --learning_rate 0.01 > logs/seg/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_best.log &

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
      --period_len 4 \
      --enc_in 8 \
      --train_epochs 30 \
      --patience 5 \
      --alpha $alpha \
      --lpf 30 \
      --gpu 3 \
      --itr 1 --batch_size $batch_size --learning_rate 0.01 > logs/seg/${model_name}_${model_id_name}_${pred_len}_${batch_size}_mft${alpha}_best.log &

done
