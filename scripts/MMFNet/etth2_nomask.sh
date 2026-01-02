if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

#model_name=SparseTSF
model_name=MMFNet

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

seq_len=1080
#seq_len=1440
#seq_len=336
#seq_len=720
#seq_len=720
#seq_len=512


for pred_len in   336

#for pred_len in   96 192 336 720
#for pred_len in   960 1200 1440 1680
do
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
    --period_len 16 \
    --enc_in 7 \
    --train_epochs 10 \
    --patience 5 \
    --gpu 1 \
    --itr 1 --batch_size 64 --learning_rate 0.001 > logs/${model_name}_${data_name}_${seq_len}_${pred_len}_nomask.log  &
done

for pred_len in   96 192

#for pred_len in   96 192 336 720
#for pred_len in   960 1200 1440 1680
do
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
    --period_len 16 \
    --enc_in 7 \
    --train_epochs 10 \
    --patience 5 \
    --gpu 1 \
    --itr 1 --batch_size 64 --learning_rate 0.01 > logs/${model_name}_${data_name}_${seq_len}_${pred_len}_nomask.log  &
done



seq_len=510


for pred_len in   720
#for pred_len in   96 192 336 720
#for pred_len in   960 1200 1440 1680
do
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
    --period_len 16 \
    --enc_in 7 \
    --train_epochs 10 \
    --patience 5 \
    --gpu 1 \
    --itr 1 --batch_size 64 --learning_rate 0.01 > logs/${model_name}_${data_name}_${seq_len}_${pred_len}_nomask.log  &

done
