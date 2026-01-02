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

for lpf in   129
#for lpf in   144
do
for alpha in  0.01
do
for pred_len in    720
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
    --patience 20 \
    --patch_len 2 \
    --stride 2 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 0 \
    --itr 1 --batch_size 64 --learning_rate 0.01 > logs/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${lpf}_${alpha}.log  &
done
done
done
