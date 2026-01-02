if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MMFNet

root_path_name=./dataset/
data_path_name=solar_AL.csv
model_id_name=Solar
data_name=custom

seq_len=1080

for lpf in   129
#for lpf in  144
do
for alpha in  0.01
do
for pred_len in  96 192 336 720
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
    --enc_in 7 \
    --train_epochs 30 \
    --patch_len 3 \
    --stride 2 \
    --patience 3 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 3 \
    --itr 1 --batch_size 64 --learning_rate 0.0001 > logs/1080/${model_name}_${data_name}_${seq_len}_${pred_len}_${lpf}_${alpha}.log  &
done
done
done