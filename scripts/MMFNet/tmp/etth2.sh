if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

#model_name=SparseTSF
model_name=MMFNet

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

#seq_len=1080
#seq_len=1440
#seq_len=336
#seq_len=720
#seq_len=720
#seq_len=512
seq_len=1080


for lpf in   129
do
for alpha in 3
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
    --period_len 24 \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 24 \
    --stride 24 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 3 \
    --itr 1 --batch_size 64 --learning_rate 0.001 > logs/tmp/${model_name}_${data_name}_${seq_len}_${pred_len}_${lpf}_${alpha}.log  &
done
done
done

