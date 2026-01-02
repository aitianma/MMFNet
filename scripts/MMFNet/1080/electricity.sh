if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

#model_name=SparseTSF
model_name=MMFNet

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
alpha=0.5
lpf=1
seq_len=1080
#seq_len=1440

for lpf in   129
do
for alpha in  1
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
    --enc_in 321 \
    --patch_len 16 \
    --stride 8 \
    --patience 1 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 0 \
    --itr 1 --batch_size 64 --learning_rate 0.0001 --test_flop > logs/1080/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${lpf}_${alpha}.log  &

done
done
done


