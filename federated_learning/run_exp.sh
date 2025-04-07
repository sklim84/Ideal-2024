seed=2024

method='fedsgd' # fedavg fedsgd local total
model_name='ctgan'  # ctgan tvae

num_samples_org=300
num_samples_syn=300

epoch=10
batch_size=50

results_path='./results/eval_results_ht.csv'

# fedsgd & ctgan 조합은 dim 16 이외 조합에서 오류

for method in fedavg fedsgd local total; do
  for model_name in ctgan tvae; do
    for emb_dim in 4 8 16 32; do
      for gen_dim in 4 8 16 32; do
        for dis_dim in 4 8 16 32; do
          python -u 2_make_syn_data_by_$method'_'$model_name.py \
            --method $method \
            --model_name $model_name \
            --seed $seed \
            --num_samples_org $num_samples_org \
            --num_samples_syn $num_samples_syn \
            --epoch $epoch \
            --batch_size $batch_size \
            --emb_dim $emb_dim \
            --gen_dim $gen_dim \
            --dis_dim $dis_dim \
            --results_path $results_path
        done
      done
    done
  done
done