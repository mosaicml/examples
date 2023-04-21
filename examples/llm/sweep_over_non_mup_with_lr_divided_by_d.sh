set -e
source /mnt/workdisk/sasha/examples/venv-llm/bin/activate
source /secrets/secrets.env
d_models=(256 512 1024 2048)
lr_divisors=(1 2 4 8)

for i in ${!d_models[@]} 
do
    for n_heads in 16
    do

    d_model=${d_models[$i]}
    base_lr=0.6
    lr_divisor=${lr_divisors[$i]}
    new_lr=$(echo "scale=2; $base_lr/ $lr_divisor" | bc)

    echo "d_model: ${d_model}"
    echo "n_heads: ${n_heads}"
    echo "${new_lr}"

    composer main.py \
        yamls/mosaic_gpt/small_non_mup_scaling.yaml \
            train_loader.dataset.split=train_small \
            max_duration=10ba \
            eval_interval=0 \
            data_remote=oci://mosaicml-internal-datasets/c4/base/pretok-gpt2-2k  \
            model.d_model=${d_model} \
            model.n_heads=${n_heads} \
            optimizer.lr=${new_lr} \
            run_name=small_2_layers_no_mup_change_lr_scaled_d_model_${d_model}_n_head_${n_heads} no_bias=True
done
done