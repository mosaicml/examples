set -e
set -o xtrace
source /mnt/workdisk/sasha/examples/venv-llm/bin/activate
source /secrets/secrets.env
for d_model in 256 512 1024 2048 
do
    for n_heads in 16
    do

    echo "d_model: ${d_model}"
    echo "n_heads: ${n_heads}"

    composer main.py \
        yamls/mosaic_gpt/small_mup_scaling.yaml \
            train_loader.dataset.split=train_small \
            max_duration=10ba \
            eval_interval=0 \
            data_remote=oci://mosaicml-internal-datasets/c4/base/pretok-gpt2-2k  \
            model.d_model=${d_model} \
            model.n_heads=${n_heads} \
            run_name=small_2_layers_mup_scaled_d_model_${d_model}_n_head_${n_heads} no_bias=True
done
done