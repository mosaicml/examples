set -e
set -o xtrace
source /mnt/workdisk/sasha/examples/venv-llm/bin/activate
source /secrets/secrets.env
for d_model in 192 384 768
do
    for n_heads in 12
    do
    for optimizer in "mup_decoupled_lionw" "mup_decoupled_adamw"
    do

    echo "d_model: ${d_model}"
    echo "n_heads: ${n_heads}"
    echo "optimizer: ${optimizer}"

    composer main.py \
        yamls/mosaic_gpt/125m_mup.yaml \
            train_loader.dataset.split=train_small \
            max_duration=10ba \
            eval_interval=0 \
            data_remote=oci://mosaicml-internal-datasets/c4/base/pretok-gpt2-2k  \
            model.d_model=${d_model} \
            model.n_heads=${n_heads} \
            run_name=large_12_layers_mup_scaled_d_model_${d_model}_n_head_${n_heads}_optimizer_${optimizer} no_bias=True \
            optimizer.name=${optimizer}
done
done
done