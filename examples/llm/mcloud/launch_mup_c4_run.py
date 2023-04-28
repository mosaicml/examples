from mcli.sdk import RunConfig, create_run
import copy
import math

# or create directly in python with RunConfig(...)
config = RunConfig.from_file('mcli-base-train-for-mup-c4-d-model-256.yaml')
parameters = copy.deepcopy(config.parameters)

# Sweep over a few different values of 'foo'
runs = []
base_lr = 2.e-4
c4_data = "oci://mosaicml-internal-datasets/c4/base/pretok-gpt2-2k/"
s2_data = "oci://mosaicml-internal-datasets/s2/base/pretok-gpt2-2k/"
for embed_scale in [1.0, 10.0]:
    for friendly_data, dataset in zip(["c4", "s2"], [c4_data, s2_data]):
        for lr_scaler in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            # set the name of the run
            updated_lr = base_lr * (2**lr_scaler)
            print(updated_lr)
            print(dataset)

            rounded_num = round(updated_lr, 2 - int(math.floor(math.log10(abs(updated_lr)))) - 1)
            print(rounded_num)

            config.name = f'mpt-40m-warmup-d-256-lr-{updated_lr}-data-{friendly_data}-e-scale-{int(embed_scale)}'

            print(config.name)
            # Update the parameters
            # deepcopy for safety
            run_params = copy.deepcopy(parameters)
            run_params['optimizer']["lr"] = updated_lr
            run_params['data_remote'] = dataset
            run_params["model"]["mup"]["embed_scale"] = embed_scale
            config.parameters = run_params
            print(config.parameters)

            # And run!
            run = create_run(config)
            print(f'Launching run {run.name} with lr {updated_lr} and data {friendly_data} and embed scale {embed_scale}')
            runs.append(run)