from mcli.sdk import RunConfig, create_run
import copy

# or create directly in python with RunConfig(...)
config = RunConfig.from_file('mcli-125m-base-train-for-sp.yaml')
parameters = copy.deepcopy(config.parameters)

# Sweep over a few different values of 'foo'
runs = []
base_width = 192
for width_mult in [1, 2, 4, 8]:
    # set the name of the run
    updated_width = base_width * width_mult

    config.name = f'mgpt-4-layer-d-model-{updated_width}-SP-10k-steps-alibi-clip-lion'

    # Update the parameters
    # deepcopy for safety
    run_params = copy.deepcopy(parameters)
    run_params['model']["d_model"] = updated_width
    config.parameters = run_params
    print(config.parameters)

    # And run!
    run = create_run(config)
    print(f'Launching run {run.name} with d_model {updated_width}')
    runs.append(run)