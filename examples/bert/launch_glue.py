import argparse
import copy

from mcli.sdk import RunConfig, create_run


def build_checkpoint_path(scheduler, init_rate, final_rate, seed, ckpt):
    return f"s3://mosaicml-internal-checkpoints-shared/zack/paper-mlm-schedule/schedule-{scheduler}-{init_rate}-{final_rate}-seed-{seed}/ep0-ba{ckpt}-rank0.pt"


def main(args):
    cfg_path = "yamls/glue/mlm_scheduled/base.yaml"
    config = RunConfig.from_file(cfg_path)

    if args.mode == "final":
        ckpts = [70_000]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    for ckpt in ckpts:
        for seed in [17, 2048, 3047]:
            run = copy.deepcopy(config)

            run.name = f"{args.scheduler}-{args.initial_mlm_rate}-to-{args.final_mlm_rate}-seed-{seed}-glue"
            run.name = run.name.replace(".", "")
            run.parameters[
                "base_run_name"] = f"schedule-{args.scheduler}-{args.initial_mlm_rate}-{args.final_mlm_rate}-seed-{seed}"

            run.parameters[
                "starting_checkpoint_load_path"] = build_checkpoint_path(
                    args.scheduler, args.initial_mlm_rate, args.final_mlm_rate,
                    seed, ckpt)

            run.parameters["loggers"]["wandb"]["tags"] = [
                "glue", args.scheduler, f"initial-{args.initial_mlm_rate}",
                f"final-{args.final_mlm_rate}", f"og-seed-{seed}"
            ]
            run.parameters["loggers"]["wandb"]["group"] = "-".join([
                args.scheduler, f"initial-{args.initial_mlm_rate}",
                f"final-{args.final_mlm_rate}", f"og-seed-{seed}"
            ])

            create_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--initial-mlm-rate", type=float, required=True)
    parser.add_argument("--final-mlm-rate", type=float, required=True)
    parser.add_argument("--mode", type=str, default="final")
    args = parser.parse_args()

    main(args)
