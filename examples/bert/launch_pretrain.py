import argparse
import copy

from mcli.sdk import RunConfig, create_run


def main(args):
    cfg_path = "yamls/main/mlm_scheduled/base.yaml"
    config = RunConfig.from_file(cfg_path)

    for seed in [17, 2048, 3047]:
        run = copy.deepcopy(config)

        run.name = f"{args.scheduler}-{args.initial_mlm_rate}-to-{args.final_mlm_rate}-seed-{seed}"
        run.name = run.name.replace(".", "")
        run.parameters[
            "run_name"] = f"schedule-{args.scheduler}-{args.initial_mlm_rate}-{args.final_mlm_rate}"

        run.parameters["seed"] = seed

        run.parameters["train_loader"]["dataset"]["mlm_schedule"][
            "name"] = args.scheduler
        run.parameters["train_loader"]["dataset"]["mlm_schedule"][
            "initial_mlm_rate"] = args.initial_mlm_rate
        run.parameters["train_loader"]["dataset"]["mlm_schedule"][
            "final_mlm_rate"] = args.final_mlm_rate

        run.parameters["loggers"]["wandb"]["tags"] = [
            "pretraining", args.scheduler, f"initial-{args.initial_mlm_rate}",
            f"final-{args.final_mlm_rate}", f"bert-{args.model}"
        ]

        run.parameters["loggers"]["wandb"]["groups"] = [
            "pretraining", args.scheduler, f"initial-{args.initial_mlm_rate}",
            f"final-{args.final_mlm_rate}"
        ]

        create_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--initial-mlm-rate", type=float, required=True)
    parser.add_argument("--final-mlm-rate", type=float, required=True)
    args = parser.parse_args()

    main(args)
