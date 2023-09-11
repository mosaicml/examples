import argparse
import copy

from mcli import RunConfig, create_run


def build_checkpoint_path(scheduler, init_rate, final_rate, seed, ckpt, model):
    return f"s3://mosaicml-internal-checkpoints-shared/zack/rebuttal/rts-{'large-' if model == 'large' else ''}schedule-{scheduler}-{init_rate}-{final_rate}-seed-{seed}/ep0-ba{ckpt}-rank0.pt"


def main(args):
    cfg_path = "yamls/glue/mlm_scheduled/base.yaml"
    config = RunConfig.from_file(cfg_path)

    if args.mode == "final":
        ckpts = [70_000]
    elif args.mode == "speed-up":
        # ckpts = [*range(0, 69_999, 10_000)][1:]
        ckpts = [50_000, 60_000]
    elif args.ckpt is not None:
        ckpts = [args.ckpt]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    for ckpt in ckpts:
        #for seed in [17, 2048, 3047]:
        for seed in [17, 2048]:
            run = copy.deepcopy(config)

            run.name = f"rts-{args.scheduler}-{args.initial_mlm_rate}-to-{args.final_mlm_rate}-seed-{seed}-glue"
            run.name = run.name.replace(".", "")
            rn = f"rts-schedule-{args.scheduler}-{args.initial_mlm_rate}-{args.final_mlm_rate}-seed-{seed}-ckpt-{ckpt}"
            if args.model == "large":
                rn = "large-" + rn
            run.parameters["base_run_name"] = rn

            # if args.mode == "speed-up" or args.priority == "low":
            run.scheduling = {"priority": "high"}

            run.parameters[
                "starting_checkpoint_load_path"] = build_checkpoint_path(
                    args.scheduler, args.initial_mlm_rate, args.final_mlm_rate,
                    seed, ckpt, args.model)

            run.parameters["loggers"]["wandb"]["project"] = "rebuttal-mlm-schedule"
            run.parameters["loggers"]["wandb"]["tags"] = [
                "glue", args.scheduler, f"initial-{args.initial_mlm_rate}",
                f"final-{args.final_mlm_rate}", f"og-seed-{seed}",
                f"bert-{args.model}"
            ]
            if args.mode == "final" or args.mode == "ckpt":
                run.parameters["loggers"]["wandb"]["tags"].append("best-ckpt")
            elif args.mode == "speed-up":
                run.parameters["loggers"]["wandb"]["tags"].append("speed-up")

            run.parameters["loggers"]["wandb"]["group"] = "-".join([
                args.scheduler, f"initial-{args.initial_mlm_rate}",
                f"final-{args.final_mlm_rate}", f"og-seed-{seed}", f"ckpt-{ckpt}"
            ])

            # Setting all ocurrences of specific model
            model_name_fmt = f"bert-{args.model}-uncased"
            run.parameters["tokenizer_name"] = model_name_fmt
            run.parameters["model"]["pretrained_model_name"] = model_name_fmt
            run.parameters["model"]["tokenizer_name"] = model_name_fmt

            # Handling model config if bert large
            if args.model == "large":
                model_config = {
                    "attention_probs_dropout_prob": 0.1,
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_attention_heads": 16,
                    "num_hidden_layers": 24
                }
                run.parameters["model"]["model_config"] = model_config

            run.cluster = args.cluster
            if run.cluster not in ["r1z1", "r8z6"]:
                run.gpu_type = "a100_40gb"

            created_run = create_run(run)
            print(f"Created run:", created_run.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--initial-mlm-rate", type=float, required=True)
    parser.add_argument("--final-mlm-rate", type=float, required=True)
    parser.add_argument("--cluster", type=str, default="r1z1")
    parser.add_argument("--mode", type=str, default="final")
    parser.add_argument("--priority", type=str, default="medium")
    parser.add_argument("--ckpt", type=int, default=None)
    args = parser.parse_args()

    main(args)
