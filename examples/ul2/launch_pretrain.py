import argparse
import copy

from mcli import RunConfig, create_run


def main(args):
    cfg_path = "yamls/scheduled/base.yaml"
    config = RunConfig.from_file(cfg_path)

    for seed in args.seeds:
        run = copy.deepcopy(config)

        run.name = f"t5-{args.scheduler}-{args.initial_mlm_rate}-to-{args.final_mlm_rate}-seed-{seed}"
        run.name = run.name.replace(".", "")
        rn = f"t5-schedule-{args.scheduler}-{args.initial_mlm_rate}-{args.final_mlm_rate}"
        if args.model == "large":
            rn = "shortenedlarge-" + rn
        if args.subset_mask:
            rn = "subset-mask-" + rn
        #print(run.parameters)
        run.parameters["run_name"] = rn
        run.parameters["seed"] = seed
        run.parameters["train_loader"]["dataset"]["shuffle_seed"] = seed

        #run.parameters["train_loader"]["dataset"]["mlm_schedule"][
        #"name"] = args.scheduler
        #run.parameters["train_loader"]["dataset"]["mlm_schedule"][
        #"initial_mlm_rate"] = args.initial_mlm_rate
        #run.parameters["train_loader"]["dataset"]["mlm_schedule"][
        #"final_mlm_rate"] = args.final_mlm_rate
        #run.parameters["train_loader"]["dataset"]["mlm_schedule"][
        #"subset_masking_rate"] = 0.15 if args.subset_mask else None

        run.parameters["train_loader"]["scheduler_name"] = args.scheduler
        run.parameters["train_loader"]["mixture_of_denoisers"] = [
            3, args.initial_mlm_rate, args.final_mlm_rate
        ]

        run.parameters["loggers"]["wandb"]["tags"] = [
            "pretraining", args.scheduler, f"initial-{args.initial_mlm_rate}",
            f"final-{args.final_mlm_rate}", f"t5-{args.model}"
        ]
        run.parameters["loggers"]["wandb"]["group"] = "-".join([
            "pretraining", args.scheduler, f"initial-{args.initial_mlm_rate}",
            f"final-{args.final_mlm_rate}"
        ])

        if args.subset_mask:
            run.parameters["loggers"]["wandb"]["tags"].append("subset-mask")
            run.parameters["loggers"]["wandb"][
                "group"] = "subset-mask-" + run.parameters["loggers"]["wandb"][
                    "group"]

        run.scheduling = {"resumable": True, "priority": "low"}

        run.cluster = args.cluster
        if run.cluster not in ["r1z1", "r8z6"]:
            run.gpu_type = "a100_40gb"
            run.gpu_num = 16

        # Setting all ocurrences of specific model
        #model_name_fmt = f"bert-{args.model}-uncased"
        model_name_fmt = f"t5-{args.model}"
        run.parameters["tokenizer_name"] = model_name_fmt
        #run.parameters["model"]["pretrained_model_name"] = model_name_fmt
        #run.parameters["model"]["tokenizer_name"] = model_name_fmt
        #run.parameters["train_loader"]["dataset"][
        #"tokenizer_name"] = model_name_fmt
        #run.parameters["eval_loader"]["dataset"][
        #"tokenizer_name"] = model_name_fmt

        # Handling model config if bert large
        #if args.model == "large":
        #run.parameters
        #model_config = {
        #"attention_probs_dropout_prob": 0.1,
        #"hidden_size": 1024,
        #"intermediate_size": 4096,
        #"num_attention_heads": 16,
        #"num_hidden_layers": 24
        #}
        #run.parameters["model"]["model_config"] = model_config

        #run.parameters["optimizer"]["lr"] = 2.0e-4

        #run.parameters["max_duration"] = "286720000sp"

        create_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--initial-mlm-rate", type=float, required=True)
    parser.add_argument("--final-mlm-rate", type=float, required=True)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--cluster", default="r1z1")
    parser.add_argument("--seeds",
                        nargs="+",
                        type=int,
                        default=[17, 2048, 3047])
    parser.add_argument("--subset-mask", action="store_true")
    args = parser.parse_args()

    main(args)