from src.tokenizer import TOKENIZER_REGISTRY



from composer.core import Evaluator
from composer.datasets.in_context_learning_evaluation import get_icl_task_dataloader

from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from omegaconf import OmegaConf as om
import sys
from icl_eval.model_loading import load_model
import time


def validate_cfg(eval_cfg):
    assert "dataset_uri" in eval_cfg
    assert "type" in eval_cfg
    assert "num_fewshot" in eval_cfg
    assert "batch_size" in eval_cfg
    assert "metrics" in eval_cfg
    assert "formatting_options" in eval_cfg
    assert "prompt_string" in eval_cfg.get("formatting_options")
    assert "example_delimiter" in eval_cfg.get("formatting_options")
    assert "continuation_delimiter" in eval_cfg.get("formatting_options")
    assert 'label' in eval_cfg

def build_evaluators(cfg):
    
    tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.type](**cfg.tokenizer.args)
    evaluators = []
    logger_keys = []
    for eval_cfg in cfg.icl_tasks:
        validate_cfg(eval_cfg)
        dataset_uri = eval_cfg.get("dataset_uri")
        type = eval_cfg.get("type")
        num_fewshots = eval_cfg.get("num_fewshot")
        batch_size = eval_cfg.get("batch_size")
        metrics = list(eval_cfg.get("metrics"))
        prompt_string = eval_cfg.get("formatting_options").get("prompt_string")
        example_delimiter = eval_cfg.get("formatting_options").get("example_delimiter")
        continuation_delimiter = eval_cfg.get("formatting_options").get("continuation_delimiter")

        for num_fewshot in num_fewshots:
            label = f"{eval_cfg.get('label')}_{num_fewshot}-shot"
            dl = get_icl_task_dataloader(
                type,
                dataset_uri,
                tokenizer,
                batch_size=batch_size,
                max_seq_len=cfg.tokenizer.args.max_seq_len,
                eos_tok_id=tokenizer.pad_token_id,
                num_fewshot=num_fewshot,
                prompt_string=prompt_string,
                example_delimiter=example_delimiter,
                continuation_delimiter=continuation_delimiter
            )
            logger_keys.extend([f"metrics/{label}/{metric}" for metric in metrics])
            evaluators.append(Evaluator(label=label, dataloader=dl, metric_names=metrics))

    return evaluators, logger_keys


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)

    model = load_model(**cfg.get("model"))
    evaluators, logger_keys = build_evaluators(cfg)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
   
    trainer = Trainer(model=model, max_duration='1ba', loggers=in_memory_logger)
    for evaluator in evaluators:
        model.add_eval_metrics(evaluator)
    
    a = time.time()
    trainer.eval(eval_dataloader=evaluators)
    b = time.time()
    print(f"Ran eval in: {b-a} seconds")

    for key in logger_keys:
        if key in in_memory_logger.data:
            result = in_memory_logger.data[key][0][1].item()
            print(f"{key}: {result}")