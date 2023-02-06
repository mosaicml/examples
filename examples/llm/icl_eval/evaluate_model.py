
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from omegaconf import OmegaConf as om
import sys
from icl_eval.model_loading import load_model
import time
import torch
from examples.common.builders import build_evaluators



if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)

    model_dict = load_model(**cfg.get("model"))
    evaluators, logger_keys = build_evaluators(cfg)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    
    trainer = Trainer(
        model=model_dict.get('model'),
        loggers=in_memory_logger,
        fsdp_config=model_dict.get('fsdp_config', None),
        load_path=model_dict.get('checkpoint', None),
        load_weights_only=True,
        log_to_console=True)
    trainer.state.evaluators = evaluators
    for evaluator in evaluators:
        model_dict['model'].add_eval_metrics(evaluator)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(eval_dataloader=evaluators)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()

    print(f"Ran eval in: {b-a} seconds")

    for key in logger_keys:
        if key in in_memory_logger.data:
            result = in_memory_logger.data[key][0][1].item()
            print(f"{key}: {result}")