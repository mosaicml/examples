
from src.optim.OPEStepAdam import OPEStepAdam
from src.optim.AdaBound import AdaBound
from src.optim.EMASmoothStepAdam import EMASmoothStepAdam
from src.optim.SecondOrderAdaLR import SecondOrderAdaLR
from composer.optim import DecoupledAdamW

__all__ = [
   "AdaBound", "EMASmoothStepAdam", "OPEStepAdam", "DecoupledAdamW"
]

OPTIMIZERS = {
    "decoupled_adamw": DecoupledAdamW, 
    "ope_step_adam": OPEStepAdam,
    "ema_step_adam": EMASmoothStepAdam,
    "adabound": AdaBound,
    "second_order_adalr": SecondOrderAdaLR
}