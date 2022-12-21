
from src.optim.AdaLR import AdaLR
from src.optim.AdaBound import AdaBound
from src.optim.EMASmoothStepAdam import EMASmoothStepAdam
from src.optim.SecondOrderAdaLR import SecondOrderAdaLR
from composer.optim import DecoupledAdamW

__all__ = [
   "AdaBound", "EMASmoothStepAdam", "OPEStepAdam", "DecoupledAdamW"
]

OPTIMIZERS = {
    "decoupled_adamw": DecoupledAdamW, 
    "adalr": AdaLR,
    "ema_step_adam": EMASmoothStepAdam,
    "adabound": AdaBound,
    "second_order_adalr": SecondOrderAdaLR
}