
from examples.common.optim.AdaLR import AdaLR
from examples.common.optim.AdaBound import AdaBound
from examples.common.optim.EMASmoothStepAdam import EMASmoothStepAdam
from examples.common.optim.SecondOrderAdaLR import SecondOrderAdaLR
from composer.optim import DecoupledAdamW

__all__ = [
   "AdaBound", "AdaLR", "EMASmoothStepAdam", "OPEStepAdam", "DecoupledAdamW"
]

OPTIMIZERS = {
    "decoupled_adamw": DecoupledAdamW, 
    "adalr": AdaLR,
    "ema_step_adam": EMASmoothStepAdam,
    "adabound": AdaBound,
    "second_order_adalr": SecondOrderAdaLR
}