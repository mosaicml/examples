from examples.common.resumption_callbacks.callbacks import LayerFreezing, GlobalLRScaling

RESUMPTION_STRATEGIES = {
    "layer_freezing": LayerFreezing,
    'global_lr_scaling': GlobalLRScaling,
}