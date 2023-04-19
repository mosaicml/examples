# TODO (sasha): Fix hacky name based filtering, maybe filter on module type?

def _is_input_type_layer(name):
    return "wte" in name or "wpe" in name
def _is_layernorm_type(name):
    return "ln_1" in name or "ln_2" in name or "ln_f" in name
def _is_ffn_down(name):
    return "mlp_down" in name

def _filter_named_parameters(named_parameters):
    """
    returns groups that will be used for param_groups
    
    unscaled param group: LR scale: 1.0
    ffn_down: LR scale: / d_ffn_ratio
    rest_of_network: LR scale: / d_model_ratio

    """
    unscaled_param_group = {}
    ffn_down_group = {}
    rest_of_network_group = {}

    for name, param in named_parameters:
        print(f"{name} {param.shape}")

        if _is_input_type_layer(name) or _is_layernorm_type(name):
            print(f"unscaled param type: {name}")
            unscaled_param_group[name] = param
        elif _is_ffn_down(name):
            print(f"ffn down:{name}")
            ffn_down_group[name] = param
        else:
            rest_of_network_group[name] = param

    return unscaled_param_group, ffn_down_group, rest_of_network_group

        
def mup_setup_lrs_for_params(named_parameters, mup_config, base_lr):
    filtered_groups = _filter_named_parameters(named_parameters)

    param_groups = [] 

    for group, scale_factor in zip(filtered_groups, [1.0, mup_config.ffn_scale_ratio, mup_config.d_model_scale_ratio ]):
        param_groups.append({"params" : list(group.values()), "lr": base_lr / float(scale_factor)})

    print(param_groups)
    
    return param_groups
   
    
