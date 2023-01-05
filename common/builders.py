from composer import algorithms

def build_algorithm(name, kwargs):
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')