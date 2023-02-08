from examples.stable_diffusion.model import build_stable_diffusion_model
import torch
import pytest


@pytest.mark.parametrize('model_name', ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"])
def test_model_builder(model_name, num_classes):
    model = build_stable_diffusion_model(model_name, num_classes)

    rand_input = torch.randn(1, 3, 64, 64)
    rand_label = torch.randint(0, num_classes - 1, (1,))
    output = model((rand_input, rand_label))
    assert output.shape == (1, num_classes)
    assert output.dtype == torch.float
