from examples.stable_diffusion.model import build_stable_diffusion_model
import torch
import pytest


@pytest.mark.parametrize('model_name', ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"])
def test_model_builder(model_name):
    model = build_stable_diffusion_model(model_name)
    batch_size = 1
    image = torch.randn(batch_size, 3, 64, 64)
    caption = torch.randint(low=0, high=128, size=(batch_size, 77,), dtype=torch.long)

    batch = {'image_tensor': image, 'input_ids': caption}
    output = model(batch)
    print(output)
