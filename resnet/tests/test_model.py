import pytest

from composer.models import ComposerClassifier

from ..model import build_composer_resnet

@pytest.mark.parametrize('model_name', ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
@pytest.mark.parametrize('loss_name', ['binary_cross_entropy', 'cross_entropy'])
@pytest.mark.parametrize('num_classes', [10, 100])
def test_model_builder(model_name, loss_name, num_classes):
    model = build_composer_resnet(model_name, loss_name, num_classes)
    assert isinstance(model, ComposerClassifier)
