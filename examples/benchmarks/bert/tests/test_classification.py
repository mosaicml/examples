# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from examples.bert.sequence_classification import main
from omegaconf import DictConfig, OmegaConf


def test_classification_script():
    with open('tests/smoketest_config_classification.yaml') as f:
        config = OmegaConf.load(f)
    assert isinstance(config, DictConfig)

    # The test is that `main` runs successfully
    main(config)
