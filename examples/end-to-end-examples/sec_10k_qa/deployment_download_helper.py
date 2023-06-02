# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from composer.utils import get_file


def download_model(remote_uri: str):
    get_file(
        path=remote_uri,
        destination='/downloaded_hf_checkpoint/',
    )
