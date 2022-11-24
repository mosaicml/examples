# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import unittest
from argparse import Namespace

from convert_c4 import main


class C4DownloadTest(unittest.TestCase):
    def test_download_script_from_api(self):
        # test calling it directly
        main(Namespace(**{'splits': ['val'], 'out_root': './my-copy-c4-1'}))
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'my-copy-c4-1')))
        shutil.rmtree(os.path.join(os.getcwd(), 'my-copy-c4-1'))

    def test_download_script_from_cmdline(self):
        # test calling it via the cmd line interface
        os.system("python convert_c4.py --out_root ./my-copy-c4-2 --splits val")
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'my-copy-c4-2')))
        shutil.rmtree(os.path.join(os.getcwd(), 'my-copy-c4-2'))
