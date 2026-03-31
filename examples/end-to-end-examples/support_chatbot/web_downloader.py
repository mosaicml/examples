# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import html
import os
import re
import urllib.request

from bs4 import BeautifulSoup

all_links = [
    'https://docs.mosaicml.com',
    'https://docs.mosaicml.com/projects/composer/',
    'https://docs.mosaicml.com/projects/composer/en/stable/getting_started/installation.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/getting_started/quick_start.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/getting_started/welcome_tour.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/getting_started.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/functional_api.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/medical_image_segmentation.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/custom_speedup_methods.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/ffcv_dataloaders.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/finetune_huggingface.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/pretrain_finetune_huggingface.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/migrate_from_ptl.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/early_stopping.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/auto_microbatching.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/checkpoint_autoresume.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/exporting_for_inference.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/TPU_Training_in_composer.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/examples/training_with_submitit.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/tutorials/train_resnet50_on_aws.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/algorithms.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/functional_api.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/composer_model.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/dataloaders.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/evaluation.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/schedulers.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/time.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/events.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/checkpointing.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/logging.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/performance_tutorials/profiling.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/trainer/performance_tutorials/analyzing_traces.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/notes/distributed_training.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/notes/early_stopping.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/notes/numerics.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/notes/auto_microbatching.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/notes/resumption.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/notes/tensorboard_logger.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/notes/run_name.html',
    'https://docs.mosaicml.com/projects/composer/en/stable/method_cards/methods_overview.html'
    'https://docs.mosaicml.com/projects/mcli/',
    'https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/getting_started.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/environment.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/quick_start_training.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/quick_start_inference.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/quick_start/managing_clusters.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/training/common_commands.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/training/yaml_schema.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/training/run_lifecycle.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/training/working_with_runs.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/training/interactive.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_commands.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/inference/inference_schema.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/inference/working_with_deployments.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/inference/deployment_features.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/python/python_api.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/guides/first_llm.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/guides/sweeps.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/guides/advanced_sweeps_with_optuna.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/git.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/system_dependencies.html'
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/pypi.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/wandb.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/integrations/comet.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/docker.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/git.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/env.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/wandb.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/mosaicml.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/s3.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/oci.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/coreweave.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/cloudflare.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/mounted.html',
    'https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/ssh.html',
    'https://docs.mosaicml.com/projects/streaming/',
    'https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/installation.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/quick_start.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/user_guide.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/dataset_format.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/dataset_conversion_guide.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/compression.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/hashing.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/environments.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/shuffling.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/sampling.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/how_to_guides/configure_cloud_storage_credentials.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/how_to_guides/dataset_conversion_to_mds_format.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/examples/cifar10.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/examples/facesynthetics.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/examples/synthetic_nlp.html',
    'https://docs.mosaicml.com/projects/streaming/en/stable/examples/multiprocess_dataset_conversion.html',
]


class WebScraper:

    def __init__(self, path: str, target_links: list[str] = all_links):
        self.target_links = target_links
        self.destination_folder = os.path.join(path, 'scraped')

        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)

    def _clean_text(self, text: str) -> str:
        """Cleans the extracted text by removing excessive newlines and
        spaces."""
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()  # Remove starting and ending white spaces
        return text

    def _extract_codecells(self, soup: BeautifulSoup) -> list[str]:
        code_blocks = []

        for pre_tag in soup.find_all(
                'pre', id=lambda x: x and x.startswith('codecell')):
            # Combining the text from each span within the pre tag
            code_text = ''.join(
                span.get_text() for span in pre_tag.find_all('span'))
            code_blocks.append(code_text)

        return code_blocks

    @staticmethod
    def url_to_filename(url: str) -> str:
        return url.replace('/',
                           '{slash}').replace('.',
                                              '{dot}').replace(':', '{colon}')

    def scrape(self) -> None:
        for link in self.target_links:
            self._save_content_from_link(link)

    def _save_content_from_link(self, link: str) -> None:
        try:
            link_response = urllib.request.urlopen(link)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # Not Found
                return
            else:
                raise  # You might want to consider propagating the exception for other HTTP errors.
        except Exception as e:
            return

        link_content = link_response.read().decode('utf-8')

        # Detect content type based on file extension or MIME type
        if link.endswith('.html') or 'text/html' in link_response.headers.get(
                'Content-Type', ''):
            parser_type = 'html.parser'
        else:
            parser_type = 'xml'

        soup_content = BeautifulSoup(link_content, parser_type)

        # Extract the 'codecell' div content if present
        code_cells = self._extract_codecells(soup_content)

        # Extract relevant textual content
        text_sections = soup_content.find_all(
            ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        text_content = '\n'.join(
            section.get_text() for section in text_sections)

        # Add the highlights (code snippets) to the text content
        text_content += '\n\n' + '\n\n'.join(code_cells)

        # Clean the text content for better readability
        text_content = self._clean_text(text_content)

        # Unescape HTML entities for HTML content
        if parser_type == 'html.parser':
            text_content = html.unescape(text_content)

        filename = os.path.join(self.destination_folder,
                                self.url_to_filename(link) + '.txt')
        with open(filename, 'w') as file:
            file.write(text_content)
