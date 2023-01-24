# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
"""image captioning dataset creation tools and preprocessing."""
import random
import math

import torch
from torch.utils.data import DataLoader

import numpy as np
from composer.utils import dist

from datasets.load import load_dataset
from torchvision import transforms
from composer.core import DataSpec
from torch.utils.data import Dataset



def collate_fn(examples: dict):
    image_tensor = torch.stack(
        [example["image_tensor"] for example in examples])
    image_tensor = image_tensor.to(
        memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"image_tensor": image_tensor, "input_ids": input_ids}


def build_image_caption_datapsec(name:str,
                                resolution: int,
                                tokenizer: callable,
                                mean:list=[0.5],
                                std:list=[0.5],
                                image_column:str = 'image',
                                caption_column:str = 'text',
                                center_crop: bool = True,
                                random_flip: bool = True,
                                *,
                                batch_size: int,
                                drop_last: bool = True,
                                shuffle: bool = True,
                                seed: int = 1337,
                                **dataloader_kwargs):
    train_transforms = transforms.Compose([
        transforms.Resize(resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution)
        if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip()
        if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    def tokenize_captions(examples: dict,
                          tokenizer: callable = tokenizer,
                          caption_column: str = caption_column,
                          is_train: bool = True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(
                    random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions,
                           max_length=tokenizer.model_max_length,
                           padding="max_length",
                           truncation=True,
                           return_tensors="pt")
        return inputs.input_ids

    def preprocess(examples: dict):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["image_tensor"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with dist.run_local_rank_zero_first():
        dataset = load_dataset(name, split='train')

    # add pixel_values and input_ids columns (processed images and text)
    dataset = dataset.with_transform(preprocess)
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)
    return DataSpec(dataloader=DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          sampler=sampler,
                                          drop_last=drop_last,
                                          collate_fn=collate_fn,
                                          **dataloader_kwargs))


def build_prompt_dataspec(prompts: list, batch_size: int, **dataloader_kwargs):
    """prompt dataset for eval"""
    dataset = PromptDataset(prompts)
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)
    
    def split_list(l, microbatch_size: int): # needed for gradient accumulation
        if len(l) < microbatch_size:
            microbatch_size = len(l)
        num_microbatches = math.ceil(len(l) / microbatch_size)
        # Note: this is to match the behavior of tensor.chunk, which is used in _split_tensor
        chunked_microbatch_size = math.ceil(len(l) / num_microbatches)
        return [l[start:start + chunked_microbatch_size] for start in range(0, len(l), chunked_microbatch_size)]

    return DataSpec(dataloader=DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          sampler=sampler,
                                          drop_last=False,
                                          **dataloader_kwargs),
                                          split_batch=split_list,
                                          get_num_samples_in_batch=lambda x: len(x))


class PromptDataset(Dataset):
    """Convert a list of strings into a pytorch dataset because the Trainer expects a dataloader for batching purposes"""
    def __init__(self, prompts:list):
        self.prompts = prompts

    def __getitem__(self, index:int):
        return self.prompts[index]

    def __len__(self):
        return len(self.prompts)
