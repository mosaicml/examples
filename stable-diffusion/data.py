import random

import torch
from torch.utils.data import DataLoader

import numpy as np
from composer.utils import dist


from datasets import load_dataset
from torchvision import transforms
from composer.core import DataSpec



def collate_fn(examples: dict):
    image_tensor = torch.stack(
        [example["image_tensor"] for example in examples])
    image_tensor = image_tensor.to(
        memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"image_tensor": image_tensor, "input_ids": input_ids}


def build_pokemon_datapsec(tokenizer: callable,
                           resolution: int,
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
        transforms.Normalize([0.5], [0.5]),
    ])

    def tokenize_captions(examples: dict,
                      tokenizer: callable=tokenizer,
                      caption_column: str = 'text',
                      is_train: bool = True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
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
        images = [image.convert("RGB") for image in examples['image']]
        examples["image_tensor"] = [
            train_transforms(image) for image in images
        ]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with dist.run_local_rank_zero_first():
        dataset = load_dataset("lambdalabs/pokemon-blip-captions", split='train')

    # add pixel_values and input_ids columns (processed images and text)
    dataset = dataset.with_transform(preprocess)
    sampler = dist.get_sampler(dataset,
                            drop_last=drop_last,
                            shuffle=shuffle)
    return DataSpec(dataloader=DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          sampler=sampler,
                                          drop_last=drop_last,
                                          collate_fn=collate_fn,
                                          **dataloader_kwargs))
