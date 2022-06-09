# Mosaic ResNet 

The most efficient recipes for training ResNets on ImageNet.  Follow the steps below to reproduce our results.

The following recipes are provided:

   | Recipe | Training Time | Speed Up Methods |
   | --- | --- | --- |
   | [resnet50_mild.yaml](recipes/resnet50_mild.yaml) | Short | `BCELoss`, `BlurPool`, `FixRes`, `Label Smoothing`, `Progressive Resizing` |
   | [resnet50_medium.yaml](recipes/resnet50_medium.yaml) | Longer | `BCELoss`, `BlurPool`, `FixRes`, `Label Smoothing`, `Progressive Resizing`, `MixUp`, `SAM` |
   | [resnet50_hot.yaml](recipes/resnet50_hot.yaml) | Longest | `BCELoss`, `BlurPool`, `FixRes`, `Label Smoothing`, `Progressive Resizing`, `MixUp`, `SAM`, `RandAugment`, `Stochastic Depth`, `MosaicML ColOut` |

## Prequisites

* [MosaicML's Resnet50 Recipes Docker Image](https://hub.docker.com/r/mosaicml/pytorch_vision/tags)
   * Tag: `mosaicml/pytorch_vision:resnet50_recipes`
   * The image pre-configured with the following dependencies
      * Composer Version: 0.7.1
      * PyTorch Version: 1.11.0
      * CUDA Version: 11.3
      * Python Version: 1.9
      * Ubuntu Version: 20.04
* [Docker](https://www.docker.com/) or your container orchestration framework of choice
* [Imagenet Dataset](http://www.image-net.org/)
    
## Running a Recipe

1. Launch a Docker container using the `mosaicml/pytorch_vision:resnet50_recipes` Docker Image on your training system.
   
   ```
   docker pull mosaicml/pytorch_vision:resnet50_recipes
   docker run -it mosaicml/pytorch_vision:resnet50_recipes
   ``` 
   **Note:** The `mosaicml/resnet50_recipes` Docker image can also be used with your container orchestration framework of choice.

1. Download the ImageNet dataset from http://www.image-net.org/.

1. Create the dataset folder and extract training and validation images to the appropriate subfolders.
   The [following script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) can be used to faciliate this process.
   Be sure to note the directory path of where you extracted the dataset.

1. Pick the recipe you would like to train with and kick off the training run.

   ```
   composer -n {num_gpus} train.py -f recipes/{recipe_yaml}
   ```

   Replace `{num_gpus}` and `{recipe_yaml}` with the total number of GPU's and recipe configuration you would like to train with, respectively.
   For example:
   
   ```
   composer -n 8 train.py -f recipes/resnet50_mild.yaml
   ``

   The example above will train on 8 GPU's using the `mild` recipe.
