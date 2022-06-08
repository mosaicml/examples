ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

#########################
# MosaicML ResNet Recipes
#########################

# Copy required files
COPY recipes /mosaicml-resnet/recipes
COPY requirements.txt /mosaicml-resnet/
COPY train.py /mosaicml-resnet/

# Set image default working directory
WORKDIR /mosaicml-resnet

# Install dependencies
RUN pip install -r requirements.txt
