### Overview

In this tutorial, we will be creating an application that answers questions based on [SEC Form 10-K](https://en.wikipedia.org/wiki/Form_10-K) documents. These are annual financial performance summaries that all companies are required to file with the SEC annually. Analyzing these filings and extracting information from them is an important part staying on top of financial data about public companies.

The goal of this tutorial is to show how we can build a prototype of this application using the MosaicML platform. We will cover:
- Setting up your development environment and the MosaicML platform
- Downloading and processing the data for domain specific pretraining of [MPT-7b](https://huggingface.co/mosaicml/mpt-7b)
- Using the MosaicML inference integration in LangChain to build the text retrieval component
- Deploying your finetuned MPT-7b for inference using the MosaicML inference service
- Building a simple frontend to tie everything together

At the end of this tutorial, you will have a simple web application that answers questions about SEC Form 10-K documents using a model that you finetuned and deployed using MosaicML!

### Setup
describes both setups, each step will show how to run with either

## Local setup

## MosaicML platform setup
make sure you do the mcli things

### Acquiring the SEC 10-K data
We will use the version of the 10-K data on HuggingFace (https://huggingface.co/datasets/JanosAudran/financial-reports-sec). Note that reprocessing the data may improve the quality, as this version of the data appears to largely be missing tables, which are an important part of financial statements. Each row in this dataset corresponds to a sentence, so we will need to reprocess the data into full text documents.

download it
rejoin the lines
upload to cloud

### Finance Finetune MPT-7b
train/val/test split based on time
process data using foundry script
finetune

### Instruct Finetune MPT-7b
just use the dolly thing in foundry

### Deploy the finetuned model
deployment yaml

### Retrieval using MosaicML inference
query the endpoint, make the index

### Application with gradio
play around
prompt dataset to eval

### What next?
