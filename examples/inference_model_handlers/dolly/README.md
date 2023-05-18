## Model Description

An instruction fine-tuned model licensed for comercial use by Databricks. It's based on pythia-12b and trained with the databricks-dolly-15k dataset. Check out the [Dolly-v2-12b huggingface repo](https://huggingface.co/databricks/dolly-v2-12b) for more information.

## Deploy

To deploy, simply run `mcli deploy -f dolly.yaml --cluster <your_cluster>`.

## Input

```
{
    {
        "input_strings": "Write 3 reasons why you should train an AI model on domain specific data set.",
        "temperature": 0.01
    }
}
```

```{csv-table}
:header: >
:    "Parameters", "Type", "Required", "Default", "Description"
:widths: 20, 20, 5, 10, 50

"input_string","List[str]","yes","N/A","The prompt to generate a completion for."
"top_p","float","no","0.95","Defines the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p."
"temperature","float","no","0.8","The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability."
"max_length","int","no","256","Defines the maximum length in tokens of the output summary."
"use_cache","bool","no","TRUE","Whether to use KV cacheing during autoregressive decoding. This will use more memory but improve speed."
"do_sample","bool","no","TRUE","Whether or not to use sampling, use greedy decoding otherwise."
```

## Output

```
{
    'data': [
        '1. The model will be more accurate.\n2. The model will be more efficient.\n3. The model will be more interpretable.'
    ]
}
```
