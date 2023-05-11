## Model Description

The [instructor-large](https://huggingface.co/hkunlp/instructor-large) and [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) models are both instruction-finetuned text embedding models developed by the NLP Group of The University of Hong Kong. The instructor model series is state-of-the-art on the Massive Text Embedding Benchmark (MTEB).

## Deploy

To deploy the instructor-large model, run: `mcli deploy -f instructor_large.yaml --cluster <your_cluster>`

To deploy the instructor-xl model, run: `mcli deploy -f instructor_xl.yaml --cluster <your_cluster>`.

## Input

```
{
    "input_strings": [
        ["Represent the Science title:", "3D ActionSLAM: wearable person tracking in multi-floor environments"]
    ]
}
```

## Output

```
{
    "data":[
        [
            -0.06155527010560036,0.010419987142086029,0.005884397309273481...-0.03766140714287758,0.010227023623883724,0.04394740238785744
        ]
    ]
}
```
