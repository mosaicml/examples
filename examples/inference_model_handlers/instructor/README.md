## Model Description

The [instructor-large](https://huggingface.co/hkunlp/instructor-large) and [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) models are both instruction-finetuned text embedding models developed by the NLP Group of The University of Hong Kong. The instructor model series is state-of-the-art on the [Massive Text Embedding Benchmark (MTEB)](https://huggingface.co/spaces/mteb/leaderboard).

## Deploy

To deploy the instructor-large model, run: `mcli deploy -f instructor_large.yaml --cluster <your_cluster>`

To deploy the instructor-xl model, run: `mcli deploy -f instructor_xl.yaml --cluster <your_cluster>`.

## Input

The inputs to the instructor model are pairs of strings where the first is the instruction and the second is the text. More information can be found in the [instructor repo](https://github.com/HKUNLP/instructor-embedding).

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
