from langchain.llms import MosaicML
from mcli import predict, get_inference_deployment

def main():
    deployment = get_inference_deployment("mpt-7b-support-bot-finetuned-pdx6a9")
    context = '# 🥸 ALiBi[\[How to Use\]](#howtouse)[\[Suggested Hyperparameters\]](#suggestedhyperparameters)[\[Technical Details\]](#technicaldetails)[\[Attribution\]](#attribution)[\[API Reference\]](#apireference)`Natural Language Processing`ALiBi (Attention with Linear Biases) dispenses with position embeddings for tokens in transformer-based NLP models, instead encoding position information by biasing the query-key attention scores proportionally to each token pair’s distance. ALiBi yields excellent extrapolation to unseen sequence lengths compared to other position embedding schemes. We leverage this extrapolation capability by training with shorter sequence lengths, which reduces the memory and computation load.| ![Alibi](https://storage.googleapis.com/docs.mosaicml.com/images/methods/alibi.png) | |:--: |*The matrix on the left depicts the attention score for each key-query token pair.'
    question = 'What algorithm dispenses with position embeddings for tokens in transformer-based NLP models, instead encoding position information by biasing the query-key attention scores proportionally to each token pairs distance?'
    llm_prompt = f'Provide a simple answer given the following context to the question. If you do not know, just say "I do not know".\n{context}\nQuestion: {question}'
    
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    PROMPT_FOR_GENERATION_FORMAT = """{intro}
    {instruction_key}
    {instruction}
    {response_key}
    """.format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction="{instruction}",
        response_key=RESPONSE_KEY,
    )
    prompt = PROMPT_FOR_GENERATION_FORMAT.format(
                instruction=llm_prompt,
            )
    print(predict(deployment, {"inputs": [prompt], "parameters": {"output_len": 40, "top_k": 1}})['outputs'])

if __name__ == '__main__':
    main()