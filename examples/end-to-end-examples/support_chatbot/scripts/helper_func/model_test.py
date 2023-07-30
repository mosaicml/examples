from langchain.llms import MosaicML
from mcli import predict, get_inference_deployment

def main():
    deployment = get_inference_deployment("mpt-7b-support-bot-finetuned-pdx6a9")
    context = '# 🎨 AugMix[\[How to Use\]](#howtouse)[\[Suggested Hyperparameters\]](#suggestedhyperparameters)[\[Technical Details\]](#technicaldetails)[\[Attribution\]](#attribution)[\[API Reference\]](#apireference)`Computer Vision`For each data sample, AugMix creates an _augmentation chain_ by sampling `depth` image augmentations from a set (e.g. translation, shear, contrast). It then applies these augmentations sequentially with randomly sampled intensity. This is repeated `width` times in parallel to create `width` different augmented images. The augmented images are then combined via a random convex combination to yield a single augmented image, which is in turn combined via a random convex combination sampled from a Beta(`alpha`, `alpha`) distribution with the original image.'
    question = 'What creates width sequences of depth image augmentations, applies each sequence with random intensity, and returns a convex combination of the width augmented images and the original image such that coefficients for mixing the augmented images are drawn from a uniform Dirichlet(alpha, alpha, ...) distribution and the coefficient for mixing the combined augmented image and the original image is drawn from a Beta(alpha, alpha) distribution, using the same alpha??'
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