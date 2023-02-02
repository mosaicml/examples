from composer import Trainer
from composer.utils import dist
from examples.llm.src.hf_t5 import create_hf_t5
import torch
from accelerate import init_empty_weights

def main():
    # tag = 't5-base'
    tag = 'google/flan-t5-xxl'

    do_meta = False
    use_pretrained = True
    model_config={'dropout_rate': 0.0}#, 'vocab_size': 32100}
    if do_meta:
        with init_empty_weights():
            model = create_hf_t5(tag, use_pretrained=use_pretrained, model_config=model_config)
    else:
        model = create_hf_t5(tag, use_pretrained=use_pretrained, model_config=model_config)

    model.eval()
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(tag)
    enc_toks = tokenizer('Hello there this is a test', return_tensors='pt')
    dec_toks = tokenizer('And you have passed', return_tensors='pt')
    batch = {
        'input_ids': enc_toks['input_ids'],
        'attention_mask': enc_toks['attention_mask'],
        'labels': dec_toks['input_ids'],
        'decoder_attention_mask': dec_toks['attention_mask']
    }
    if dist.get_global_rank() == 0 and not do_meta:
        for p in model.parameters():
            print('\n\nBEFORE\n')
            print(p)
            break
        before_output = model.forward(batch)

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'min_params': 4e8,
        'mixed_precision': 'DEFAULT',
        'activation_checkpointing': False,
        'activation_cpu_offload': False,
        'verbose': False,
    }

    trainer = Trainer(
        model=model,
        fsdp_config=fsdp_config,
        precision='amp_bf16',
    )
    model = trainer.state.model

    model.eval()
    after_output = model.forward({k: v.cuda(torch.cuda.current_device()) for k, v in batch.items()})

    # model.model.from_pretrained
    if dist.get_global_rank() == 0:
        for p in model.parameters():
            print('\n\nAFTER\n')
            print(p)
            break
        if not do_meta:
            print('Mean/Max abs diffs:')
            for k in before_output.keys():
                if not isinstance(before_output[k], torch.Tensor):
                    continue
                b = before_output[k].cuda()
                a = after_output[k]
                d = b - a
                ad = d.view(-1).abs()
                print(f'\t{k}\n\t{ad.mean()}, {ad.max()}\n')

if __name__ == "__main__":
    main()
