import argparse
import transformers
from transformers import GPTNeoXTokenizerFast
import torch
from loguru import logger

from models.modeling_gpt_neox_legacy import (
    GPTNeoXForCausalLM as GPtNeoXForCausalLMLegacy,
)
from models.modeling_gpt_neox_new import GPTNeoXForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Convert legacy GPT-NeoX model to new format.")
    parser.add_argument("--legacy_model_dir", type=str, default="EleutherAI/pythia-1b-deduped", help="Path to the legacy model directory.")
    parser.add_argument("--new_model_dir", type=str, default="./save/pythia-1b/ckpt/pythia-1b-deduped-new", help="Path to save the new model.")
    return parser.parse_args()

def main(args):
    
    dtype = torch.float16

    if torch.cuda.is_available():
        logger.info("CUDA is available. Setting seeds for reproducibility.")
        torch.manual_seed(10)
        torch.cuda.manual_seed(10)
        torch.cuda.manual_seed_all(10)
    
    logger.info(f"Loading configuration from {args.legacy_model_dir}")
    config = transformers.AutoConfig.from_pretrained(args.legacy_model_dir)

    logger.info(f"Loading tokenizer from {args.legacy_model_dir}")
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.legacy_model_dir,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    logger.info(f"Loading legacy model from {args.legacy_model_dir}")
    legacy_model = GPtNeoXForCausalLMLegacy.from_pretrained(
        pretrained_model_name_or_path=args.legacy_model_dir,
        config=config,
        torch_dtype=dtype,
        device_map="cuda",
    )

    logger.info(f"Loading new model from {args.legacy_model_dir}")
    new_model = GPTNeoXForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.legacy_model_dir,
        config=config,
        torch_dtype=dtype,
        device_map="cuda",
    )

    logger.info("Transferring weights to new format")
    n_layers = len(new_model.gpt_neox.layers)
    for layer_i in range(n_layers):
        logger.info(f"Processing layer {layer_i + 1}/{n_layers}")
        layer = new_model.gpt_neox.layers[layer_i]
        legacy_layer = legacy_model.gpt_neox.layers[layer_i]
        q_heads, k_heads, v_heads = [], [], []
        q_heads_b, k_heads_b, v_heads_b = [], [], []
        head_size = config.hidden_size // config.num_attention_heads

        for h in range(config.num_attention_heads):
            st = 3 * h * head_size
            q_heads.append(legacy_layer.attention.query_key_value.weight.data[st:st+head_size])
            q_heads_b.append(legacy_layer.attention.query_key_value.bias.data[st:st+head_size])

            k_heads.append(legacy_layer.attention.query_key_value.weight.data[st+head_size:st+2*head_size])
            k_heads_b.append(legacy_layer.attention.query_key_value.bias.data[st+head_size:st+2*head_size])

            v_heads.append(legacy_layer.attention.query_key_value.weight.data[st+2*head_size:st+3*head_size])
            v_heads_b.append(legacy_layer.attention.query_key_value.bias.data[st+2*head_size:st+3*head_size])
        
        layer.attention.q_proj.weight.data = torch.cat(q_heads, dim=0)
        layer.attention.q_proj.bias.data = torch.cat(q_heads_b, dim=0)

        layer.attention.k_proj.weight.data = torch.cat(k_heads, dim=0)
        layer.attention.k_proj.bias.data = torch.cat(k_heads_b, dim=0)

        layer.attention.v_proj.weight.data = torch.cat(v_heads, dim=0)
        layer.attention.v_proj.bias.data = torch.cat(v_heads_b, dim=0)
    
        layer.attention.o_proj.weight.data = legacy_layer.attention.dense.weight.data
        layer.attention.o_proj.bias.data = legacy_layer.attention.dense.bias.data

    logger.info("Checking model equivalence")
    input_ids = torch.randint(1, 100, (1, 100)).cuda()
    logger.info(f"Generated input IDs with shape: {input_ids.shape}")

    legacy_model_output = legacy_model.generate(input_ids, do_sample=False, max_length=128)
    new_model_output = new_model.generate(input_ids, do_sample=False, max_length=128)
    logger.info(f"Legacy model output: {legacy_model_output}")
    logger.info(f"New model output: {new_model_output}")

    if torch.allclose(legacy_model_output, new_model_output, atol=1e-3):
        logger.info("Outputs are close enough")
    else:
        logger.error("Outputs differ")
        exit(1)

    logger.info(f"Saving new model to {args.new_model_dir}")
    new_model.save_pretrained(args.new_model_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.new_model_dir)
    logger.info("Model and tokenizer saved successfully")

if __name__ == "__main__":
    args = parse_args()
    main(args)