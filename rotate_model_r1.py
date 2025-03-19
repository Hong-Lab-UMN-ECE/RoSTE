import argparse
from transformers import AutoConfig, AutoTokenizer
import torch
from datasets import load_dataset
from loguru import logger
import copy

from models.modeling_gpt_neox_new_rq_online import GPTNeoXForCausalLM as GPTNeoXForCausalLM_RQuant
from models.modeling_qwen2_rq_online import Qwen2ForCausalLM as Qwen2ForCausalLM_RQuant
from rotation_utils.main import rotate_model
from utils.train_utils import model_info


def main(args):
    
    config = AutoConfig.from_pretrained(args.model_dir)
    
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_dir,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    config_rq = copy.deepcopy(config)
    if not hasattr(config_rq, "quant_config") or config_rq.quant_config is None:
        config_rq.quant_config = {}
    config_rq.quant_config.update(
        {
            "w_bits": 16,
            "a_bits": 16,
            "kv_bits": 16,
            "w_sym": True,
            "a_sym": True,
            "kv_sym": True,
            "w_clip_ratio": 1,
            "a_clip_ratio": 1,
            "kv_clip_ratio": 1,
        }
    )
    
    if not hasattr(config_rq, "rotation_config") or config_rq.rotation_config is None:
        config_rq.rotation_config = {}
    config_rq.rotation_config.update(
        {
            "is_search_rotation_config": False,
            "is_rotate_R1": args.is_rotate_R1,
            "in_block_rotation": {
                str(layer_idx): {
                    "is_rotate_R2": False,
                    "is_rotate_R3": False,
                    "is_rotate_R4": False,
                }
                for layer_idx in range(config_rq.num_hidden_layers)
            },
        }
    )

    logger.info("Loading model")
    config_rq._attn_implementation = "eager"
    if "pythia" in args.model_dir.lower():
        model = GPTNeoXForCausalLM_RQuant.from_pretrained(
            args.model_dir, torch_dtype="auto", config=config_rq
        )
    elif "qwen" in args.model_dir.lower():
        model = Qwen2ForCausalLM_RQuant.from_pretrained(
            args.model_dir, torch_dtype="auto", config=config_rq
        )
    # for qwen2/5
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model_info(model)

    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = None
    original_model = copy.deepcopy(model).cuda()

    logger.info("Rotating model")
    rotated_model = rotate_model(model).cuda()

    if args.is_tldr_data:
        dataset = load_dataset("trl-lib/tldr")
        raw_test_dataset = dataset["train"].select(range(4))
        text_column, summary_column = raw_test_dataset.column_names
        prompt = raw_test_dataset[text_column][2]

        print(f"Prompt: {prompt}")
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to("cuda")
        attn_mask = tokens.attention_mask.to("cuda")
    else:
        input_ids = torch.randint(1, 1000, (1, 512), dtype=torch.long).cuda()
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")

    output1 = original_model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=False,
        max_new_tokens=32,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded_output1 = tokenizer.decode(output1[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"Decoded Output1: {decoded_output1}")

    output2 = rotated_model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=False,
        max_new_tokens=32,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded_output2 = tokenizer.decode(output2[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"Decoded Output2: {decoded_output2}")

    if torch.allclose(output1, output2, atol=1e-3):
        logger.info("Outputs are close enough")
    else:
        logger.error("Outputs differ")
        exit(1)

    if args.is_save:
        logger.info("Saving rotated model")
        rotated_model.save_pretrained(args.rotated_model_dir, safe_serialization=False)
        tokenizer.save_pretrained(args.rotated_model_dir)
        logger.info("Rotated model and tokenizer saved to {}", args.rotated_model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-0.5B", help="Path to the pre-trained model directory")
    parser.add_argument("--is_tldr_data", action="store_true", help="Use TLDR dataset for testing")
    parser.add_argument("--is_rotate_R1", action="store_true", help="Enable R1 rotation")
    parser.add_argument("--is_save", action="store_true", help="Save the rotated model")
    parser.add_argument("--rotated_model_dir", type=str, default="./save/qwen2.5-0.5b/ckpts/qwen2.5-0.5b-r1", help="Path to save the rotated model")
    args = parser.parse_args()

    main(args)
