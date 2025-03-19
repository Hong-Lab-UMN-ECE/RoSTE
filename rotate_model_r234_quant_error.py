import argparse
from transformers import AutoConfig, AutoTokenizer
import torch
from datasets import load_dataset
from loguru import logger
import copy
import os

from models.modeling_gpt_neox_new_rq_online import GPTNeoXForCausalLM as GPTNeoXForCausalLM_RQuant
from models.modeling_qwen2_rq_online import Qwen2ForCausalLM as Qwen2ForCausalLM_RQuant


def main(args):
    
    config = AutoConfig.from_pretrained(args.model_dir)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset")
    dataset = load_dataset("trl-lib/tldr")
    raw_test_dataset = dataset["train"].select(range(128))
    text_column, summary_column = raw_test_dataset.column_names

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            targets,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_test_datasets = raw_test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        desc="Running tokenizer on dataset",
    )

    columns = ["input_ids", "attention_mask", "labels"]
    processed_test_datasets.set_format(type="torch", columns=columns)

    input_ids = processed_test_datasets["input_ids"].cuda()
    attention_mask = processed_test_datasets["attention_mask"].cuda()
    labels = processed_test_datasets["labels"].cuda()

    logger.info("Loading model")
    config_rq = copy.deepcopy(config)

    config_rq.quant_config.update(
        {
            "w_bits": 4,
            "a_bits": 4,
            "kv_bits": 4,
        }
    )

    if "r1234" in args.file_path:
        config_rq.rotation_config.update(
            {
                "is_search_rotation_config": True,
                "is_rotate_R1": True,
                "in_block_rotation": {
                    str(layer_idx): {
                        "is_rotate_R2": True,
                        "is_rotate_R3": True,
                        "is_rotate_R4": True,
                    }
                    for layer_idx in range(config_rq.num_hidden_layers)
                },
            }
        )
    else:
        config_rq.rotation_config.update(
            {
                "is_search_rotation_config": True,
                "is_rotate_R1": True,
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

    logger.info(f"Current Quantization Config: {config_rq.quant_config}")
    logger.info(f"Current Rotation Config: {config_rq.rotation_config}")

    config_rq._attn_implementation = "eager"
    if "pythia" in args.model_dir.lower():
        model = GPTNeoXForCausalLM_RQuant.from_pretrained(
            args.model_dir, torch_dtype="auto", config=config_rq
        )
    elif "qwen" in args.model_dir.lower():
        model = Qwen2ForCausalLM_RQuant.from_pretrained(
            args.model_dir, torch_dtype="auto", config=config_rq
        )
    model = model.cuda()
    model.eval()

    model.config.quant_error_path = args.file_path

    with open(args.file_path, "w") as f:
        f.write(f"Current Rotation Config: {model.config.rotation_config}\n")
        f.write("layer_idx, rotation type, layer_type, quant_error\n")

    logger.info("Running model")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    # logger.info(f"Output logits shape: {outputs.logits.shape}")
    # print(outputs.logits[0,0,0:10])

    logger.info(f"Quantization error saved to {args.file_path}")

    # Clear GPU memory
    del input_ids, attention_mask, labels, model, outputs
    torch.cuda.empty_cache()
    gpu_memory = torch.cuda.memory_allocated()
    logger.info(f"Current GPU memory allocated: {gpu_memory / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./save/qwen2.5-0.5b/ckpt/qwen2.5-0.5b-r1", help="Path to the model directory")
    parser.add_argument("--output_folder", type=str, default="./rotation_config/qwen/", help="Path to save quantization error logs")
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    args.file_path = os.path.join(args.output_folder, "quant_error_r1.txt")
    main(args)
    
    args.file_path = os.path.join(args.output_folder, "quant_error_r1234.txt")
    main(args)
