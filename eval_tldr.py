import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import evaluate
import torch
import os
from loguru import logger
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import transformers
import copy
import argparse

from models.modeling_gpt_neox_new_q import GPTNeoXForCausalLM as GPTNeoXForCausalLM_Quant
from models.modeling_gpt_neox_new_rq_online import GPTNeoXForCausalLM as GPTNeoXForCausalLM_RQuant
from models.modeling_gpt_neox_new import GPTNeoXForCausalLM

from models.modeling_qwen2_q import Qwen2ForCausalLM as Qwen2ForCausalLM_Quant
from models.modeling_qwen2_rq_online import Qwen2ForCausalLM as Qwen2ForCausalLM_RQuant

from utils.train_utils import model_info


def load_model(model_dir, method):

    if "qwen" in model_dir.lower():
        model_arch = "qwen"
    elif "pythia" in model_dir.lower():
        model_arch = "pythia"
    
    if "1b" in model_dir.lower():
        model_bit = "1b"
    elif "0.5b" in model_dir.lower():
        model_bit = "0.5b"
    elif "7b" in model_dir.lower():
        model_bit = "7b"

    if "qwen" == model_arch:
        if method == "roste":
            model = Qwen2ForCausalLM_RQuant.from_pretrained(model_dir, torch_dtype="auto")
        elif method == "ste":
            model = Qwen2ForCausalLM_Quant.from_pretrained(model_dir, torch_dtype="auto")
        elif method == "sft":
            config = AutoConfig.from_pretrained(model_dir)
            # for qwen2.5
            process_word_embeddings = False
            if config.tie_word_embeddings:
                config.tie_word_embeddings = False
                process_word_embeddings = True
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto",config=config)
            if process_word_embeddings:
                model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
        else:
            config = AutoConfig.from_pretrained(model_dir)
            # for qwen2.5
            process_word_embeddings = False
            if config.tie_word_embeddings:
                config.tie_word_embeddings = False
                process_word_embeddings = True
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto",config=config)
            if process_word_embeddings:
                model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    elif "pythia" == model_arch:
        if method == "roste":
            model = GPTNeoXForCausalLM_RQuant.from_pretrained(model_dir, torch_dtype="auto")
        elif method == "ste":
            model = GPTNeoXForCausalLM_Quant.from_pretrained(model_dir, torch_dtype="auto")
        elif method == "sft":
            model = GPTNeoXForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")

    return model



# https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py
def main(args) -> None:

    accelerator = Accelerator()

    if not accelerator.is_main_process:
        logger.remove()

    set_seed(2024)

    logger.info("loading model")
    model = load_model(args.model_dir, args.method)
    
    model_info(model)

    logger.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = accelerator.prepare(model, tokenizer)

    dataset = load_dataset("trl-lib/tldr")
    raw_test_dataset = dataset["test"].select(range(6528))
    logger.info(f"Dataset shape: {raw_test_dataset.shape}")

    text_column, summary_column = raw_test_dataset.column_names

    prompts = raw_test_dataset[text_column]
    completions = raw_test_dataset[summary_column]
    logger.info(f"Prompts shape: {len(prompts)}")
    logger.info(f"Completions shape: {len(completions)}")

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

    with accelerator.main_process_first():
        processed_test_datasets = raw_test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=16,
            desc="Running tokenizer on dataset",
        )

    eval_dataloader = DataLoader(
        processed_test_datasets.with_format("torch"), batch_size=args.batch_size
    )
    # num_data = 6528 = batch_size * num_gpus * 51 = 16 * 8 * 51
    eval_dataloader = accelerator.prepare(eval_dataloader)

    model.eval()
    predictions = []
    for batch in tqdm(eval_dataloader, desc="Generating summaries"):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()

            summaries = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions.extend(summaries)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("*" * os.get_terminal_size().columns)
        logger.info("Predictions:")
        logger.info(predictions[:2])

        def extract_tldr(text: str) -> str:
            keyword = "TL;DR:"
            return (
                text.split(keyword, 1)[1].strip() if keyword in text else text.strip()
            )

        processed_predictions = [extract_tldr(pred) for pred in predictions]
        logger.info("Processed Predictions:")
        logger.info(processed_predictions[:2])

        references = completions
        logger.info("References:")
        logger.info(references[:2])

        logger.info(f"Number of predictions: {len(processed_predictions)}")
        logger.info(f"Number of references: {len(references)}")

    if accelerator.is_main_process:
        rouge = evaluate.load("rouge")
        results = rouge.compute(
            predictions=processed_predictions, references=references
        )

        logger.info(f"ROUGE-1: {results['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {results['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {results['rougeL']:.4f}")
        logger.info(f"ROUGE-Lsum: {results['rougeLsum']:.4f}")


    torch.distributed.barrier() 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-0.5B", help="Path to the model directory")
    parser.add_argument("--method", type=str, choices=["roste", "ste", "sft", "base"], required=True, help="Quantization method")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()
    
    main(args)