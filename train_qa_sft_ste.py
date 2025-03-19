import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen2Config
from loguru import logger
from trl import SFTTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from accelerate import Accelerator
import sys
import copy
import json
import os

from utils.train_utils import process_args, safe_save_model_for_hf_trainer, model_info

from models.modeling_qwen2_q import Qwen2ForCausalLM as Qwen2ForCausalLM_Quant
from models.modeling_gpt_neox_new_q import GPTNeoXForCausalLM as GPTNeoXForCausalLM_Quant

def train():

    accelerator = Accelerator()
    if not accelerator.is_main_process:
        logger.remove()
    
    script_args, training_args, model_config, model_args = process_args()

    logger.info("Start to load model...")
    model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_kwargs = dict(
        revision=model_config.model_revision,
        torch_dtype=model_config.torch_dtype,
    )

    
    config = Qwen2Config.from_pretrained(model_config.model_name_or_path)
    config_rq = copy.deepcopy(config)
    if not hasattr(config_rq, "quant_config") or config_rq.quant_config is None:
        config_rq.quant_config = {}

    config_rq.quant_config.update({
        "w_bits": model_args.w_bits,
        "a_bits": model_args.a_bits,
        "kv_bits": model_args.kv_bits,
        "w_sym": model_args.w_sym,
        "a_sym": model_args.a_sym,
        "kv_sym": model_args.kv_sym,
        "w_clip_ratio": model_args.w_clip_ratio,
        "a_clip_ratio": model_args.a_clip_ratio,
        "kv_clip_ratio": model_args.kv_clip_ratio,
    })
    if "qwen" in model_config.model_name_or_path:
        model = Qwen2ForCausalLM_Quant.from_pretrained(
            pretrained_model_name_or_path=model_config.model_name_or_path,
            config=config_rq,
            **model_kwargs,
        )
    else:
        model = GPTNeoXForCausalLM_Quant.from_pretrained(
            pretrained_model_name_or_path=model_config.model_name_or_path,
            config=config_rq,
            **model_kwargs,
        )

    model_info(model)

    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = None
    logger.info("Complete model loading...")
    # sys.exit()

    logger.info("Start to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        use_fast=True,
        clean_up_tokenization_spaces=True,
        padding_side="right",
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.chat_template = None
    logger.info("Complete tokenizer loading...")

    logger.info("Start to load dataset...")
    dataset = load_dataset(script_args.dataset_name, num_proc=16)
    logger.info("Complete dataset loading...")

    if training_args.do_train:
        logger.info("Start to train...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split],
            tokenizer=tokenizer,
        )
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info("Complete training...")

    if training_args.do_eval:
        logger.info("Start to evaluate...")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("Complete evaluation...")

    logger.info("Saving the model...")
    trainer.save_model(training_args.output_dir)
    # safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    
    if accelerator.is_main_process:
        training_args_dict = training_args.to_dict()
        training_args_path = os.path.join(
            training_args.output_dir, "training_args.json"
        )
        with open(training_args_path, "w", encoding="utf-8") as f:
            json.dump(training_args_dict, f, ensure_ascii=False, indent=4)

    logger.info("Complete saving...")

    accelerator.wait_for_everyone()


    model = Qwen2ForCausalLM_Quant.from_pretrained(
        training_args.output_dir, torch_dtype="auto"
    )

    model_info(model)

    logger.info("Done")

if __name__ == "__main__":
    train()
