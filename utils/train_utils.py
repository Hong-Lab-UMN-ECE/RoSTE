from trl import (
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
    TrlParser,
)
import torch

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class ModelArguments:
    w_bits: Optional[int] = field(
        default=16,
        metadata={"help": "Weight quantization bits.", "choices": [2, 4, 8, 16]},
    )
    a_bits: Optional[int] = field(
        default=16,
        metadata={"help": "Activation quantization bits.", "choices": [2, 4, 8, 16]},
    )
    kv_bits: Optional[int] = field(
        default=16,
        metadata={"help": "KV cache quantization bits.", "choices": [2, 4, 8, 16]},
    )
    w_sym: Optional[bool] = field(
        default=False,
        metadata={"help": "Weight symmetric quantization."},
    )
    a_sym: Optional[bool] = field(
        default=False,
        metadata={"help": "Activation symmetric quantization."},
    )
    kv_sym: Optional[bool] = field(
        default=False,
        metadata={"help": "KV cache symmetric quantization."},
    )
    w_clip_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "Weight clipping ratio."},
    )
    a_clip_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "Activation clipping ratio."},
    )
    kv_clip_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "KV cache clipping ratio."},
    )


def process_args():
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig, ModelArguments))
    script_args, training_args, model_config, model_args = (
        parser.parse_args_and_config()
    )

    return script_args, training_args, model_config, model_args


def safe_save_model_for_hf_trainer(trainer, output_dir):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {}
        for key in state_dict.keys():
            cpu_state_dict[key] = state_dict[key]
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def model_info(model):
    logger.info(model)
    logger.info(model.dtype)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters: {total_params/1e9:.3f} B")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable Parameters: {trainable_params/1e9:.3f} B")

    # useful for single-gpu inference, useles for multi-gpu training
    logger.info(f"Allocated Model Memory: {model.get_memory_footprint()/1e9:.3f} GB")

    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_gb = param_size + buffer_size
    logger.info(f"Model Size: {size_all_gb/1e9:.3f} GB")

    # for name, param in model.named_parameters():
    #     logger.info(f"Parameter: {name}, dtype: {param.dtype}")

    logger.info(f"The model is on: {next(model.parameters()).device}")


def load_training_args(file_dir):
    training_args = torch.load(file_dir)
    return training_args