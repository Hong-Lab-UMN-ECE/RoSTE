import numpy as np
import random
import torch
from loguru import logger

from .fuse_norm import fuse_layer_norms
from .rotation import rotate_R1, rotate_R2_offline, rotate_R4_offline


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logger.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def rotate_model(model):

    set_seed(42)
    fuse_layer_norms(model)

    if model.config.rotation_config["is_rotate_R1"]:
        rotate_R1(model)

    
    cleanup_memory(verbos=True)
    
    return model
