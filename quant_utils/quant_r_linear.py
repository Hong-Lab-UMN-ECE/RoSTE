import torch
import torch.nn as nn

from .quant_linear import QuantLinear
from rotation_utils.rotation import (
    rotate_R4_act,
    rotate_R4_weight,
    rotate_R2_v,
    rotate_R2_o,
)


class QuantLinear_R_up(QuantLinear):
    def forward(self, x):
        weight = self.weight

        w_q = self._quantize_weights(weight, self.w_bits)
        x_q = self._quantize_activations(x, self.a_bits)

        # if self.config.rotation_config["is_search_rotation_config"]:
        #     quant_error = torch.mean((w_q - weight) ** 2) + torch.mean((x_q - x) ** 2)

        #     file_path = self.config.quant_error_path
        #     with open(file_path, "a") as f:
        #         f.write(f"{self.layer_idx}, R1, w_up, {quant_error.item():.6f}\n")

        output = nn.functional.linear(x_q, w_q, self.bias)

        return output


class QuantLinear_R_down(QuantLinear):
    def forward(self, x):
        weight = self.weight

        if self.config.rotation_config["in_block_rotation"][str(self.layer_idx)][
            "is_rotate_R4"
        ]:
            x = rotate_R4_act(x)
            weight = rotate_R4_weight(weight)

        w_q = self._quantize_weights(weight, self.w_bits)
        x_q = self._quantize_activations(x, self.a_bits)

        if self.config.rotation_config["is_search_rotation_config"]:
            quant_error = torch.mean((w_q - weight) ** 2) + torch.mean((x_q - x) ** 2)

            file_path = self.config.quant_error_path
            with open(file_path, "a") as f:
                f.write(f"{self.layer_idx}, R4, w_down, {quant_error.item():.6f}\n")

        output = nn.functional.linear(x_q, w_q, self.bias)

        return output


class QuantLinear_R_q(QuantLinear):
    def forward(self, x):
        weight = self.weight

        w_q = self._quantize_weights(weight, self.w_bits)
        x_q = self._quantize_activations(x, self.a_bits)

        # if self.config.rotation_config["is_search_rotation_config"]:
        #     quant_error = torch.mean((w_q - weight) ** 2) + torch.mean((x_q - x) ** 2)

        #     file_path = self.config.quant_error_path
        #     with open(file_path, "a") as f:
        #         f.write(f"{self.layer_idx}, R1, w_query, {quant_error.item():.6f}\n")

        output = nn.functional.linear(x_q, w_q, self.bias)

        return output


class QuantLinear_R_k(QuantLinear):
    def forward(self, x):
        weight = self.weight

        w_q = self._quantize_weights(weight, self.w_bits)
        x_q = self._quantize_activations(x, self.a_bits)

        # if self.config.rotation_config["is_search_rotation_config"]:
        #     quant_error = torch.mean((w_q - weight) ** 2) + torch.mean((x_q - x) ** 2)

        #     file_path = self.config.quant_error_path
        #     with open(file_path, "a") as f:
        #         f.write(f"{self.layer_idx}, R1, w_key, {quant_error.item():.6f}\n")

        output = nn.functional.linear(x_q, w_q, self.bias)

        return output


class QuantLinear_R_v(QuantLinear):
    def forward(self, x):
        weight = self.weight

        w_q = self._quantize_weights(weight, self.w_bits)
        x_q = self._quantize_activations(x, self.a_bits)

        # if self.config.rotation_config["is_search_rotation_config"]:
        #     quant_error = torch.mean((w_q - weight) ** 2) + torch.mean((x_q - x) ** 2)

        #     file_path = self.config.quant_error_path
        #     with open(file_path, "a") as f:
        #         f.write(f"{self.layer_idx}, R1, w_value, {quant_error.item():.6f}\n")

        output = nn.functional.linear(x_q, w_q, self.bias)

        if self.config.rotation_config["in_block_rotation"][str(self.layer_idx)][
            "is_rotate_R2"
        ]:
            output = rotate_R2_v(output, self.head_size)

        return output


class QuantLinear_R_o(QuantLinear):
    def forward(self, x):
        weight = self.weight

        if self.config.rotation_config["in_block_rotation"][str(self.layer_idx)][
            "is_rotate_R2"
        ]:
            weight = rotate_R2_o(weight, self.head_size)

        w_q = self._quantize_weights(weight, self.w_bits)
        x_q = self._quantize_activations(x, self.a_bits)

        if self.config.rotation_config["is_search_rotation_config"]:
            quant_error = torch.mean((w_q - weight) ** 2) + torch.mean((x_q - x) ** 2)

            file_path = self.config.quant_error_path
            with open(file_path, "a") as f:
                f.write(f"{self.layer_idx}, R2, w_attn_o, {quant_error.item():.6f}\n")

        output = nn.functional.linear(x_q, w_q, self.bias)

        return output