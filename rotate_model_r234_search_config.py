import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import ace_tools_open as tools
from loguru import logger
import argparse

def read_rotation_config(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    config = eval(lines[0].strip().split("Current Rotation Config: ")[1])

    data = {}

    for line in lines[1:]:
        parts = line.strip().split(", ")
        if parts[0] == "layer_idx":
            continue
        if len(parts) == 4:
            layer_idx, rotation_type, _, quant_error = parts  # Ignore layer_type
            key = (int(layer_idx), rotation_type)
            quant_error = float(quant_error)

            if key in data:
                data[key] += quant_error
            else:
                data[key] = quant_error

    df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in data.items()],
        columns=["layer_idx", "rotation_type", "quant_error"],
    )

    df["quant_error"] = df["quant_error"].round(4).apply(lambda x: f"{x:.4f}")

    df["is_rotate"] = df.apply(
        lambda row: (
            config["in_block_rotation"]
            .get(str(row["layer_idx"]), {})
            .get(f"is_rotate_{row['rotation_type']}", False)
        ),
        axis=1,
    )

    return df, config



def save_final_rotation_config(merged_df, folder_path):
    final_rotation_config = {
        "in_block_rotation": {},
        "is_rotate_R1": True,
        "is_search_rotation_config": False,
    }

    for layer_idx in merged_df["layer_idx"].unique():
        layer_idx_str = str(layer_idx)
        final_rotation_config["in_block_rotation"][layer_idx_str] = {
            f"is_rotate_{rotation_type}": bool(merged_df[
                (merged_df["layer_idx"] == layer_idx) &
                (merged_df["rotation_type"] == rotation_type)
            ]["is_rotate_optimal"].values[0])
            for rotation_type in ["R2", "R3", "R4"]
        }

    # Save JSON file
    json_path = os.path.join(folder_path, "final_rotation_config.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(final_rotation_config, json_file, indent=4)


    logger.info(f"Final rotation config saved to: {json_path}")


def main(folder_path):

    file1 = os.path.join(folder_path, "quant_error_r1.txt")
    file2 = os.path.join(folder_path, "quant_error_r1234.txt")

    df1, config1 = read_rotation_config(file1)
    df2, config2 = read_rotation_config(file2)

    merged_df = pd.merge(
        df1,
        df2,
        on=["layer_idx", "rotation_type"],
        suffixes=("_file1", "_file2"),
        how="outer",
    )

    merged_df["is_rotate_optimal"] = merged_df.apply(
        lambda row: (
            row["is_rotate_file1"]
            if row["quant_error_file1"] <= row["quant_error_file2"]
            else row["is_rotate_file2"]
        ),
        axis=1,
    )


    merged_df.to_csv(os.path.join(folder_path, "merged_rotation_config.csv"), index=False)

    save_final_rotation_config(merged_df, folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="./rotation_config/qwen/", help="Path to save quantization error logs")
    args = parser.parse_args()
    
    main(args.output_folder)