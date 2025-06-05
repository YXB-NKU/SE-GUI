# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig, Qwen2VLGRPOVLLMTrainer,Qwen2VLGRPOTrainer
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 4028160
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy","format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=4028160,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

import json
import os
import random
from PIL import Image
import yaml
from torch.utils.data import Dataset

class LazySupervisedDataset(Dataset):
    """A dataset class to process conversations with system, human, and GPT messages, including images."""
    def __init__(self, data_path: str, script_args, question_template: str = None):
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the data file (.json or .yaml).
            script_args: Arguments containing image_root and other configurations.
            question_template (str, optional): Kept for compatibility, not used here.
        """
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template  # Unused but kept for compatibility

        # Load data based on file type
        if data_path.endswith(".json"):
            # Direct JSON file containing conversations
            with open(data_path, "r") as json_file:
                self.list_data_dict = json.load(json_file)
            print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        elif data_path.endswith(".yaml"):
            # Original YAML-based loading (for backward compatibility)
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets", [])
                for data in datasets:
                    json_path = data.get("json_path")
                    if json_path.endswith(".jsonl"):
                        cur_data_dict = [json.loads(line.strip()) for line in open(json_path, "r")]
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
            print(f"Loaded {len(self.list_data_dict)} samples from YAML config")
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.list_data_dict)

    def __getitem__(self, i):
        """
        Retrieve a processed sample by index.

        Args:
            i (int): Index of the sample.

        Returns:
            dict: Contains 'image', 'prompt', and 'solution'.
        """
        example = self.list_data_dict[i]
        conversations = example["conversations"]
        images = example.get("images", [])
        bbox = example.get("bbox", [])

        # Extract messages (assuming one of each role)
        try:
            system_message = next(msg["value"] for msg in conversations if msg["from"] == "system")
            human_message = next(msg["value"] for msg in conversations if msg["from"] == "human")
            gpt_message = next(msg["value"] for msg in conversations if msg["from"] == "gpt")
        except StopIteration:
            raise ValueError("Conversation missing required system, human, or gpt message.")

        # Handle image if present
        image = None
        image_root = self.script_args.image_root
        if "<image>" in human_message and images:
            image_path = os.path.join(image_root, images[0])
            # Fallback: try another sample if image is missing
            tries = 0
            max_tries = 10
            while tries < max_tries and not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, selecting another sample")
                i = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[i]
                conversations = example["conversations"]
                images = example.get("images", [])
                try:
                    system_message = next(msg["value"] for msg in conversations if msg["from"] == "system")
                    human_message = next(msg["value"] for msg in conversations if msg["from"] == "human")
                    gpt_message = next(msg["value"] for msg in conversations if msg["from"] == "gpt")
                    
                except StopIteration:
                    tries += 1
                    continue
                if "<image>" not in human_message or not images:
                    image_path = None
                    break
                image_path = os.path.join(image_root, images[0])
                tries += 1
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
            elif tries >= max_tries:
                print("Warning: No valid image found after max tries, proceeding without image")
                image = None
        height,width = image.size if image else (0, 0)
        resized_height, resized_width = smart_resize(height, width)
        image = image.resize((resized_height, resized_width))
        print(f"Image size: {image.size}")
        # Construct user content with image if applicable
        if image and "<image>" in human_message:
            # Split human message around <image> placeholder
            parts = human_message.split("<image>", 1)
            user_content = []
            if parts[0]:  # Text before <image>
                user_content.append({"type": "text", "text": parts[0]})
            user_content.append({"type": "image"})  # Image placeholder
            if len(parts) > 1 and parts[1]:  # Text after <image>
                user_content.append({"type": "text", "text": parts[1]})
        else:
            user_content = human_message  # Plain text if no image

        # Build the prompt
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        # Return processed sample
        return {
            "image": image,        # PIL Image or None
            "prompt": prompt,      # List of messages for the model
            "solution": bbox  # GPT response (e.g., tool call)
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    # print("Module file:", vlm_module_cls.__module__)
    # print("available attributes:",dir(vlm_module_cls))
    # print("using vlm module:", vlm_module_cls.__name__)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.point_reward,
        # "accuracy_v2": vlm_module_cls.point_reward_v2,
        "format": vlm_module_cls.format_reward_rec,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="rec"))

    trainer_cls = Qwen2VLGRPOTrainer
    print('-'*100)
    print(script_args.max_pixels)
    print(script_args.min_pixels)
    print('-'*100)
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        attn_implementation=model_args.attn_implementation,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
