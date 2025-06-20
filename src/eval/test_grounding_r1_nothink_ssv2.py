from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

steps = 3800
if rank == 0:
    print("Steps: ", steps)
# #RUN_NAME = "base"
# RUN_NAME = "Qwen2.5-VL-7B-GRPO-GUI-Grounding_showui_desktop_high_quality_attention_filtered_only_one_continual_dense_reward_quadratic_decay_0.5_format_bs16_kl0.004_nothink_10e"
# #MODEL_PATH="/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/Qwen2.5-VL-7B-Instruct"
# MODEL_PATH=f"/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/VLM-R1/src/open-r1-multimodal/output/{RUN_NAME}/checkpoint-{steps}" 
# OUTPUT_PATH="./logs/rec_results_{DATASET}_{RUN_NAME}_{STEPS}.json"
#RUN_NAME = "base"

MODEL_PATH= "ByteDance-Seed/UI-TARS-2B-SFT"
OUTPUT_PATH="./logs/rec_results_ui_tras_2B.json"

BSZ=32
DATA_ROOT = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/ScreenSpot-Pro-GUI-Grounding/ScreenSpot-v2"

TEST_DATASETS = ['screenspot_desktop_v2','screenspot_mobile_v2','screenspot_web_v2']
IMAGE_ROOT = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/ScreenSpot-Pro-GUI-Grounding/ScreenSpot-v2/screenspotv2_image"


# TEST_DATASETS = ['lisa_test']
# IMAGE_ROOT = "/data10/shz/dataset/lisa"


#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)
# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH,max_pixels=3512320,min_pixels=3136)
# processor.image_processor.min_pixels=3136
# processor.image_processor.max_pixels=2007040
print(processor.image_processor.min_pixels)
print(processor.image_processor.max_pixels)
# def extract_point_answer(content):
#     # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
#     answer_tag_pattern = r'<answer>(.*?)</answer>'
#     content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
#     if content_answer_match:
#         content_answer = content_answer_match.group(1).strip()
#         tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content_answer, re.DOTALL)
#         if tool_call_match:
#                 tool_call_content = tool_call_match.group(1).strip()
#                 # 解析 JSON
#                 tool_call_json = json.loads(tool_call_content)
#                 arguments = tool_call_json.get("arguments", {})
#                 coordinate = arguments.get("coordinate", None)
#                 if coordinate and isinstance(coordinate, list) and len(coordinate) == 2:
#                     x, y = coordinate
#                     extracted_coordinate = [x, y]
#                     return extracted_coordinate
#     return [0, 0]


# def extract_point_answer(content):
#     # 尝试在 <answer> 标签中查找内容，如果找不到则返回 [0, 0]
#     tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
#     if tool_call_match:
#         tool_call_content = tool_call_match.group(1).strip()
#         # 首先尝试将 tool_call_content 解析为 JSON
#         try:
#             tool_call_json = json.loads(tool_call_content)
#             print(tool_call_json)
#             arguments = tool_call_json.get("arguments", {})
#             coordinate = arguments.get("coordinate", None)
#             if coordinate and isinstance(coordinate, list) and len(coordinate) == 2:
#                 try:
#                     x = float(coordinate[0])
#                     y = float(coordinate[1])
#                     return [x, y]
#                 except (ValueError, TypeError):
#                     pass  # 如果转换失败，继续尝试正则提取
#         except json.JSONDecodeError:
#             pass  # 如果 JSON 解析失败，继续尝试正则提取
#         # 回退到正则表达式提取两个数字
#         numbers = re.findall(r'\d+(?:\.\d+)?', tool_call_content)
#         if len(numbers) >= 2:
#             x = float(numbers[-2])
#             y = float(numbers[-1])
#             return [x, y]
#     return [0, 0]



def extract_point_answer(content):
    # 尝试在 <answer> 标签中查找内容，如果找不到则返回 [0, 0]
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    if tool_call_match:
        tool_call_content = tool_call_match.group(1).strip()
        # 首先尝试将 tool_call_content 解析为 JSON
        try:
            numbers = re.findall(r'\d+(?:\.\d+)?', tool_call_content)
            if len(numbers) >= 2:
                x = float(numbers[-2])
                y = float(numbers[-1])
                return [x, y]
        except json.JSONDecodeError:
            pass  # 如果 JSON 解析失败，继续尝试正则提取
        # 回退到正则表达式提取两个数字
    return [0, 0]

def point_in_box(point, box):
    x,y = point
    if box[0] <= x < box[2] and box[1] <= y < box[3]:
        return 1
    else:
        return 0

num_samples = 2000
num_all_sample = 0
num_correct_sample = 0
for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    random.seed(42)
    random.shuffle(data)
    data = data[:num_samples]
    
    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []

    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['img_filename'])
        width,height = x['img_size'][0],x['img_size'][1]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor = processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels = processor.image_processor.min_pixels,
            max_pixels = processor.image_processor.max_pixels,
        )
        system_content = """You are a helpful assistant.
#Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {{screen_width}}x{{screen_height}}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* key: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n* type: Type a string of text on the keyboard.\n* mouse_move: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* left_click: Click the left mouse button.\n* left_click_drag: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n* right_click: Click the right mouse button.\n* middle_click: Click the middle mouse button.\n* double_click: Double-click the left mouse button.\n* scroll: Performs a scroll of the mouse scroll wheel.\n* wait: Wait specified seconds for the change to happen.\n* terminate: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by action=key.", "type": "array"}, "text": {"description": "Required only by action=type.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by action=mouse_move and action=left_click_drag.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by action=scroll.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by action=wait.", "type": "number"}, "status": {"description": "The status of the task. Required only by action=terminate.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>""".replace("{{screen_width}}", str(resized_width)).replace("{{screen_height}}", str(resized_height))
        message = [
            {
             "role": "system",
             "content": [
                {
                    "type": "text",
                    "text": system_content
                }
              ]
            },
            {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": x['instruction']
                }
            ]
        },
        ]
        # print(message)
        messages.append(message)


    rank_outputs = [] # List to store answers for this rank
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        rank_outputs.extend(batch_output_text)

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    # Gather all outputs from all ranks
    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    # The main process will collect all results
    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        final_output = []
        correct_number = 0

        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example['bbox']
            ground_truth = [ground_truth[0] / input_example['img_size'][0], ground_truth[1] / input_example['img_size'][1], (ground_truth[0]+ground_truth[2]) / input_example['img_size'][0], (ground_truth[1]+ground_truth[3]) / input_example['img_size'][1]]
            model_answer = extract_point_answer(original_output)
            resized_height, resized_width = smart_resize(
            input_example['img_size'][1],
            input_example['img_size'][0],
            factor = processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels = processor.image_processor.min_pixels,
            max_pixels = processor.image_processor.max_pixels,
        )
            model_answer = [model_answer[0]/resized_width,model_answer[1]/resized_height]
            # Count correct answers
            correct = 0
            if model_answer is not None:
                correct = point_in_box(model_answer, ground_truth)
            correct_number += correct
            num_all_sample +=1
            num_correct_sample += correct
            
            # Create a result dictionary for this example
            result = {
                'image': input_example['img_filename'],
                'question': input_example['instruction'],
                'resized_size': [resized_height, resized_width],
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer,
                'correct': correct
            }
            final_output.append(result)

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")

        # Save results to a JSON file
        output_path = OUTPUT_PATH.format(DATASET=ds, RUN_NAME=RUN_NAME, STEPS=steps)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'accuracy': accuracy,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-"*100)
# 将最后的统计和打印移到rank==0的条件块内
    if rank == 0:
        accuracy = num_correct_sample / num_all_sample * 100
        print(f"\nnumber of correct samples: {num_correct_sample}")
        print(f"number of all samples: {num_all_sample}")
        print(f"Accuracy of all datasets: {accuracy:.2f}%")

    # Synchronize all processes
    dist.barrier()




