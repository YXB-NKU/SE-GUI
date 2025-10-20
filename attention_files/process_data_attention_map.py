import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
import torch.nn.functional as F
#Distributed environment setup
def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return local_rank, world_size, rank

#Initialize distributed environment
local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

#Load model and processor (once per process)
model_name_or_path = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-GUI-Grounding_showui_desktop_high_quality_attention_filtered_only_one_continual_dense_reward_quadratic_decay_0.5_format_bs16_kl0.004_nothink_10e/checkpoint-3860"  # Replace with your actual model path
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map={"": local_rank},
    torch_dtype=torch.bfloat16
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name_or_path, max_pixels=2007040, min_pixels=3136)

def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        # print(layer.shape)
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            torch.tensor([0.]),
            attns_per_head[-1][1:].cpu(),
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

def get_token_img_shape(image_size, patch_size=28):
    """
    Calulate the shape of the token representation of the image, e.g. an image of (286, 237) is mapped to a (10, 8) token array.
    Qwen2.5VL uses ViT of patch size 14, plus a merger of 2*2 tokens, resulting in an equivalent of 28*28 tokens.
    """
    return image_size[0] // patch_size, image_size[1] // patch_size

def compute_attention(image, image_path_or_url, prompt_text):
    """Core function to compute attention heatmap"""
    # Image preprocessing
    resized_height, resized_width = smart_resize(
        image.height, image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=2007040
    )
    image = image.resize((resized_width, resized_height))
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
    # Build message template
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": system_content}]
    }, {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path_or_url}"},
            {"type": "text", "text": prompt_text}
        ]
    }]
    # Model inference
    inputs = processor(
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            use_cache=True, 
            max_new_tokens=256, 
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    # 解码输入的文本以定位视觉标记
    decoded_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)  # 添加解码逻辑
    aggregated_prompt_attention = []

    for i, layer in enumerate(outputs["attentions"][0]):  # step 0
        layer_attns = layer.squeeze(0)  
        attns_per_head = layer_attns.mean(dim=0)  
        cur = attns_per_head.cpu().clone()
        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
    llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention) 
    + list(map(aggregate_llm_attention, outputs["attentions"]))
    )
    gamma_factor = 1
    llm_attn_matrix = torch.from_numpy(np.power(llm_attn_matrix.numpy(), 1 / gamma_factor))
    len_prompt_tokens = len(inputs.input_ids[0])
    len_all_tokens = len(outputs["sequences"][0])
    vision_token_start = len(tokenizer(decoded_text.split("<|vision_start|>")[0], return_tensors='pt')["input_ids"][0]) + 1
    vision_token_end = len(tokenizer(decoded_text.split("<|vision_end|>")[0], return_tensors='pt')["input_ids"][0])
    output_token_start = len_prompt_tokens
    output_token_end = len_all_tokens
    output_token_len = output_token_end - output_token_start
    token_shape = get_token_img_shape(image.size, patch_size=28)

    attn_list = []
    vis_overlayed_with_attn = True
    output_token_inds = list(range(output_token_start, output_token_end))
    for i in range(output_token_len):
        target_token_ind = output_token_inds[i]
        attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
        # 归一化，使注意力权重总和为 1
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()
        attn_list.append(attn_weights_over_vis_tokens)

    #将注意力权重堆叠并取平均
    avg_attn = torch.stack(attn_list).mean(dim=0)  # 形状：(num_visual_tokens,)

    #获取 token 图像形状
    token_img_shape = get_token_img_shape((resized_width, resized_height), patch_size=28)  # 返回 (grid_w, grid_h)
    print(token_img_shape)
    #重塑为一维张量到二维网格
    avg_attn_reshaped = avg_attn.reshape(1, 1, *token_img_shape[::-1])  # 形状：(1, 1, grid_h, grid_w)

    #归一化，使最大值为 1（与原始代码保持一致）
    avg_attn_reshaped = avg_attn_reshaped / avg_attn_reshaped.max()

    #插值到原始图像大小
    attn_over_image = F.interpolate(
        avg_attn_reshaped,
        size=image.size[::-1],  # (H, W)
        mode='nearest',
    ).squeeze()  # 形状：(H, W)

    return attn_over_image

def is_attention_on_target(attn_map, gt_box, threshold=0.2):
    """
    判断注意力是否聚焦在 ground truth box 区域
    
    Args:
        attn_map (torch.Tensor): 注意力热图 (H, W)
        gt_box (tuple): ground truth 框坐标 (x_min, y_min, x_max, y_max)
        threshold (float): 判定阈值 (0-1)，默认0.7

    Returns:
        bool: True 表示注意力覆盖目标区域，False 反之
    """
    # 确保注意力图已经归一化
    if attn_map.max() > 1 or attn_map.min() < 0:
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # 提取目标区域注意力权重
    x_min, y_min, x_max, y_max = gt_box
    target_attn = attn_map[y_min:y_max, x_min:x_max]
    
    if target_attn.numel() == 0:  # 处理空区域
        return False
    
    # 双标准判定策略
    has_peak = (target_attn > threshold).any().item()       # 存在显著激活点
    higher_than_global = (target_attn.mean() > attn_map.mean()).item() # 高于全局平均
    print(attn_map.mean().item())
    
    return has_peak and higher_than_global

# def process_dataset(input_path, output_path, image_base_dir):
#     # Load dataset
#     with open(input_path) as f:
#         dataset = json.load(f)
#     # Data partitioning: Split dataset across GPUs
#     per_rank_data = len(dataset) // world_size
#     start_idx = rank * per_rank_data
#     end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(dataset)
#     rank_data = dataset[start_idx:end_idx]
#     # Process data on this rank
#     results = []
#     for item in tqdm(rank_data, desc=f"Rank {rank} processing"):
#         try:
#             # Parse data
#             img_path = os.path.join(image_base_dir, item["images"][0])
#             gt_box = tuple(item["bbox"][:4])
#             prompt = next(m["value"].replace("<image>", "") for m in item["conversations"] if m["from"] == "human")
#             # Load image
#             image = Image.open(img_path).convert("RGB")
#             # Compute attention heatmap
#             attn_map = compute_attention(image,img_path, prompt)
#             # Store result
#             results.append({
#                 "instruction": prompt.strip(),
#                 "gt_box": gt_box,
#                 "image_path": item["images"][0],
#                 "attention_score": bool(is_attention_on_target(attn_map, gt_box)),
#             })
#         except Exception as e:
#             print(f"Rank {rank}: Error processing {item.get('images', ['unknown'])[0]}: {str(e)}")
#     # Gather results from all GPUs
#     all_results = [None] * world_size
#     dist.all_gather_object(all_results, results)
#     # Merge and save results on rank 0
#     if rank == 0:
#         final_results = []
#         for res in all_results:
#             final_results.extend(res)
#         with open(output_path, "w") as f:
#             json.dump(final_results, f, indent=2)
#         print(f"Results saved to {output_path}")



def process_dataset(input_path, output_path, image_base_dir):
    # Load dataset
    with open(input_path) as f:
        dataset = json.load(f)
    
    # Get world size and rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Data partitioning
    per_rank_data = len(dataset) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(dataset)
    rank_data = dataset[start_idx:end_idx]
    
    batch_size = 100  # 可调整的批次大小
    if rank == 0:
        # 初始化输出文件
        with open(output_path, 'w') as f:
            json.dump([], f)

    for batch_start in range(0, len(rank_data), batch_size):
        batch_results = []
        batch_data = rank_data[batch_start : batch_start + batch_size]
        
        for idx, item in enumerate(tqdm(batch_data, desc=f"Rank {rank} processing batch")):
            try:
                # 解析数据
                img_path = os.path.join(image_base_dir, item["images"][0])
                gt_box = tuple(item["bbox"][:4])
                prompt = next(m["value"].replace("<image>", "") for m in item["conversations"] if m["from"] == "human")
                
                # 处理图像
                image = Image.open(img_path).convert("RGB")
                attn_map = compute_attention(image, img_path, prompt)
                
                # 生成结果条目
                batch_results.append({
                    "index": start_idx + batch_start + idx,
                    "instruction": prompt.strip(),
                    "gt_box": gt_box,
                    "image_path": item["images"][0],
                    "attention_score": bool(is_attention_on_target(attn_map, gt_box)),
                })
                
            except Exception as e:
                # 错误处理
                error_msg = f"Error processing {item.get('images', ['unknown'])[0]}: {str(e)}"
                print(f"Rank {rank}: {error_msg}")
                batch_results.append({
                    "index": start_idx + batch_start + idx,
                    "error": error_msg,
                    "image_path": item.get('images', [None])[0],
                    "instruction": "N/A",
                    "gt_box": None,
                    "attention_score": None
                })

        # 收集所有rank的结果
        gathered_batch = [None] * world_size
        dist.all_gather_object(gathered_batch, batch_results)

        # 仅rank 0写入结果
        if rank == 0:
            # 合并并排序结果
            merged_results = []
            for results in gathered_batch:
                merged_results.extend(results)
            merged_results.sort(key=lambda x: x["index"])
            
            # 读取现有数据并扩展
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            existing_data.extend(merged_results)
            
            # 写入更新后的数据
            with open(output_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            print(f"Rank {rank}: Successfully saved batch {batch_start//batch_size} with {len(merged_results)} entries")

        # 同步所有进程
        dist.barrier()
        print(f"Rank {rank}: Finished processing batch {batch_start//batch_size}")

    # 最终同步（可选）
    dist.barrier()


if __name__ == "__main__":
    input_path = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/VLM-R1/data/rec_jsons_processed/Aria_desktop_qwen25vl_grpo_2007040_wo_think_img_size_10k.json"  #这里是原始数据集文件的路径
    output_path = "Aria_desktop_high_quality_qwen25vl_2007040_attention_0.2_two_stage.json"  # 这里是经过attention过滤的文件路径
    image_base_dir = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/VLM-R1/data"  #这里是数据集的路径