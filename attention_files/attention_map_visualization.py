import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from io import BytesIO
import base64
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


# def convert_pil_image_to_base64(image):
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()

# def get_qwen2_5vl_prompt_msg(img_url, instruction, screen_width, screen_height):
#     return [
#         {
#             "role": "system",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "You are a helpful assistant."
#                 },
#                 {
#                     "type": "text",
#                     "text": """


# # Tools

# You may call one or more functions to assist with the user query.

# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {{screen_width}}x{{screen_height}}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n* `right_click`: Click the right mouse button.\n* `middle_click`: Click the middle mouse button.\n* `double_click`: Double-click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
# </tools>

# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>""".replace("{{screen_width}}", str(screen_width)).replace("{{screen_height}}", str(screen_height))
#                 }
#             ]
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "min_pixels": 3136,
#                     "max_pixels": 2007040,
#                     "image_url": {
#                         img_url
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": instruction
#                 }
#             ]
#         }
#     ]


# def get_qwen2_5vl_prompt_msg(image, instruction, screen_width, screen_height):
#     return [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "min_pixels": 3136,
#                     "max_pixels": 2007040,
#                     "image_url": {
#                         "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": instruction
#                 }
#             ]
#         }
#     ]


# ===> specify the model path
model_name_or_path = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/VLM-R1-2/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-GUI-Grounding_no_position_high_quality_continual_reward_quadratic_decay_0.5_format_bs16_kl0.004_nothink_4e/checkpoint-6036"
#model_name_or_path = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/saves/qwen2.5_vl-7b_showui_desktop_showui_web_100k_uground_web_100k_amex_bs128_1e6_toolcall/full/sft"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path, 
    device_map="cuda", 
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2" # Do not use flash-attn or getting attention scores will fail!
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name_or_path,max_pixels=2007040,min_pixels=3136)

#image_path_or_url = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/ScreenSpot-Pro-GUI-Grounding/ScreenSpot-Pro/images/illustrator_windows/screenshot_2024-11-29_17-33-36.png"
image_path_or_url = "/data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/VLM-R1/data/showui_desktop_images/message/screen_1.png"
prompt_text = "emoji."
image = Image.open(image_path_or_url).convert('RGB')
# image_path_or_url = "https://github.com/open-compass/MMBench/blob/main/samples/MMBench/1.jpg?raw=true"
# prompt_text = "What python code can be used to generate the output in the image?"
# image = Image.open(requests.get(image_path_or_url, stream=True).raw)

# generate the response

resized_height, resized_width = smart_resize(
    image.height,
    image.width,
    factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
    min_pixels=processor.image_processor.min_pixels,
    max_pixels=2007040,
)
print(resized_width,resized_height)

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
                    "image": f"file://{image_path_or_url}"
                },
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        },
        ]
    
# Preparation for inference
text_input = processor.apply_chat_template(
    message, tokenize=False, add_generation_prompt=True
)
# image_inputs, video_inputs = process_vision_info(message)
image = image.resize((resized_width,resized_height))
inputs = processor(
    text=text_input,
    images=[image],
    padding=True,
    return_tensors="pt",
).to("cuda")
outputs = model.generate(
    **inputs, 
    use_cache=True, 
    max_new_tokens=256, 
    do_sample=False,
    return_dict_in_generate=True,
    output_attentions=True,
)
decoded_text = processor.batch_decode(
    outputs["sequences"], skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]
print("Generated text: ", decoded_text)


# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py
# here it nullifies the attention over the first token (<bos>)
# which in practice we find to be a good idea
from io import BytesIO
from PIL import Image
import requests
import torch
import numpy as np
import cv2


def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        # print(layer.shape)
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
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


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image

# constructing the llm attention matrix
aggregated_prompt_attention = []
print(f"outputs_attention:{len(outputs['attentions'][0])}")
print(len(outputs['attentions']))
# print(len(outputs['attentions'][1]))
for i, layer in enumerate(outputs["attentions"][0]):  # step 0
    # print(layer.shape)
    layer_attns = layer.squeeze(0)  # (num_heads, seq_len(this step), seq_len(previous all))
    attns_per_head = layer_attns.mean(dim=0)  # (seq_len(this step), seq_len(previous all))
    # cur = attns_per_head[:-1].cpu().clone()  # Why [:-1]? 
    cur = attns_per_head.cpu().clone()
    # following the practice in `aggregate_llm_attention`
    # we are zeroing out the attention to the first <bos> token
    # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
    # we don't do this because <bos> is the only token that it can attend to
    cur[1:, 0] = 0.
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    aggregated_prompt_attention.append(cur)
aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

# llm_attn_matrix will be of torch.Size([N, N])
# where N is the total number of input (both image and text ones) + output tokens
llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention) 
    + list(map(aggregate_llm_attention, outputs["attentions"]))
)

# visualize the llm attention matrix
# ===> adjust the gamma factor to enhance the visualization
#      higer gamma brings out more low attention values
gamma_factor = 1
llm_attn_matrix = torch.from_numpy(np.power(llm_attn_matrix.numpy(), 1 / gamma_factor))

# fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
# ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")

# identify length or index of tokens
len_prompt_tokens = len(inputs.input_ids[0])
len_all_tokens = len(outputs["sequences"][0])

# The <|vision_start|> and <|vision_end|> are kept in the tokenized input, thus the +1
vision_token_start = len(tokenizer(decoded_text.split("<|vision_start|>")[0], return_tensors='pt')["input_ids"][0]) + 1
vision_token_end = len(tokenizer(decoded_text.split("<|vision_end|>")[0], return_tensors='pt')["input_ids"][0])
print(f"vision_tokens:{vision_token_end-vision_token_start}")
output_token_start = len_prompt_tokens
print(f"len_prompt_tokens:{len_prompt_tokens}")
output_token_end = len_all_tokens
print(f"len_all_tokens:{len_all_tokens}")
output_token_len = output_token_end - output_token_start


def get_token_img_shape(image_size, patch_size=28):
    """
    Calulate the shape of the token representation of the image, e.g. an image of (286, 237) is mapped to a (10, 8) token array.
    Qwen2.5VL uses ViT of patch size 14, plus a merger of 2*2 tokens, resulting in an equivalent of 28*28 tokens.
    """
    return image_size[0] // patch_size, image_size[1] // patch_size
print("Num of prompt tokens:", len(llm_attn_matrix[len_prompt_tokens:]))
token_shape = get_token_img_shape(image.size, patch_size=28)
print(f"Num of image tokens: {token_shape[0]} per row * {token_shape[1]} per colomn =", vision_token_end - vision_token_start)
print("Num of all tokens (prompt + generated):", len(outputs["sequences"][0].tolist()))


# look at the attention weights over the vision tokens
overall_attn_weights_over_vis_tokens = []
for i, (row, token) in enumerate(
    zip(
        llm_attn_matrix[len_prompt_tokens:], 
        outputs["sequences"][0][len_prompt_tokens:].tolist()
    )
):
    # print(
    #     i + len_prompt_tokens, 
    #     f"{tokenizer.decode(token, add_special_tokens=False).strip():<15}", 
    #     f"{row[vision_token_start:vision_token_end].sum().item():.4f}"
    # )

    overall_attn_weights_over_vis_tokens.append(
        row[vision_token_start:vision_token_end].sum().item()
    )

# plot the trend of attention weights over the vision tokens
fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(overall_attn_weights_over_vis_tokens)
ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
ax.set_xticklabels(
    [tokenizer.decode(token, add_special_tokens=False).strip() for token in outputs["sequences"][0][len_prompt_tokens:].tolist()],
    rotation=75
)
ax.set_title("at each token, the sum of attention weights over all the vision tokens")
fig.savefig("attention_weights.png", dpi=300, bbox_inches="tight")

# Defines how to lay the mask over the image.

# COLORMAP + alpha
def show_mask_on_image(img, mask, alpha=0.3):
    img = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_OCEAN)
    heatmap = np.float32(heatmap) / 255.0
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return np.uint8(255 * overlay), np.uint8(255 * heatmap)


# Alpha
# def show_mask_on_image(img, mask, base_alpha=0.3):
#     img = np.float32(img) / 255.0
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_OCEAN)
#     heatmap = np.float32(heatmap) / 255.0
#     overlay = heatmap * base_alpha + img * np.expand_dims(mask, -1) * (1 - base_alpha)
#     return np.uint8(255 * overlay), np.uint8(255 * heatmap)

# Blue to Red

# def show_mask_on_image(img, mask, alpha=0.3):
#     # Normalize image to 0-1
#     img = np.float32(img) / 255.0
    
#     # Create blue-to-red map from the mask (low values blue, high values red)
#     blue_to_red_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
#     # Blue for low values, Red for high values
#     blue_to_red_map[..., 0] = (255 * mask).astype(np.uint8)  # Blue channel (low values)
#     blue_to_red_map[..., 2] = (255 * (1 - mask)).astype(np.uint8)  # Red channel (high values)
    
#     # Convert to float for proper blending
#     heatmap = np.float32(blue_to_red_map) / 255.0
    
#     # Apply the overlay with blending
#     overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
#     # Return the final image and heatmap (scaled back to 0-255 range)
#     return np.uint8(255 * overlay), np.uint8(255 * heatmap)




# connect with the vision encoder attention
# to visualize the attention over the image

# vis_attn_matrix will be of torch.Size([N, N])
# where N is the number of vision tokens/patches
# `all_prev_layers=True` will average attention from all layers until the selected layer
# otherwise only the selected layer's attention will be used
# vis_attn_matrix = aggregate_vit_attention(
#     model.get_vision_tower().image_attentions,
#     select_layer=-1,
#     all_prev_layers=True
# )

# Looks like we can not get attention scores from the ViT...

num_image_per_row = 8
image_ratio = image.size[0] / image.size[1]
num_rows = output_token_len // num_image_per_row + (1 if output_token_len % num_image_per_row != 0 else 0)
fig, axes = plt.subplots(
    num_rows, num_image_per_row,
    figsize=(10, (10 / num_image_per_row) * image_ratio * num_rows),
    dpi=150
)
plt.subplots_adjust(wspace=0.05, hspace=0.2)

# whether visualize the attention heatmap or
# the image with the attention heatmap overlayed
vis_overlayed_with_attn = True

output_token_inds = list(range(output_token_start, output_token_end))
# for i, ax in enumerate(axes.flatten()):
#     if i >= output_token_len:
#         ax.axis("off")
#         continue

#     target_token_ind = output_token_inds[i]
#     attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
#     attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

#     # -------------If somehow we can get the inner ViT attention, we can do this:-------------
#     # attn_over_image = []
#     # for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
#     #     vis_attn = vis_attn.reshape(grid_size, grid_size)
#     #     # vis_attn = vis_attn / vis_attn.max()
#     #     attn_over_image.append(vis_attn * weight)
#     # attn_over_image = attn_weights_over_vis_tokens
#     # attn_over_image = torch.stack(attn_over_image).sum(dim=0)
#     # attn_over_image = attn_over_image / attn_over_image.max()
#     # ----------------------------------------------------------------------------------------

#     # Calculate attention to the processed visual tokens.
#     token_img_shape = get_token_img_shape((resized_width,resized_height), patch_size=28)  # w, h
#     print(f"token_img_shape:{token_img_shape}")
#     attn_weights_over_vis_tokens = attn_weights_over_vis_tokens.reshape(1, 1, *token_img_shape[::-1])
#     # Rescale the attentions scores to 0-1 for better visualization
#     attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.max()

#     attn_over_image = F.interpolate(
#         attn_weights_over_vis_tokens,
#         size=image.size[::-1], # PyTorch wants (H, W)
#         mode='nearest',
#         # mode='bicubic', align_corners=False
#     ).squeeze()

#     np_img = np.array(image)[:, :, ::-1]
#     img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())
#     ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
#     ax.set_title(
#         tokenizer.decode(outputs["sequences"][0][len_prompt_tokens + i], add_special_tokens=False).strip(),
#         fontsize=7,
#         pad=1
#     )
#     ax.axis("off")
# fig.savefig("attention_visualization.png", dpi=300, bbox_inches="tight")


#收集所有 token 的注意力权重
attn_list = []
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

#将图像转换为 NumPy 格式并可视化
np_img = np.array(image)[:, :, ::-1]
img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())

#创建单个图形并绘制
fig, ax = plt.subplots(1, 1)
ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
ax.set_title("Average Attention Map")
ax.axis("off")

#保存图像
fig.savefig("average_attention_visualization.png", dpi=300, bbox_inches="tight")
plt.close(fig)


import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def is_attention_on_target(attn_map, gt_box, threshold=0.4):
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

def visualize_attention(image, attn_map, gt_box, save_path=None):
    """
    可视化注意力分布与 ground truth 框的叠加效果
    
    Args:
        image (PIL.Image): 原始图像
        attn_map (torch.Tensor): 注意力热图 (H, W)
        gt_box (tuple): (x_min, y_min, x_max, y_max)
        save_path (str): 图片保存路径，默认不保存
    """
    # 转换为numpy格式
    np_img = np.array(image)
    np_attn = attn_map.numpy() if isinstance(attn_map, torch.Tensor) else attn_map
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 显示原始图像
    ax.imshow(np_img)
    
    # 叠加注意力热图（使用jet配色，透明度50%）
    heatmap = ax.imshow(np_attn, alpha=0.5, cmap='jet')
    plt.colorbar(heatmap, ax=ax)  # 添加颜色条
    
    # 绘制ground truth框（红色边框）
    rect = patches.Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    # 标注判定结果
    result = is_attention_on_target(attn_map, gt_box)
    plt.title(f"Attention covers target: {result}", fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# 使用示例 -------------------------------------------------
# 假设已获得以下变量（根据用户代码生成）
# attn_over_image: 注意力热图（已与图像尺寸对齐）
# image: 原始PIL图像对象
# gt_box: ground truth坐标 (x_min, y_min, x_max, y_max)

# 示例数据（需要替换为实际数据）
gt_box = (1712, 1, 1760, 31)  # 替换为实际坐标
# 执行判断
result = is_attention_on_target(attn_over_image, gt_box)
print(f"[判定结果] 注意力聚焦目标区域: {result}")

# 可视化验证（保存到本地）
visualize_attention(image, attn_over_image, gt_box, save_path="attention_visualization.jpg")
# # --------Change the token offset here -----------
# offset = 24
# # ------------------------------------------------
# # Define which output token you want to visualize
# target_token_index = output_token_start + offset
# target_token_str = tokenizer.decode(outputs["sequences"][0][target_token_index], add_special_tokens=False).strip()

# # Get attention weights over vision tokens
# attn_weights_over_vis_tokens = llm_attn_matrix[target_token_index][vision_token_start:vision_token_end]
# attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

# # Rescale the attentions scores to 0-1 for better visualization
# attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.max()

# # Resize attention to image dimensions
# token_img_shape = get_token_img_shape((resized_width,resized_height), patch_size=28)
# attn_weights_over_vis_tokens = attn_weights_over_vis_tokens.reshape(1, 1, *token_img_shape[::-1])
# attn_over_image = F.interpolate(
#     attn_weights_over_vis_tokens,
#     size=image.size[::-1],  # (H, W)
#     mode='nearest'
# ).squeeze()

# # Create heatmap or overlay
# np_img = np.array(image)[:, :, ::-1]
# img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())

# # Plot a large single image
# fig, ax = plt.subplots(figsize=(6, 6 * (image.size[1] / image.size[0])), dpi=150)
# ax.imshow(img_with_attn if vis_overlayed_with_attn else heatmap)
# ax.set_title(f"Attention for token: '{target_token_str}'", fontsize=12)
# ax.axis("off")
# plt.tight_layout()
# plt.savefig("figure2.png")
# plt.show()