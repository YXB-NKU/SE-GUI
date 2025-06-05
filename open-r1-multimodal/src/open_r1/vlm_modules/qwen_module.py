from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        
        # pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        pattern = r"<tool_call>.*?\{.*\[\d+,\s*\d+\].*\}.*?</tool_call>"
        completion_contents = [completion[0]["content"] for completion in completions]
        print(completion_contents)
        print('-'*100)
        # print(completion_contents)
        # print('-'*100)
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]     

    
    def format_reward(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
 

    def point_reward(completions, solution, **kwargs):
        """Calculate reward based on whether the predicted point is inside the bounding box and its distance from the box center."""
        import re
        import json
        import os
        from datetime import datetime
        import math

        # 从每个 completion 中提取 content
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        # 遍历每个 content 和对应的 solution
        for content, sol in zip(contents, solution):
            reward = 0.0
            log_details = None
            try:
                # 使用正则表达式提取 <tool_call> 标签中的内容
                tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                if tool_call_match:
                    tool_call_content = tool_call_match.group(1).strip()
                    # 解析 JSON
                    tool_call_json = json.loads(tool_call_content)
                    arguments = tool_call_json.get("arguments", {})
                    coordinate = arguments.get("coordinate", None)
                    # 检查坐标是否是一个长度为 2 的列表
                    if coordinate and isinstance(coordinate, list) and len(coordinate) == 2:
                        x, y = coordinate
                        # 确保 x 和 y 是数值类型
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            # 提取边界框和图像尺寸
                            box = sol[:4]  # [x_min, y_min, x_max, y_max]
                            img_width, img_height = sol[4], sol[5]

                            # 检查点是否在边界框内
                            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                                base_reward = 1.0
                            else:
                                base_reward = 0.0

                            # 计算边界框中心
                            cx = (box[0] + box[2]) / 2
                            cy = (box[1] + box[3]) / 2

                            # 归一化坐标
                            nx = x / img_width
                            ny = y / img_height
                            ncx = cx / img_width
                            ncy = cy / img_height

                            # 计算边界框中心到图像四个角的归一化距离
                            d1 = math.sqrt((ncx - 0)**2 + (ncy - 0)**2)
                            d2 = math.sqrt((ncx - 1)**2 + (ncy - 0)**2)
                            d3 = math.sqrt((ncx - 0)**2 + (ncy - 1)**2)
                            d4 = math.sqrt((ncx - 1)**2 + (ncy - 1)**2)
                            max_d = max(d1, d2, d3, d4)

                            # 计算点到中心的归一化距离
                            d = math.sqrt((nx - ncx)**2 + (ny - ncy)**2)
                            d_normalized = d / max_d if max_d > 0 else 0
                            decay_term = 1 - d_normalized**2 if d <= 1 else 0

                            # 总奖励
                            reward = base_reward + decay_term

                            # 为日志记录准备数据
                            log_details = {
                                "extracted_coordinate": [x, y],
                                "base_reward": base_reward,
                                "decay_term": decay_term
                            }
            except Exception as e:
                # 如果解析失败或发生异常，reward 保持为 0.0
                pass

            rewards.append(reward)

            # 如果启用 DEBUG_MODE，则记录详细信息
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Point-in-box reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution box: {sol[:4]}\n")
                    f.write(f"Image size: {sol[4]}x{sol[5]}\n")
                    if log_details:
                        f.write(f"Extracted coordinate: {log_details['extracted_coordinate']}\n")
                        f.write(f"Base reward: {log_details['base_reward']}\n")
                        f.write(f"Decay term: {log_details['decay_term']}\n")
                    else:
                        f.write("Failed to extract coordinate\n")

        return rewards    

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards