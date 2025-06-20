<!-- # GUI-Actor -->
<h1 align="center">Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning</h1>
<div align="center">
<hr>

[Xinbin Yuan]()<sup>1,2</sup>&nbsp;
[Jian Zhang]()<sup>2</sup>&nbsp;
[Kaixin Li]()<sup>3</sup>&nbsp;
[Zhuoxuan Cai]()<sup>2</sup>&nbsp;
[Lujian Yao]()<sup>2</sup>&nbsp;
[Jie Chen]()<sup>1</sup><br>
[Enguang Wang]()<sup>1</sup>&nbsp;
[Qibin Hou]()<sup>1,†</sup>&nbsp;
[Jinwei Chen]()<sup>2</sup>&nbsp;
[Peng-Tao Jiang]()<sup>2</sup>&nbsp;
[Bo Li]()<sup>2,†</sup>&nbsp;

<sup>1</sup> Nankai University&nbsp;&nbsp;<sup>2</sup> vivo Mobile Communication Co., Ltd&nbsp;&nbsp;<sup>3</sup>National University of Singapore<br>
<sup>†</sup> Corresponding authors.  

<h4>
<a href="https://www.arxiv.org/pdf/2505.12370">📄 arXiv Paper</a> &nbsp; 
<a href="https://aka.ms/SE-GUI/">🌐 Project Page</a> &nbsp; 
<a href="https://huggingface.co/XinBB/SE-GUI-7B">🤗 Hugging Face Models</a>
</h4>

</div>

<div align="center">
<img src="images/performance.png" width="100%">
</div>

<div align="center">
<img src="images/attention.png" width="100%">
</div>

Figure 1. **Top**: Model performance on the Grounding benchmarks and Agent benchmarks. Higher and more left is better. With only 3k training samples, SE-GUI-3B/7B reaches scores up to 35.8/47.2. **Bottom**: Illustration of action attention with prompt "Pin Jack's Conversation."

## :sparkles: Highlights
🎯 **Seed Data Curation**  We curate a 3,018-sample dataset by filtering out vague, inaccurate, or overly simple tasks from a larger candidate pool. This ensures linguistic consistency and balanced task complexity, promoting better generalization and stable performance across scenarios.

🚀 **Group Relative Policy Optimization with Dense Point Reward.** To combat sparse rewards, we designed a continuous reward mechanism that evaluates the proximity between predictions and ground truth. This provides smoother feedback, enabling the model to learn from near-misses and gradually refine its grounding behavior.

🔥 **Self-Evolutionary Reinforcement Fine-Tuning.** We implement an iterative learning loop, where attention maps serve as intermediate supervision signals. These maps highlight which visual tokens the model attends to for each instruction, helping align its focus with relevant interface elements over time.

<!-- ## :fire: News
* **[2025.06.03]**  We released the GUI-Actor training/inference code and model checkpoints!
-->

## :bookmark_tabs: Todos
We will be releasing all the following contents:
- [x] Model training and evaluation based on Qwen2.5-VL
- [x] Model checkpoint
- [ ] Code for attention generation
- [ ] Seed data
- [ ] Demo

## :bar_chart: Main Results
Table 1. Main results on ScreenSpot-Pro, ScreenSpot, and ScreenSpot-v2 with **Qwen2.5-VL** as the backbone. † indicates scores obtained from our own evaluation of the official models on Huggingface.
| Method            | ScreenSpot-Pro | ScreenSpot | ScreenSpot-v2 |
|------------------|----------------|------------|----------------|
| **_72B models:_**
| AGUVIS-72B           | -              | 89.2       | -              |
| UGround-V1-72B       | 34.5           | **89.4**   | -              |
| UI-TARS-72B          | **38.1**       | 88.4       | **90.3**       |
| **_7B models:_**
| OS-Atlas-7B          | 18.9           | 82.5       | 84.1           |
| AGUVIS-7B            | 22.9           | 84.4       | 86.0†          |
| UGround-V1-7B        | 31.1           | 86.3       | 87.6†          |
| UI-TARS-7B           | 35.7           | -          | -              |
| SE-GUI-7B            | **47.2**       | 88.2       | 90.3           |
| **_3B models:_**
| UI-R1-3B      |   17.8           | -       | -           |
| GUI-R1-3B     |   28.2           |  -      | -           |
| SE-GUI-3B     |  **35.9**        | -       | -           |


## :rescue_worker_helmet: Installation
1. Clone this repo to your local machine:
```bash
https://github.com/YXB-NKU/SE-GUI.git
cd SE-GUI
```
2. Create a conda environment and install the dependencies:
```bash
conda create -n se_gui python=3.10
conda activate se_gui
bash setup.sh
```
## :minidisc: Data Preparation
1. Download the processed data from [here (coming soon)]().
2. Modify the paths in the [data_config.yaml](./data/data_config.yaml) file to point to the downloaded data.

## :building_construction: Model Training
```
Full-parameter training stage:
```bash
bash src/open-r1-multimodal/run_scripts/run_grpo_gui_grounding.sh
```

## :checkered_flag: Evaluation on GUI Grounding Benchmarks
For evaluation on ScreenSpot and ScreenSpot-v2, you can directly run the scripts under the `scripts/` folder like `python eval/screenSpot.py` or `python eval/screenSpot_v2.py`.

For evaluation on ScreenSpot-Pro, you first need to download the data from [here](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro), then run the following command:
```bash
python eval/screenSpot_pro.py --save_path <path_to_save_results> --data_path <path_to_data_dir>
```

Example usage:
```python
import torch

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from transformers import Qwen2VLProcessor
from gui_actor.constants import chat_template
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.inference import inference


# load model
model_name_or_path = "SE-GUI-7B"
data_processor = Qwen2VLProcessor.from_pretrained(model_name_or_path)
tokenizer = data_processor.tokenizer
model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2"
).eval()

# prepare example
dataset = load_dataset("rootsautomation/ScreenSpot")["test"]
example = dataset[0]
print(f"Intruction: {example['instruction']}")
print(f"ground-truth action region (x1, y1, x2, y2): {[round(i, 2) for i in example['bbox']]}")

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.",
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": example["image"], # PIL.Image.Image or str to path
                # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
            },
            {
                "type": "text",
                "text": example["instruction"]
            },
        ],
    },
]

# inference
pred = inference(conversation, model, tokenizer, data_processor, use_placeholder=True, topk=3)
px, py = pred["topk_points"][0]
print(f"Predicted click point: [{round(px, 4)}, {round(py, 4)}]")

# >> Model Response
# Intruction: close this window
# ground-truth action region (x1, y1, x2, y2): [0.9479, 0.1444, 0.9938, 0.2074]
# Predicted click point: [0.9709, 0.1548]
```

## :+1: Acknowledgements

This project is built upon the following projects. Thanks for their great work!
- [Transformers](https://github.com/huggingface/transformers)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [AGUVIS](https://github.com/xlang-ai/aguvis)

We also thank the authors of the following projects for their insightful work, as well as for providing datasets and engaging in valuable discussions.
- [AGUVIS](https://github.com/xlang-ai/aguvis)
- [UGround](https://github.com/OSU-NLP-Group/UGround)
- [OS-Atlas](https://github.com/OS-Copilot/OS-Atlas)
- [SeeClick](https://github.com/njucckevin/SeeClick)
- [GUI-Actor](https://github.com/microsoft/GUI-Actor)

## :memo: Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{yuan2025enhancing,
  title={Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning},
  author={Yuan, Xinbin and Zhang, Jian and Li, Kaixin and Cai, Zhuoxuan and Yao, Lujian and Chen, Jie and Wang, Enguang and Hou, Qibin and Chen, Jinwei and Jiang, Peng-Tao and others},
  journal={arXiv preprint arXiv:2505.12370},
  year={2025}
}
```