<!-- # GUI-Actor -->

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
- [x] Model training and evaluation based on Qwen2-VL
- [x] Model checkpoint
- [ ] Code for grounding verifier
- [ ] Support for Qwen2.5-VL
- [ ] Processed training data
- [ ] Demo

## :bar_chart: Main Results
Table 1. Main results on ScreenSpot-Pro, ScreenSpot, and ScreenSpot-v2 with **Qwen2-VL** as the backbone. † indicates scores obtained from our own evaluation of the official models on Huggingface.
| Method           | Backbone VLM | ScreenSpot-Pro | ScreenSpot | ScreenSpot-v2 |
|------------------|--------------|----------------|------------|----------------|
| **_72B models:_**
| AGUVIS-72B       | Qwen2-VL     | -              | 89.2       | -              |
| UGround-V1-72B   | Qwen2-VL     | 34.5           | **89.4**   | -              |
| UI-TARS-72B      | Qwen2-VL     | **38.1**       | 88.4       | **90.3**       |
| **_7B models:_**
| OS-Atlas-7B      | Qwen2-VL     | 18.9           | 82.5       | 84.1           |
| AGUVIS-7B        | Qwen2-VL     | 22.9           | 84.4       | 86.0†          |
| UGround-V1-7B    | Qwen2-VL     | 31.1           | 86.3       | 87.6†          |
| UI-TARS-7B       | Qwen2-VL     | 35.7           | **89.5**   | **91.6**       |
| GUI-Actor-7B     | Qwen2-VL     | **40.7**       | 88.3       | 89.5           |
| GUI-Actor-7B + Verifier     | Qwen2-VL    | 44.2       | 89.7       | 90.9           |
| **_2B models:_**
| UGround-V1-2B    | Qwen2-VL     | 26.6           | 77.1       | -              |
| UI-TARS-2B       | Qwen2-VL     | 27.7           | 82.3       | 84.7           |
| GUI-Actor-2B     | Qwen2-VL     | **36.7**       | **86.5**   | **88.6**       |
| GUI-Actor-2B + Verifier     | Qwen2-VL    | 41.8       | 86.9       | 89.3           |

Table 2. Main results on the ScreenSpot-Pro and ScreenSpot-v2 with **Qwen2.5-VL** as the backbone.
| Method         | Backbone VLM | ScreenSpot-Pro | ScreenSpot-v2 |
|----------------|---------------|----------------|----------------|
| **_7B models:_**
| Qwen2.5-VL-7B  | Qwen2.5-VL    | 27.6           | 88.8           |
| Jedi-7B        | Qwen2.5-VL    | 39.5           | 91.7           |
| GUI-Actor-7B   | Qwen2.5-VL    | **44.6**       | **92.1**       |
| GUI-Actor-7B + Verifier   | Qwen2.5-VL    | 47.7       | 92.5       |
| **_3B models:_**
| Qwen2.5-VL-3B  | Qwen2.5-VL    | 25.9           | 80.9           |
| Jedi-3B        | Qwen2.5-VL    | 36.1           | 88.6           |
| GUI-Actor-3B   | Qwen2.5-VL    | **42.2**       | **91.0**       |
| GUI-Actor-3B + Verifier   | Qwen2.5-VL    | 45.9       | 92.4       |

## :rescue_worker_helmet: Installation
1. Clone this repo to your local machine:
```bash
git clone https://github.com/microsoft/GUI-Actor.git
cd GUI-Actor
```
2. Create a conda environment and install the dependencies:
```bash
conda create -n gui_actor python=3.10
conda activate gui_actor
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -e .
```
## :minidisc: Data Preparation
1. Download the processed data from [here (coming soon)]().
2. Modify the paths in the [data_config.yaml](./data/data_config.yaml) file to point to the downloaded data.

## :building_construction: Model Training
1. Warmup stage:
```bash
bash scripts/warmup.sh
```
2. Full-parameter training stage:
```bash
bash scripts/train.sh
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
model_name_or_path = "microsoft/GUI-Actor-7B-Qwen2-VL"
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

## :memo: Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{wu2025guiactor,
    title={GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents}, 
    author={Qianhui Wu and Kanzhi Cheng and Rui Yang and Chaoyun Zhang and Jianwei Yang and Huiqiang Jiang and Jian Mu and Baolin Peng and Bo Qiao and Reuben Tan and Si Qin and Lars Liden and Qingwei Lin and Huan Zhang and Tong Zhang and Jianbing Zhang and Dongmei Zhang and Jianfeng Gao},
    year={2025},
    eprint={2506.03143},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.03143},
}
```