cd src/open-r1-multimodal
export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

RUN_NAME="Qwen2.5-VL-3B-GRPO-GUI-Grounding_showui_desktop_high_quality_attention_0.2_filtered_continual_dense_reward_quadratic_decay_0.5_format_bs16_kl0.004_nothink_10e_max_pixel_4028160"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_gui_grounding.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/gui_grounding.yaml \
    --image_root /data/vjuicefs_ai_camera_jgroup_research/public_data/11178625/LLaMA-Factory/VLM-R1/data  \
    --max_prompt_length 4096 \
    --max_completion_length 1400 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true 