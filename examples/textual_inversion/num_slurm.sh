#!/bin/bash

GPU_INFO="--partition viscam --account viscam --gpu_type 3090 --cpus_per_task 8 --num_gpus 1 --mem 100G"
# GPU_INFO="--partition viscam --account viscam --gpu_type a5000 --cpus_per_task 8 --num_gpus 1 --mem 64G"
# GPU_INFO="--partition viscam --account viscam --gpu_type titanrtx --cpus_per_task 8 --num_gpus 1 --mem 64G"

# GPU_INFO="--partition svl --account viscam --gpu_type titanrtx --cpus_per_task 8 --num_gpus 1 --mem 64G"

EXTRA_GPU_INFO="exclude=viscam1,viscam5,viscam7,svl[1-6],svl[8-10]"

num=1

# python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
# --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
# --job "08-10-infer-normal-chair" --command 'python inference.py --prompt "a photo of a dog" --model_id "results-black_chair_rendered" --output_image_name "output_imgs/dog.png"'  $GPU_INFO "$EXTRA_GPU_INFO"

# python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
# --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
# --job "08-10-infer-normal-cat" --command 'python inference.py --prompt "a photo of a dog" --model_id "textual_inversion_cat" --output_image_name "output_imgs/dog2.png"'  $GPU_INFO "$EXTRA_GPU_INFO"

# note: this is 10x lower lr, original is 5.0e-04
vals=(500 1000 200)
for val in "${vals[@]}"; do
    python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    --job "08-10-infer-normal-chair" --command 'accelerate launch textual_inversion.py   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"   --train_data_dir="images/black_chair_rendered"   --learnable_property="object"   --placeholder_token="<chair-toy>"   --initializer_token="chair"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-05   --scale_lr   --lr_scheduler="constant"   --lr_warmup_steps=0   --push_to_hub   --output_dir="results-black_chair_rendered-${val}" && python inference.py --prompt "a photo of a <chair-toy>" --model_id "results-black_chair_rendered-${val}" --output_image_name "output_imgs/black_chair_rendered-${val}.png"'  $GPU_INFO "$EXTRA_GPU_INFO"

    python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    --job "08-10-infer-normal-chair" --command 'accelerate launch textual_inversion.py   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"   --train_data_dir="images/brown_chair_rendered"   --learnable_property="object"   --placeholder_token="<chair-toy>"   --initializer_token="chair"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-05   --scale_lr   --lr_scheduler="constant"   --lr_warmup_steps=0   --push_to_hub   --output_dir="results-brown_chair_rendered-${val}" && python inference.py --prompt "a photo of a <chair-toy>" --model_id "results-brown_chair_rendered-${val}" --output_image_name "output_imgs/brown_chair_rendered-${val}.png"'  $GPU_INFO "$EXTRA_GPU_INFO"

    python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    --job "08-10-infer-normal-chair" --command 'accelerate launch textual_inversion.py   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"   --train_data_dir="images/cat-one"   --learnable_property="object"   --placeholder_token="<chair-toy>"   --initializer_token="cat"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-05   --scale_lr   --lr_scheduler="constant"   --lr_warmup_steps=0   --push_to_hub   --output_dir="results-cat-one-${val}" && python inference.py --prompt "a photo of a <chair-toy>" --model_id "results-cat-one-${val}" --output_image_name "output_imgs/cat-one-${val}.png"'  $GPU_INFO "$EXTRA_GPU_INFO"

done