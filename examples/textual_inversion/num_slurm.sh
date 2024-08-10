#!/bin/bash

GPU_INFO="--partition viscam --account viscam --gpu_type 3090 --cpus_per_task 8 --num_gpus 1 --mem 100G"
# GPU_INFO="--partition viscam --account viscam --gpu_type a5000 --cpus_per_task 8 --num_gpus 1 --mem 64G"
# GPU_INFO="--partition viscam --account viscam --gpu_type titanrtx --cpus_per_task 8 --num_gpus 1 --mem 64G"

# GPU_INFO="--partition svl --account viscam --gpu_type titanrtx --cpus_per_task 8 --num_gpus 1 --mem 64G"

EXTRA_GPU_INFO="exclude=viscam1,viscam5,viscam7,svl[1-6],svl[8-10]"

num=1

python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
--proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
--job "08-10-infer-normal-chair" --command "python inference.py --prompt "a photo of a dog" --model_id "results-black_chair_rendered" --output_image_name "output_imgs/dog.png""  $GPU_INFO "$EXTRA_GPU_INFO"

python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
--proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
--job "08-10-infer-normal-cat" --command "python inference.py --prompt "a photo of a dog" --model_id "textual_inversion_cat" --output_image_name "output_imgs/dog2.png""  $GPU_INFO "$EXTRA_GPU_INFO"


# vals=("0" "0.1" "0.02" "0.00001" "0.0000001")
# for val in "${vals[@]}"; do
    # python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    # --proj_dir /viscam/projects/image2Blender/differentiable_engine --conda_env diff_engine \
    # --job "07-12-dp-stool-mask-ref_mask-loss-${val}" --command "python training/training_transform_with_mask.py --initial_xml initial_xmls/stool.xml --target_image ref_imgs/stool.png --target_mask ref_imgs/stool_binary_mask.npy --output_dir logs/2024-07-12-dp-transform-stool-mask-ref_mask-loss-${val} --num_epochs 1000 --constraint_factor ${val} --learning_rate 0.05"  $GPU_INFO "$EXTRA_GPU_INFO"