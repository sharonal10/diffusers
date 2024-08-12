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
# vals=(500 1000 2000)
# for val in "${vals[@]}"; do
    # python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    # --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    # --job "08-10-infer-black-chair-${val}" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'   --train_data_dir='images/black_chair_rendered'  --save_as_full_pipeline --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='chair'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-05   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0   --output_dir='results-black_chair_rendered-${val}' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-black_chair_rendered-${val}' --output_image_name 'output_imgs/black_chair_rendered-${val}.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

    # python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    # --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    # --job "08-10-infer-brown-chair-${val}" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'   --train_data_dir='images/brown_chair_rendered' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='chair'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-05   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0    --output_dir='results-brown_chair_rendered-${val}' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-brown_chair_rendered-${val}' --output_image_name 'output_imgs/brown_chair_rendered-${val}.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

    # python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    # --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    # --job "08-10-infer-cat-${val}" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'   --train_data_dir='cat-one' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='cat'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-05   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0  --output_dir='results-cat-one-${val}' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-cat-one-${val}' --output_image_name 'output_imgs/cat-one-${val}.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

# done

# accelerate launch textual_inversion_phase_2.py   --pretrained_model_name_or_path='results-black_chair_rendered-500-LR5e-06'  --train_data_dir='images/black_chair_real' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-real>'   --initializer_token='chair' --existing_token='<chair-toy>'  --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=1   --learning_rate=5.0e-04   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0  --output_dir='results-black_chair_real-test2'

# python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-black_chair_real-test2' --output_image_name 'output_imgs/results-black_chair_real-test-rendered.png' && python inference.py --prompt 'a photo of a <chair-real>' --model_id 'results-black_chair_real-test2' --output_image_name 'output_imgs/results-black_chair_real-test-real.png' && python inference.py --prompt 'a photo of a chair in the style of <chair-real>' --model_id 'results-black_chair_real-test2' --output_image_name 'output_imgs/results-black_chair_real-test-both3.png'


vals=(500 1000 2000 3000)
for val in "${vals[@]}"; do

    python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    --job "08-10-p2-black-chair-4pics-${val}-LR5e-06" --command "accelerate launch textual_inversion_phase_2.py   --pretrained_model_name_or_path='results-black_chair_rendered-500-LR5e-06'  --train_data_dir='images/black_chair_real-4pics' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-real>'   --initializer_token='chair' --existing_token='<chair-toy>'  --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-06   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0  --output_dir='results-black_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06' && python inference.py --prompt 'a photo of a <chair-real>' --model_id 'results-black_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06' --output_image_name 'output_imgs/results-black_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06.png' && python inference.py --prompt 'a photo of a <chair-toy> in the style of <chair-real>' --model_id 'results-black_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06' --output_image_name 'output_imgs/results-black_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06-both.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

    python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    --job "08-10-p2-brown-chair-4pics-${val}-LR5e-06" --command "accelerate launch textual_inversion_phase_2.py   --pretrained_model_name_or_path='results-brown_chair_rendered-500-LR5e-06'  --train_data_dir='images/brown_chair_real-4pics' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-real>'   --initializer_token='chair' --existing_token='<chair-toy>'  --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-06   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0  --output_dir='results-brown_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06' && python inference.py --prompt 'a photo of a <chair-real>' --model_id 'results-brown_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06' --output_image_name 'output_imgs/results-brown_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06.png' && python inference.py --prompt 'a photo of a <chair-toy> in the style of <chair-real>' --model_id 'results-brown_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06' --output_image_name 'output_imgs/results-brown_chair-from_rendered-500-LR5e-06-real-4pics-${val}-LR5e-06-both.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

    # ---


    python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    --job "08-10-infer-black-chair-4pics-${val}-LR5e-06" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'   --train_data_dir='images/black_chair_rendered-4pics' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='chair'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-06   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0   --output_dir='results-black_chair_rendered-4pics-${val}-LR5e-06' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-black_chair_rendered-4pics-${val}-LR5e-06' --output_image_name 'output_imgs/black_chair_rendered-4pics-${val}-LR5e-06.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

    python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    --job "08-10-infer-brown-chair-4pics-${val}-LR5e-06" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'   --train_data_dir='images/brown_chair_rendered-4pics' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='chair'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-06   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0    --output_dir='results-brown_chair_rendered-4pics-${val}-LR5e-06' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-brown_chair_rendered-4pics-${val}-LR5e-06' --output_image_name 'output_imgs/brown_chair_rendered-4pics-${val}-LR5e-06.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

    # python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
    # --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
    # --job "08-10-infer-cat-${val}-LR5e-06" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'   --train_data_dir='cat-one' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='cat'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-06   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0  --output_dir='results-cat-one-${val}-LR5e-06' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-cat-one-${val}-LR5e-06' --output_image_name 'output_imgs/cat-one-${val}-LR5e-06.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

    # ---


done

# vals=(500 1000 2000)
# for val in "${vals[@]}"; do
#     python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
#     --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
#     --job "08-10-infer-black-chair-${val}-model1.5" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'   --train_data_dir='images/black_chair_rendered' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='chair'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-04   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0   --output_dir='results-black_chair_rendered-${val}-model1.5' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-black_chair_rendered-${val}-model1.5' --output_image_name 'output_imgs/black_chair_rendered-${val}-model1.5.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

#     python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
#     --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
#     --job "08-10-infer-brown-chair-${val}-model1.5" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5'   --train_data_dir='images/brown_chair_rendered' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='chair'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=5.0e-04   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0    --output_dir='results-brown_chair_rendered-${val}-model1.5' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-brown_chair_rendered-${val}-model1.5' --output_image_name 'output_imgs/brown_chair_rendered-${val}-model1.5.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

#     # python -m tu.sbatch.sbatch_sweep --time 96:00:00 \
#     # --proj_dir /viscam/projects/image2Blender/diffusers/examples/textual_inversion --conda_env diffusers \
#     # --job "08-10-infer-cat-${val}-LR5e-06" --command "accelerate launch textual_inversion.py   --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'   --train_data_dir='cat-one' --save_as_full_pipeline  --learnable_property='object'   --placeholder_token='<chair-toy>'   --initializer_token='cat'   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=${val}   --learning_rate=1.0e-06   --scale_lr   --lr_scheduler='constant'   --lr_warmup_steps=0  --output_dir='results-cat-one-${val}-LR5e-06' && python inference.py --prompt 'a photo of a <chair-toy>' --model_id 'results-cat-one-${val}-LR5e-06' --output_image_name 'output_imgs/cat-one-${val}-LR5e-06.png'"  $GPU_INFO "$EXTRA_GPU_INFO"

# done