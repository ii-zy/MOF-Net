base_dir="/home/law/HDD/i_zzy/20260606/MOF-Net"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
/home/law/HDD/i_zzy/20260422_IML/MOF-Net/test_save_images.py \
    --model MOFNet \
    --world_size 4 \
    --test_data_path "/home/law/HDD/i_zzy/datasets/DSO_resized/Train" \
    --checkpoint_path "/home/law/HDD/i_zzy/MOF-Net/Protocol-MVSS-weight.pth" \
    --test_batch_size 2 \
    --image_size 224 \
    --no_model_eval \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log