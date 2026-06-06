base_dir="/home/law/HDD/i_zzy/20260422_IML/Protocol-MVSS-0531"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
/home/law/HDD/i_zzy/20260422_IML/MOF-Net/train.py \
    --model MOFNet \
    --world_size 2 \
    --batch_size 16 \
    --data_path /home/law/HDD/i_zzy/20260422_IML/MOF-Net/balanced_dataset.json \
    --epochs 200 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --test_data_path "/home/law/HDD/i_zzy/Sota_code/Method2/test_datasets.json" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 8 \
    --seed 42 \
    --test_period 3 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log