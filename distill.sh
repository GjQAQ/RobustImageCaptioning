# similar to train_teacher.sh
id="distill"
ckpt_id="res-aoa"
dataset_root="/root/autodl-tmp/datasets/MSCOCO"
nohup python distill.py --id $id \
    --caption_model aoa \
    --refine 1 \
    --refine_aoa 1 \
    --use_ff 0 \
    --decoder_type AoA \
    --use_multi_head 2 \
    --num_heads 8 \
    --multi_head_scale 1 \
    --mean_feats 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --label_smoothing 0.2 \
    --input_json aoanet/data/cocotalk.json \
    --input_label_h5 aoanet/data/cocotalk_label.h5 \
    --input_fc_dir  aoanet/data/cocotalk_fc \
    --input_att_dir  aoanet/data/cocotalk_att  \
    --input_box_dir  aoanet/data/cocobu_box \
    --dataset_root "$dataset_root" \
    --training_device cuda \
    --seq_per_img 5 \
    --batch_size 12 \
    --beam_size 1 \
    --learning_rate 2e-4 \
    --num_layers 2 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log/"log_$id"  \
    --save_checkpoint_every 6000 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 25 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 3 \
    --teacher_checkpoint log/"log_$ckpt_id"/model-best.pth \
    --distilling_temperature 20 \
    --corrupter blur
