dataset_root="/root/autodl-tmp/datasets/MSCOCO"
ckpt_id="res-aoa"
python eval.py \
	--evaluation_device cuda \
	--dataset_root "$dataset_root" \
	--model log/"log_$ckpt_id"/model-best.pth \
	--infos_path log/"log_$ckpt_id"/"infos_$ckpt_id-best.pkl" \
	--corrupter blur \
	--dump_images 0 \
	--dump_json 1 \
	--num_images -1 \
	--language_eval 1 \
	--beam_size 2 \
	--batch_size 100 \
	--split test
