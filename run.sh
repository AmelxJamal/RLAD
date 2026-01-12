cd /mnt/lts4/scratch/home/abdelrah/RLAD

# MAX_CONTEXT_TOKENS=4096 MAX_NEW_TOKENS=2048  python test_hf.py


# [
#     {"name":"baseline","hint_model_id":"Qwen/Qwen3-4B","solver_model_id":"Qwen/Qwen3-1.7B"},
#     {"name":"rlad_main","hint_model_id":"CMU-AIRe/RLAD-Hint-Gen","solver_model_id":"CMU-AIRe/RLAD-Sol-Gen"},
#     {"name":"hint_ckpt_00_00","hint_model_id":"CohenQu/Qwen3-4B-Base_HintGen-withSol.00.00","solver_model_id":"CMU-AIRe/RLAD-Sol-Gen"},
#     {"name":"hint_ckpt_00_01","hint_model_id":"CohenQu/Qwen3-4B-Base_HintGen-withSol.00.01","solver_model_id":"CMU-AIRe/RLAD-Sol-Gen"}
#   ]

# python eval_gsmplus.py \
#   --out_dir /mnt/lts4/scratch/home/abdelrah/RLAD/runs/gsmplus \
#   --pipelines_json pipelines.json \
#   --perturbation_types 'critical thinking' \
#   --max_per_type 200 \
#   --resume

# python eval_gsmplus_stats.py 
# python quick_test.py --records /mnt/lts4/scratch/home/abdelrah/RLAD/runs/gsmplus/records.jsonl


# python eval_gsmplus.py \
#   --out_dir /mnt/lts4/scratch/home/abdelrah/RLAD/runs/gsmplus_all \
#   --pipelines_json pipelines.json \
#   --perturbation_types 'all' \
#   --max_per_type 100 \
#   --resume


# python eval_gsmplus.py \
#  --out_dir runs/gsmplus_test_eval \
#  --max_per_type 5 \

# python eval_gsmplus.py --out_dir runs/gsmplus_gpu_batch_size_8 --max_per_type 5 --log_gpu_stats --batch_size 8

# python eval_gsmplus.py --out_dir runs/gsmplus_gpu_batch_size_8_max_16 --max_per_type 16 --log_gpu_stats --batch_size 8

python eval_gsmplus.py \
 --out_dir runs/gsmplus_gpu_batch_size_8_max_200 \
 --max_per_type 200 --log_gpu_stats --batch_size 8