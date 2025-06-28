# vllm 0.9.1

# bench_serving.py is in sglang/python/bench_serving

# python -m vllm.entrypoints.openai.api_server --model ~/autodl-tmp/Mistral-7B-Instruct --disable-log-requests
# python -m vllm.entrypoints.openai.api_server --model ~/autodl-tmp/DeepSeek-R1-0528-Qwen3-8B --disable-log-requests

# vllm_offline

python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 256 --random-output 256 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 256 --random-output 1024 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 256 --random-output 4096 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 256 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 1024 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 4096 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000

python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 256 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 1024 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 4096 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000

python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 1024 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 5000 --random-input 1024 --random-output 1024 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000

python3 bench_serving.py --backend vllm --dataset-name sharegpt --num-prompts 1000 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name sharegpt --num-prompts 3000 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name sharegpt --num-prompts 5000 --output-file vllm_offline.jsonl --host 127.0.0.1 --port 8000

# vllm_online

python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 300 --request-rate 1 --output-file vllm_online.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 600 --request-rate 2 --output-file vllm_online.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 1200 --request-rate 4 --output-file vllm_online.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 2400 --request-rate 8 --output-file vllm_online.jsonl --host 127.0.0.1 --port 8000
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 4800 --request-rate 16 --output-file vllm_online.jsonl --host 127.0.0.1 --port 8000
