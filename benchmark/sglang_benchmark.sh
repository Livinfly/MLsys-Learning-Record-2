# sglang 0.4.7.post1

# python -m sglang.launch_server --model-path ~/autodl-tmp/Mistral-7B-Instruct --enable-torch-compile --disable-radix-cache
# python -m sglang.launch_server --model-path ~/autodl-tmp/Mistral-7B-Instruct --enable-torch-compile

# python -m sglang.launch_server --model-path ~/autodl-tmp/DeepSeek-R1-0528-Qwen3-8B --enable-torch-compile --disable-radix-cache
# python -m sglang.launch_server --model-path ~/autodl-tmp/DeepSeek-R1-0528-Qwen3-8B --enable-torch-compile

# sglang_offline

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 256 --random-output 256 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 256 --random-output 1024 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 256 --random-output 4096 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 256 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 1024 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 4096 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 256 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 1024 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 4096 --output-file sglang_offline.jsonl --host 127.0.0.1

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 1024 --random-output 1024 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 5000 --random-input 1024 --random-output 1024 --output-file sglang_offline.jsonl --host 127.0.0.1

python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 1000 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 3000 --output-file sglang_offline.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 5000 --output-file sglang_offline.jsonl --host 127.0.0.1

# sglang_online

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 300 --request-rate 1 --output-file sglang_online.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 600 --request-rate 2 --output-file sglang_online.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 1200 --request-rate 4 --output-file sglang_online.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 2400 --request-rate 8 --output-file sglang_online.jsonl --host 127.0.0.1
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 4800 --request-rate 16 --output-file sglang_online.jsonl --host 127.0.0.1
