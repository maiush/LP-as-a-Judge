./full.sh gemma-2-2b-it common_sense || true
./full.sh gemma-2-9b-it common_sense || true

./full.sh qwen-2.5-0.5b-it common_sense || true
./full.sh qwen-2.5-1.5b-it common_sense || true
./full.sh qwen-2.5-3b-it common_sense || true
./full.sh qwen-2.5-7b-it common_sense || true
./full.sh qwen-2.5-14b-it common_sense || true

./full.sh llama-3.1-8b-it common_sense || true

./full.sh mistral-nemo-12b-it common_sense || true


./full.sh gemma-2-2b-it text_quality || true
./full.sh gemma-2-9b-it text_quality || true

./full.sh qwen-2.5-0.5b-it text_quality || true
./full.sh qwen-2.5-1.5b-it text_quality || true
./full.sh qwen-2.5-3b-it text_quality || true
./full.sh qwen-2.5-7b-it text_quality || true
./full.sh qwen-2.5-14b-it text_quality || true

./full.sh llama-3.1-8b-it text_quality || true

./full.sh mistral-nemo-12b-it text_quality || true