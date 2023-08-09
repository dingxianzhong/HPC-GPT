## HPC-GPT : Integrating Large Language Model for High-Performance Computing
### Introduction
### Install
```markdown
https://github.com/MLG-HPCE2023/HPC-GPT.git
cd HPC-GPT
pip install -r requirements.txt --upgrade
```

### Supervised FineTuning
```shell
sh run_sft.sh
```

### Inference
```shell
python inference.py \
    --base_model path_to_llama_hf_dir \
    --lora_model path_to_lora \
    --with_prompt \
    --interactive
```
