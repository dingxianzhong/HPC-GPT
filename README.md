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
After the training is complete, we load the trained model to evaluate on Dace Race Benchmark.
```shell
python inference.py \
    --base_model path_to_llama_hf_dir \
    --lora_model path_to_lora \
    --with_prompt \
    --interactive
```

Parameter Description:

- `--base_model {base_model}`: Directory to store LLaMA model weights and configuration files in HF format
- `--lora_model {lora_model}`: The directory where the LoRA file is decompressed, and the name of the HF Model Hub model can also be used. If you have incorporated LoRA weights into the pre-trained model, you can not provide this parameter
- `--tokenizer_path {tokenizer_path}`: Store the directory corresponding to the tokenizer. If this parameter is not provided, its default value is the same as --lora_model; if the --lora_model parameter is not provided, its default value is the same as --base_model
- `--with_prompt`: Whether to merge the input with the prompt template. Be sure to enable this option if loading an Alpaca model!
- `--interactive`: start interactively for multiple single rounds of question and answer
- `--data_file {file_name}`: Start in non-interactive mode, read the contents of file_name line by line for prediction
- `--predictions_file {file_name}`: In non-interactive mode, write the predicted results to file_name in json format
- `--use_cpu`: use only CPU for inference
- `--gpus {gpu_ids}`: Specifies the number of GPU devices used, the default is 0. If using multiple GPUs, separate them with commas, such as 0,1,2

### Dataset
5.86k HPC datasets include two tasks for the high-performance computing (HPC) domain.
Task 1 is managing AI models and datasets which includes programming language processing (PLP) and MLPerf.
Task 2 is data race detection which includes c/c++ language and fortran language.
All datasets can be downloaded here (https://huggingface.co/datasets/HPC-GPT/HPC).
