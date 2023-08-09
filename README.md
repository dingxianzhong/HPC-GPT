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
#### Description of training parameters
1. If you want to train on a single card, you only need to set nproc_per_node to 1, or remove the torchrun command and run the python script directly, such as `python run_supervised_finetuning.py`.
2. The default pre-training model is LLaMA. Change `model_name_or_path` to use other pre-training model such as `meta-llama/Llama-2-7b-chat-hf`.
3. Specify the training set, `--train_file_dir` specify the training data directory, and `--validation_file_dir` specify the verification data directory. If not specified, the `--dataset_name` specified HF datasets dataset will be used by default.
4. If the gpu supports int8, plus `--load_in_8bit` Truethe representative adopts 8-bit quantization training, it can significantly reduce memory usage.
5. Debug the model, `--max_train_samples` and `--max_eval_samples` specify the maximum number of samples for the training and validation datasets to quickly verify whether the code is available. Please delete these two parameters or set them to -1 during training.

#### LoRA Training
By default, LoRA training is used. The LoRA model weights of each stage need to be merged into the base model. Use the following command to merge, and the next stage is model_name_or_path designated as the merged model folder.

LoRA layers were used at all stages to reduce memory requirements. At each stage the peft adapter layers were merged with the base model, using:
```shell
python merge_peft_adapter.py \
  --base_model_name_or_path base_model_dir \
  --peft_model_path lora_model_dir \
  --output_dir outputs-merged
```
- This script requires peft>=0.3.0
- The merged weights are saved in the output_dir directory, and can be directly loaded later by from_pretrained

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

- `--base_model {base_model}`: Directory to store LLaMA model weights and configuration files in HF format.
- `--lora_model {lora_model}`: The directory where the LoRA file is decompressed, and the name of the HF Model Hub model can also be used. If you have incorporated LoRA weights into the pre-trained model, you can not provide this parameter.
- `--tokenizer_path {tokenizer_path}`: Store the directory corresponding to the tokenizer. If this parameter is not provided, its default value is the same as --lora_model; if the --lora_model parameter is not provided, its default value is the same as --base_model.
- `--with_prompt`: Whether to merge the input with the prompt template. Be sure to enable this option if loading an Alpaca model.
- `--data_file {file_name}`: Start in non-interactive mode, read the contents of file_name line by line for prediction.
- `--predictions_file {file_name}`: In non-interactive mode, write the predicted results to file_name in JSON format.
- `--use_cpu`: use only CPU for inference
- `--gpus {gpu_ids}`: Specifies the number of GPU devices used, the default is 0. If using multiple GPUs, separate them with commas, such as 0,1,2.

### Dataset
5.86k HPC datasets include two tasks for the high-performance computing (HPC) domain.
Task 1 is managing AI models and datasets which includes programming language processing (PLP) and MLPerf.
Task 2 is data race detection which includes c/c++ language and fortran language.
All datasets can be downloaded here (https://huggingface.co/datasets/HPC-GPT/HPC).



