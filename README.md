# SwitchLoRA

Implementation of paper **SwitchLoRA: Switched Low-Rank Adaptation Can Learn Full-Rank Information**



## Enviroment setup

1. Install pytorch following https://pytorch.org/get-started/locally/

2. Clone repository & Install dependencies

   ```shell
   git clone git_url_to_this_repository && cd switchlora
   pip install -r requirements.txt
   pip install flash-attn
   ```

Our code has been test on `Ubuntu 22.04 LTS`. For details on the installed packages, please refer to `requirements.txt.ubuntu`.

## Download data

```shell
cd llama
# Download data and preprocess it to 512 sequence len
python download_data.py --save_dir preprocessed_data_512 --tokenizer t5-base --dataset allenai/c4 --dataset_config en --take 46000000 --text_field text --sequence_length 512
# Download data and preprocess it to 256 sequence len
python download_data.py --save_dir preprocessed_data_256 --tokenizer t5-base --dataset allenai/c4 --dataset_config en --take 15000000 --text_field text --sequence_length 256
```



## Options

Following are options related to the paper of SwitchLoRA. There are some more options in the code for test purpose.

- `use_lora`: Whether to use LoRA adapter. If `switch_lora` is specified, `use_lora` will be set to `true` automatically
- `switch_lora`: Whether to use SwitchLoRA
- `lora_rank`: LoRA rank
- `lr`: Learning rate
- `adam_warm_step`: How many steps to freeze LoRA vectors when their counterpart vectors are switched. Set to $5$ by default
- `switch_lora_interval`: Initial value of switch interval($interval_0$​)
- `switch_lora_descent_rate`: $ratio$ in the SwitchLoRA paper. It determines the point at which the switching frequency is reduced to one-third of its initial value, occurring at the step $total\_step \times ratio$
- `init_lora_type`: Set to `origin_lora` to test initialization method of vanilla LoRA





## Running examples

**Full-rank** training example:

```shell
# Change master_port to prevent port conflict with other running distributed program
# Take about 60GB GPU memory. Shrink batch_size if GPU memory is unenough.
# batch_size means batch size per GPU
torchrun --nproc-per-node 2 --master_port 14214 main.py --model_config configs/llama_350m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_512 --batch_size 72 --total_batch_size 1152 --lr 1e-3 --max_length 512 --num_training_steps 40002 --save_every 2000 --eval_every 1000 --keep_checkpoints 100 --num_workers 8  --save_dir checkpoints/llama_350m_full_512_batch1152_lr0.001_step40000_dp2 --autoresume True
# Or run it in single GPU for debugging with breakpoint
python main.py --model_config configs/llama_350m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_512 --batch_size 72 --total_batch_size 1152 --lr 1e-3 --max_length 512 --num_training_steps 40002 --save_every 2000 --eval_every 1000 --keep_checkpoints 100 --num_workers 8  --save_dir checkpoints/llama_350m_full_512_batch1152_lr0.001_step40000_dp2 --autoresume True
```

**LoRA** example:

```shell
torchrun  --master_port 14230 --nproc-per-node 2 main.py --model_config configs/llama_350m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_512 --batch_size 72 --total_batch_size 1152 --lr 0.01 --max_length 512 --num_training_steps 40002 --save_every 1000 --eval_every 1000 --keep_checkpoints 300 --num_workers 8 --use_lora --lora_rank 128 --lora_dropout 0. --save_dir checkpoints/llama_350m_lora_512_batc1152_lr0.01_step40000_dp2 --autoresume True
```

**SwitchLoRA** example:

```shell
torchrun --master_port 14204 --nproc-per-node 8 main.py --model_config configs/llama_350m.json --dataset_path preprocessed_data/allenai/c4_en_t5-base_512 --batch_size 72 --total_batch_size 1152 --lr 0.02 --max_length 512 --num_training_steps 40002 --save_every 1000 --eval_every 1000 --keep_checkpoints 300 --num_workers 8 --lora_rank 256 --lora_dropout 0.  --switch_lora --switch_lora_descent_rate 0.1 --zero_switch_step_state  --zero_switch_state --save_dir checkpoints/llama_350m_switchlora_512_batch1152_lr0.02_rate0.1_rank256_step40000_dp8 --autoresume True
```