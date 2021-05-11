Codes in this repo are based on the paper [Filling the Gap of Utterance-aware and Speaker-aware Representation for Multi-turn Dialogue](https://arxiv.org/pdf/2009.06504.pdf) and the accompanying github https://github.com/comprehensiveMap/MDFN

## Instruction
Our code is compatible with compatible with python 3.x so for all commands listed below python is `python3`.

We strongly suggest you to use `conda` to control the virtual environment.

- Install requirements

``
pip install -r requirements.txt
``

- To run the Electra Basline 

```
python run_MDFN.py \
--data_dir datasets/mutual \
--model_name_or_path \
google/electra-large-discriminator \
--model_type electra \
--baseline \
--task_name mutual\
--output_dir output_mutual_electra \
--cache_dir cached_models \
```

- To run the MDFN Baseline

```
python run_MDFN.py \
--data_dir datasets/mutual \
--model_name_or_path \
google/electra-large-discriminator \
--model_type electra \
--task_name mutual\
--output_dir output_mutual_electra \
--cache_dir cached_models \
```

- To run the Speaker-Aware Embedding
```
python main.py \
--data_dir datasets/mutual \
--model_name_or_path google/electra-large-discriminator \
--model_type electra \
--task_name mutual \
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 2 --eval_batch_size 2 \
--learning_rate 4e-6 --num_train_epochs 6\
--gradient_accumulation_steps 1 --local_rank -1
```

- To run the Speaker-Aware Decouple
```
python main.py \
--baseline\
--speaker_aware\
--data_dir datasets/mutual \
--model_name_or_path google/electra-large-discriminator \
--model_type electra \
--task_name mutual \
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 2 --eval_batch_size 2 \
--learning_rate 4e-6 --num_train_epochs 6\
--gradient_accumulation_steps 1 --local_rank -1
```
- To run the Response-Aware Query
```
python main.py \
--response_aware\
--data_dir datasets/mutual \
--model_name_or_path google/electra-large-discriminator \
--model_type electra \
--task_name mutual \
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 2 --eval_batch_size 2 \
--learning_rate 4e-6 --num_train_epochs 6\
--gradient_accumulation_steps 1 --local_rank -1
```
- To run the Response-Aware BiDAF
```
python main.py \
--BiDAF\
--data_dir datasets/mutual \
--model_name_or_path google/electra-large-discriminator \
--model_type electra \
--task_name mutual \
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 2 --eval_batch_size 2 \
--learning_rate 4e-6 --num_train_epochs 6\
--gradient_accumulation_steps 1 --local_rank -1
```

