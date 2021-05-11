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

## Reference
If you use this code please cite the original paper:

```
@inproceedings{liu2021filling,
  title={Filling the Gap of Utterance-aware and Speaker-aware Representation for Multi-turn Dialogue},
  author={Liu, Longxiang and Zhang, Zhuosheng and and Zhao, Hai and Zhou, Xi and Zhou, Xiang},
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},
  year={2021}
}
```