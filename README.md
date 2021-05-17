Official code for the paper: [Growing 3D Artefacts and Functional Machines with Neural Cellular Automata](https://arxiv.org/abs/2103.08737)
==================


Requirements
----
- python3.8
- This package automatically install `test-evocraft-py`, but for further functionality follow installation steps here: https://github.com/real-itu/Evocraft-py

Installation
---------------
### For general installation
```
python setup.py install
```
### For ray tune + mlflow
```
python -m pip install -r ray-requirements.txt
python setup.py install
```

Usage
-------------
Make sure an evocraft-py server is running, either with `test-evocraft-py --interactive` or by following the steps in https://github.com/real-itu/Evocraft-py.

### Configs
Each nca is trained on a specific structure w/ hyperparams and configurations defined in yaml config, which we use with [hydra](https://github.com/facebookresearch/hydra) to create the [NCA trainer class](artefact_nca/trainer/voxel_ca_trainer.py).

[Example Config](pretrained_models/PlainBlacksmith/plain_blacksmith.yaml) for generating a "Jungle Temple" Minecraft Structure:
```
trainer:
    name: PlainBlacksmith
    min_steps: 48
    max_steps: 64
    visualize_output: true
    device_id: 0
    use_cuda: true
    num_hidden_channels: 10
    epochs: 20000
    batch_size: 5
    model_config:
        normal_std: 0.1
        update_net_channel_dims: [32, 32]
    optimizer_config:
        lr: 0.002
    dataset_config:
        nbt_path: artefact_nca/data/structs_dataset/nbts/village/plain_village_blacksmith.nbt

defaults:
  - voxel
```


### 1. Interactive
See [generation notebook](notebooks/GenerateStructures.ipynb) for ways to load in a pretrained nca and generate a structure

See [training notebook](notebooks/Training.ipynb) for ways to train an nca



Authors
-------
Shyam Sudhakaran <shyamsnair@protonmail.com> <https://github.com/shyamsn97>

Djordje Grbic <djgr@itu.dk>, <https://github.com/djole>

Siyan Li <lisiyansylvia@gmail.com> <https://github.com/sli613>

Adam Katona <ak1774@york.ac.uk> <https://github.com/adam-katona>

Elias Najarro <https://github.com/enajx>

Claire Glanois <https://github.com/claireaoi>
 
Sebastian Risi <sebr@itu.dk> <https://github.com/sebastianrisi>
