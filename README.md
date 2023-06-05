# GENEA_2023
FineMotion's experiments for GENEA 2023 challenge
## Repo structure
- `PAE.ipynb` - jupyter notebook with experiments step-by-step
- `process_motion.py` - script for processing motion data: extracts features from bvh file and vice versa
- `prepare_motion.py` - script for additional motion data preprocessing: converts positions to joint velocities, 
reshapes ortho6d data to change columns order and vice versa
- `process_audio.py` - extracts audio features
- `make_dataset.py` - combines different features to .npz archives
- `train_*.py` - training one of the models
- `src` - primary package to store source code
  - `blender` - blender scripts for additional data visualisation
  - `mann`, `pae`, `sae` - different models with following structure:
    - `dataset` - basic PyTorch Dataset
    - `model` - PyTorch model
    - `system` - pytoch-lightning module
  - `training` - common utils for training: optimizer, scheduler, loss
  - `utils` - utils for data preprocessing and visualization
  
## Data preprocessing for PhaseAutoEncoder
### Joint velocities
Here steps of data preparation to fit original PAE from
[`AI4Animation`](https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022).
Assume that we have our dataset in `./data` folder  
- First fit and save `sklearn.pipeline` for motion data (we need only one bvh-file) via: 
```
python process_motion.py --mode pipeline --pipeline position --src ./data/trn/bvh/trn_2023_v0_000_main-agent.bvh --pipeline_dir ./pipe_pos
```
- Extract raw joint positions using pipeline trained for `trn` and `val` dataset
```
python process_motion.py --mode bvh2npy --pipeline ./pipe_pos --src ./data/trn/bvh --dst ./data/trn/positions
python process_motion.py --mode bvh2npy --pipeline ./pipe_pos --src ./data/val/bvh --dst ./data/val/positions
```
- Calculate velocities with
```
python prepare_motion.py --mode joint_velocities --src ./data/trn/positions --dst ./data/trn/velocities 
python prepare_motion.py --mode joint_velocities --src ./data/trn/positions --dst ./data/trn/velocities 
```
### Ortho6D
We also tried training PAE on [ortho6d](https://arxiv.org/pdf/1812.07035.pdf) joint rotations. 
To prepare data use similar pipeline.
- Train data pipeline:
```
python process_motion.py --mode pipeline --pipeline ortho6d --src ./data/trn/bvh/trn_2023_v0_000_main-agent.bvh --pipeline_dir ./pipe_ortho6d
```
- Extract features: Joint ortho6d angles with root position
```
python process_motion.py --mode bvh2npy --pipeline ./pipe_ortho6d --src ./data/trn/bvh --dst ./data/trn/ortho6d
python process_motion.py --mode bvh2npy --pipeline ./pipe_ortho6d --src ./data/val/bvh --dst ./data/val/ortho6d
```
- Convert data (recombine columns for angles to make them ordered by joint and calculate root velocity:
```
python prepare_motion.py --mode ortho6d --src ./data/trn/ortho6d --dst ./data/trn/ortho6d_fixed 
python prepare_motion.py --mode ortho6d --src ./data/val/ortho6d --dst ./data/val/ortho6d_fixed 
```
## Training PAE
To train PhaseAutoEncoder on joint velocities use following script:
```
python train_pae.py --serialize_dir ./results/pae --trn_folder ./data/trn/velocities --val_folder ./data/trn/velocities \
    --accelerator gpu --batch_size 512
```
Some parameters could also be changed, like accelerator or batch size (note: FFT doesn't work on mps). 
Model options could also be changed: list of options could be found in `src/pae/system:PAESystem.add_system_args`

To train PAE with ortho6d representation you should change some parameters:
```
python train_pae.py --serialize_dir ./results/pae_ortho6d --trn_folder data/trn/ortho6d_fixed \
 --val_folder data/val/ortho6d_fixed --batch_size 512 --accelerator gpu --joints 25 --channels 6 --phases 5 --add_root
```
Here we have 25 joints (instead of 26) because root is represented by velocity and other joints - by 6D rotation

Some examples of using these models could be found in `PAE.ipynb`. 

## Preparing data for main model
As main model we use some modification of Mixture-of-Experts (or ModeAdaptiveNeuralNetwork) 
framework like in original DeepPhase paper. 
To train this model we need to:
- Extract Phases via PAE
- Prepare motion features
- Prepare audio features
- Combine features together

### Extract Phases
For extract phases via PAE use following script:
```commandline
python .\extract_phases.py --src ./data/trn/velocities --dst ./data/trn/phases --checkpoint ./results/pae/last.ckpt
python .\extract_phases.py --src ./data/val/velocities --dst ./data/val/phases --checkpoint ./results/pae/last.ckpt
```

### Extract Audio Features
To extract MFCC features with specified fps use script
```commandline
python .\process_audio.py --src .\data\trn\main-agent\wav\ --dst data\trn\mfcc
python .\process_audio.py --src .\data\val\main-agent\wav\ --dst data\val\mfcc
```

### Preparing motion features
First, we try to use only joint rotations (without root velocity) to represent motion data extracting when training PAE of rotations:
```commandline
python .\prepare_motion.py --mode ortho6d --ignore_root --src ./data/trn/ortho6d --dst ./data/trn/ortho6d_fixed
python .\prepare_motion.py --mode ortho6d --ignore_root --src ./data/trn/ortho6d --dst ./data/trn/ortho6d_fixed
```

### Making dataset
Combine data
```commandline
python .\make_dataset.py --motion .\data\trn\ortho6d_fixed\ --audio .\data\trn\mfcc --phase .\data\trn\phases --dst .\data\trn\dataset
python .\make_dataset.py --motion .\data\val\ortho6d_fixed\ --audio .\data\val\mfcc --phase .\data\val\phases --dst .\data\val\dataset
```
## Training MoE
to be continued...
