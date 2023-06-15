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
python process_motion.py --mode bvh2npy --pipeline position --pipeline_dir ./pipe_pos --src ./data/trn/bvh --dst ./data/trn/positions
python process_motion.py --mode bvh2npy --pipeline position --pipeline_dir ./pipe_pos --src ./data/val/bvh --dst ./data/val/positions
```
- Calculate velocities with
```
python prepare_motion.py --mode velocities --src ./data/trn/positions --dst ./data/trn/velocities 
python prepare_motion.py --mode velocities --src ./data/trn/positions --dst ./data/trn/velocities 
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
python process_motion.py --mode bvh2npy --pipeline ortho6d --pipeline_dir ./pipe_ortho6d --src ./data/trn/bvh --dst ./data/trn/ortho6d
python process_motion.py --mode bvh2npy --pipeline ortho6d --pipeline_dir ./pipe_ortho6d --src ./data/val/bvh --dst ./data/val/ortho6d
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
python extract_phases.py --src ./data/trn/velocities --dst ./data/trn/phases --checkpoint ./results/pae/last.ckpt
python extract_phases.py --src ./data/val/velocities --dst ./data/val/phases --checkpoint ./results/pae/last.ckpt
```

### Extract Audio Features
To extract MFCC features with specified fps use script
```commandline
python process_audio.py --src .\data\trn\main-agent\wav\ --dst data\trn\mfcc
python process_audio.py --src .\data\val\main-agent\wav\ --dst data\val\mfcc
```

### Preparing motion features
First, we try to use only joint rotations (without root velocity) to represent motion data extracting when training PAE of rotations:
```commandline
python prepare_motion.py --mode ortho6d --ignore_root --src ./data/trn/ortho6d --dst ./data/trn/ortho6d_fixed
python prepare_motion.py --mode ortho6d --ignore_root --src ./data/val/ortho6d --dst ./data/val/ortho6d_fixed
```

### Making dataset
Combine data
```commandline
python make_dataset.py --motion .\data\trn\ortho6d_fixed\ --audio .\data\trn\mfcc --phase .\data\trn\phases --dst .\data\trn\dataset
python make_dataset.py --motion .\data\val\ortho6d_fixed\ --audio .\data\val\mfcc --phase .\data\val\phases --dst .\data\val\dataset
```
## Training MoE
To train main model you could use following script:
```commandline
python train_moe.py --serialize_dir .\result --trn_folder .\data\trn\dataset --val_folder .\data\val\dataset --num_workers 4 --accelerator gpu --batch_size 512
```

## Generation
To generate resulted motion you could use following steps
- Run inference script
```commandline
 python .\moe_inference.py --src .\data\dataset_gloved\val\val_2023_v0_002_main-agent.npz --dst .\result\pred\ --checkpoint .\result\last.ckpt --alpha 0.5 
```
- Reorganize data
```commandline
 python python .\prepare_motion.py --src .\result\pred --dst .\result\fixed --mode ortho6d_inverse
```
- Generate BVH from numpy
```commandline
 python .\process_motion.py --mode npy2bvh --pipeline_dir .\pipe_ortho6d --src .\result\fixed --dst .\result\bvh
```

## Additional processing
There are also some additional data processing like GloVe embedding and data normalization
For example, we want to normalize phases, add positions and position velocities to pose and GloVe embeddings:
- extract text embeddings:
```commandline
python process_text.py --src data\trn\main-agent\tsv --dst data\trn\gloved --embeddings data\glove\glove.6B.50d.txt --embedding_dim 50
python process_text.py --src data\val\main-agent\tsv --dst data\val\gloved --embeddings data\glove\glove.6B.50d.txt --embedding_dim 50
```

- Convert Ortho6d data to appropriate without root
```commandline
python prepare_motion.py --mode ortho6d --src ./data/trn/ortho6d --dst ./data/trn/ortho6d_noroot --ignore_root
python prepare_motion.py --mode ortho6d --src ./data/val/ortho6d --dst ./data/val/ortho6d_noroot --ignore_root
```

- Get phase velocities
```commandline
python prepare_motion.py --mode velocities --src ./data/trn/phases --dst ./data/trn/phase_velocities
python prepare_motion.py --mode velocities --src ./data/val/phases --dst ./data/val/phase_velocities
```

- normalize data:
```commandline
pyton normalize.py --src data\trn\positions --dst data\trn\positions_normalized --norm data\positions_norm.npz 
pyton normalize.py --src data\val\positions --dst data\val\positions_normalized --norm data\positions_norm.npz

pyton normalize.py --src data\trn\velocities --dst data\trn\velocities_normalized --norm data\velocities_norm.npz 
pyton normalize.py --src data\val\velocities --dst data\val\velocities_normalized --norm data\velocities_norm.npz

pyton normalize.py --src data\trn\ortho6d_noroot --dst data\trn\ortho6d_normalized --norm data\ortho6d_norm.npz 
pyton normalize.py --src data\val\ortho6d_noroot --dst data\val\ortho6d_normalized --norm data\ortho6d_norm.npz

pyton normalize.py --src data\trn\phases --dst data\trn\phases_normalized --norm data\phases_norm.npz 
pyton normalize.py --src data\val\phases --dst data\val\phases_normalized --norm data\phases_norm.npz

pyton normalize.py --src data\trn\phase_velocities --dst data\trn\phase_velocities_normalized --norm data\phase_vel_norm.npz 
pyton normalize.py --src data\val\phase_velocities --dst data\val\phase_velocities_normalized --norm data\phase_vel_norm.npz
```

- make datasets
```commandline
python make_dataset.py --motion data\trn\ortho6d_normalized data\trn\positions_normalized data\trn\velocities_normalized
                       --audio data\trn\mfcc data\trn\gloved
                       --phase data\trn\phases_normalized data\trn\phase_velocities_normalized
                       --dst data\trn\dataset_new
                       
python make_dataset.py --motion data\vel\ortho6d_normalized data\vel\positions_normalized data\vel\velocities_normalized
                       --audio data\vel\mfcc data\trn\gloved
                       --phase data\vel\phases_normalized data\vel\phase_velocities_normalized          
                       --dst data\val\dataset_new             
```

- Train model (flag --vel_included for phase velocities included in dataset)
```commandline
 python .\train_moe.py --serialize_dir .\result_new --trn_folder data\trn\dataset_new --val_folder data\val\dataset_new --num_workers 4 --accelerator gpu --vel_included --batch_size 512 --force
```

### Inference
```commandline
 python .\moe_inference.py --src .data\val\dataset_new\val_2023_v0_000_main-agent.npz --dst .\result_new\pred\ --checkpoint .\result_new\last.ckpt --alpha 0.5 --phase_norm data\phases_norm.npz --vel_norm data\phase_vel_norm.npz --vel_included
 python .\normalize.py --src .\result_new\pred\ --dst .\result_new\renorm\ --norm .\data\ortho6d_norm.npz --mode backward
 python .\prepare_motion.py --src .\result_new\renorm\ --dst .\result_new\fixed\ --mode ortho6d_inverse --ignore_root
 python .\process_motion.py --mode npy2bvh --pipeline_dir .\data\ortho6d\ --src .\result_new\fixed\ --dst .\result_new\bvh\
```

### Inference from test data
- Make dataset
```commandline
python .\process_audio.py --src .\data\tst\main-agent\wav\ --dst data\tst\mfcc
python .\process_text.py --src data\tst\main-agent\tsv --dst data\tst\gloved --embeddings data\glove\glove.6B.50d.txt --embedding_dim 50

python make_dataset.py --audio data\tst\mfcc data\trn\gloved --dst data\tst\dataset_new
```

- Inference (also provide train sample to initialize shapes)
```commandline
python .\moe_inference.py --src .\data\tst\dataset_new\tst_2023_v0_002_main-agent.npz --dst .\result_new\pred\ --checkpoint .\result_new\last.ckpt --alpha 0.5 --phase_norm data\phases_norm.npz --vel_norm data\phase_vel_norm.npz --trn_sample .\data\trn\dataset_new\trn_2023_v0_000_main-agent.npz --vel_included
```
Other steps like in validation
