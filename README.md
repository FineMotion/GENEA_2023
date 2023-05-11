# GENEA_2023
FineMotion's experiments for GENEA 2023 challenge

## Data preprocessing
Here steps of data preparation to fit models  
- First fit and save `sklearn.pipeline` for motion data via: 
```
process_motion.py --mode pipeline --pipeline position --src ./data/trn/bvh --pipeline_dir ./pipe
```
- Extract raw joint positions using pipeline trained
```
process_motion.py --mode bvh2npy --src ./data/trn/bvh --dst ./data/trn/positions
```
- Calculate velocities with