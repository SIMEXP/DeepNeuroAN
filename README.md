## DeepNeuroAN

##### References

Survey : https://arxiv.org/pdf/1903.02026.pdf

rigid brain: https://arxiv.org/pdf/1803.05982.pdf

unsupervised : https://arxiv.org/pdf/1809.06130.pdf

RL : https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14751

Importance of big batch size : https://arxiv.org/pdf/1803.08450.pdf

Segmentation : https://arxiv.org/pdf/1909.05085.pdf
https://github.com/neuronets/nobrainer

conv + maxpool VS strided conv : https://arxiv.org/pdf/1412.6806.pdf

bayesian inference : https://arxiv.org/pdf/1904.11319.pdf

patch-wise non rigid registration : https://arxiv.org/pdf/1607.02504.pdf

important reference for quaternion registration : http://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf

quaternion sampling : http://refbase.cvc.uab.es/files/PIE2012.pdf

non-rigid registration : https://arxiv.org/pdf/1809.05231.pdf

unsupervised registration: https://arxiv.org/abs/2004.04617v1

multi-modality registration : https://arxiv.org/pdf/2004.10282v1.pdf

deep-learning metric fo intra-modality registration : https://arxiv.org/pdf/1804.10735.pdf

https://github.com/FrancoisPgm/twolevel_ants_dbm

##### Input

BIDS compliant dataset 

##### Output

f-MRI normalized in standard MNI space in /derivatives/deepneuroan

##### Examples

## Install

#### Usage

###### Data generation
```
singularity exec -B /scratch/ltetrel/neuromod/:/DATA /data/cisl/CONTAINERS/deepneuroan.simg python3 /DeepNeuroAN/deepneuroan/generate_train_data.py -d /DATA -r 160 -n 10  -s 0
```

###### Training
```
singularity exec -B /scratch/ltetrel/neuromod/:/DATA /data/CONTAINERS/deepneuroan.simg python3 /scripts/train.py -d /DATA/derivatives/deepneuroan/training/generated_data/ --batch_size 32 --lr 0.05 --dropout 0 --encode_layers 5 --strides 2 2 2  --seed 0
```
