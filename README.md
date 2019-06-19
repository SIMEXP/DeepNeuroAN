## DeepNeuroAN

##### References

Survey : https://arxiv.org/pdf/1903.02026.pdf

rigid brain: https://arxiv.org/pdf/1803.05982.pdf

unsupervised : https://arxiv.org/pdf/1809.06130.pdf

RL : https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14751

##### Input

BIDS compliant dataset 

##### Output

f-MRI normalized in standard MNI space in /derivatives/deepneuroan

##### Examples

## Install

#### Usage

singularity exec -B /home/ltetrel/Documents/data/preventad_prep:/home/jovyan/data /home/ltetrel/Documents/work/DeepNeuroAN/deepneuroan.simg python3 /home/jovyan/DeepNeuroAN/deepneuroan/preproc.py -d /home/jovyan/data -m T1w
