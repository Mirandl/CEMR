# CEMR
Cross Domain Estimation on Human Mesh Reconstruction

## Description
We focus on reconstructing human mesh from Cross-domain videos. In our experiments, we train a source model (termed as *BaseModel*) on Human 3.6M. To produce accurate human mesh on Cross-domain videos, we optimize the BaseModel on target videos via CEMR at test time. 
---
## Get Started

CEMR has been implemented and tested on Ubuntu 18.04 with python = 3.6.

Clone this repo:

```bash
git clone https://github.com/Mirandl/CEMR.git
```

Install required packages:

```bash
conda create -n cemr-env python=3.6
conda activate cemr-env
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -r requirements.txt
install spacepy following https://spacepy.github.io/install_linux.html
```

Download required file from [File 1](https://drive.google.com/file/d/1_4GhHaiNIu2aidVwMBvbdcdGd2vgy-gR/view?usp=sharing) and [File 2](https://drive.google.com/file/d/1uekfFsWnLcKdrT6CxZ9zFQFy_ySdDaXK/view?usp=sharing). After unzipping files, rename `File 1` to `data` (ensuring you do not overwrite `gmm_08.pkl` in `./data`) and move the files in `File 2` to `data/retrieval_res`. Finally, they should look like this:
```
|-- data
|   |--dataset_extras
|   |   |--3dpw_0_0.npz
|   |   |--3dpw_0_1.npz
|   |   |--...
|   |--retrieval_res
|   |   |--...
|   |--smpl
|   |   |--...
|   |--spin_data
|   |   |--gmm_08.pkl
|   |--basemodel.pt
|   |--J_regressor_extra.npy
|   |--J_regressor_h36m.npy
|   |--smpl_mean_params.npz
```

Download Human 3.6M using this [tool](https://github.com/kotaro-inoue/human3.6m_downloader), and then extract images by:
```
python process_data.py --dataset h36m
```
You can process Human3.6M dataset using this [tool](https://github.com/kotaro-inoue/human3.6m_downloader)

---
## Running on the 3DPW
Download the [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) dataset. Then edit `PW3D_ROOT` in the config.py.
Then, run:
```bash
bash run_on_3dpw.sh
```

#### Results on 3DPW

| Method                                                       | Protocol | PA-MPJPE |  MPJPE   |   PVE    |
| :----------------------------------------------------------- | :------: | :------: | :------: | :------: |
| [SPIN](https://github.com/nkolot/SPIN)                       |   #PS    |   59.2   |   96.9   |  135.1   |
| [PARE](https://github.com/mkocabas/PARE)                     |   #PS    |   46.4   |   79.1   |   94.2   |
| [Mesh Graphormer](https://github.com/microsoft/MeshGraphormer) |   #PS    |   45.6   |   74.7   |   87.7   |
| CEMR (Ours)                                               |   #PS    | **40.9** | **65.8** | **82.6** |

<img src="assets/qualitative_res1.png" alt="qualitative results" style="zoom:50%;" />


## Acknowledgement
We borrow codes largely from [VIBE](https://github.com/mkocabas/VIBE) and [DynaBOA](https://github.com/syguan96/DynaBOA).
