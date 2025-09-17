# 🎬 Vid2Sim 🤖: Realistic and Interactive Simulation from Video for Urban Navigation
> [Ziyang Xie](https://ziyangxie.site/), [Zhizheng Liu](https://scholar.google.com/citations?user=Asc7j9oAAAAJ&hl=en), [Zhenghao Peng](https://pengzhenghao.github.io/), [Wayne Wu](https://wywu.github.io/), [Bolei Zhou](https://boleizhou.github.io/)
>
> [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2501.06693)
> [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://metadriverse.github.io/vid2sim/)

Vid2Sim is a novel framework that converts monocular videos into photorealistic and physically interactive simulation environments for training embodied agents with minimal sim-to-real gap.

> [!Note]
> This subfolder is used for reconstructing the simulation environment from the video.  
> Please follow the instructions below to prepare the data and run the simulation environment reconstruction.   
>   
> *For RL training, please refer to the [Vid2Sim-RL](https://github.com/Vid2Sim/Vid2Sim/tree/main/src/vid2sim_rl) subfolder.*


<p align="center">
  <img src="../../assets/teaser.png" width="100%">
</p>

## 1. Data Preparation 📂

### Step1: Generate dynamic masks from the video
Follow the instructions in the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) or [DEVA (modified)](https://github.com/ZiYang-xie/vid2sim-deva-segmentation) repository to install the segmentation model.

Convert your video into image sequences and save them in a folder `images`.
```
seq_path/
├── images/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── ...
```
We provide the raw [videos](https://drive.google.com/drive/folders/1jGmKxZL6hKvjCg6qhM9wmW1_HjMwCUGa?usp=sharing) of 30 scenes reported in our paper for you to look over.

Then, run the following command to generate dynamic masks.
```bash
cd tools/
bash generate_mask.sh $seq_path
```
Dynamic masks will be used to mask out foreground dynamic objects within the orginal video and help us only reconstruct a clean objectless static background as our simulation environment. The final dynamic masks will be saved in the folder `masks` under the sequence path.  

We provide two models for dynamic object segmentation, we recommend using the [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) model for better performance. In the original paper, we use a slightly modified DEVA model for referred dynamic object segmentation.

### Step2: Run SfM to reconstruct the camera poses and point cloud prior

First, install [COLMAP](https://colmap.github.io/) and [GLOMAP](https://github.com/metadriverse/Glomap) for SfM.
```bash
conda install -c conda-forge cudatoolkit colmap glomap
```

Then run the following command to run SfM.
```bash
bash run_sfm.sh $seq_path
```
The SfM results will be saved in the folder `sparse` under the sequence path.

### Step3: Generate depth prior for the scene
```bash
bash generate_depth.sh $seq_path
```
We use monocular depth estimation to generate depth prior for the scene. These extra geometry clues will help us reconstruct the scene more accurately. The reconstructed mesh will further serve as the foundation for agent-scene interaction and collision detection in the following RL training stage. The depth results will be saved in the depth folder `depths` under the sequence path.

## 2. Reconstruct the Scene 🗺️

Once the video is processed with dynamic masks, camera poses, and depth priors, you can reconstruct the scene. Run the training script with the prepared data to generate a photorealistic and static 3D scene, which will serve as the base for building interactive simulation environments.

```bash
python train.py -s <SEQ_PATH> -m <RESULT_PATH>
```

## 3. Export the Scene 📦

After reconstruction, export the scene mesh for agent interaction and collision detection. By default, the exported mesh excludes the floor. You can optionally apply `use_ground_mask` and `use_sky_mask` to filter out specific regions using custom masks.

```bash
python export_mesh.py -m <RESULT_PATH> --voxel-size <TSDF_VOXEL_SIZE>
```

The exported mesh should locate in `mesh` folder under the result path.

## 4. Build the Simulation Environment 🌏
Once you have the GS-reconstructed scene and its corresponding mesh both in `.ply` format, you can build a simulation environment using the hybrid scene representation.  

Refer to [BUILD_YOUR_OWN_ENV.md](../vid2sim_rl/BUILD_YOUR_OWN_ENV.md) for a quick guide on setting up your own real-to-sim environment from scratch. Or you could download the provided environments ([Vid2Sim-Envs](https://drive.google.com/drive/folders/1LCruqb6M3mCgsjaqI1ON6WVoZ-9CmQDY?usp=sharing)) and put them in `envs/` folder.

Now you can turn to [Vid2Sim-RL](https://github.com/Vid2Sim/Vid2Sim/tree/main/src/vid2sim_rl) to train the agent in the simulation environment.

## Citation 📝

If you find this work useful in your research, please consider citing:

```bibtex
@article{xie2024vid2sim,
  title={Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation},
  author={Xie, Ziyang and Liu, Zhizheng and Peng, Zhenghao and Wu, Wayne and Zhou, Bolei},
  journal={CVPR},
  year={2025}
}
```
