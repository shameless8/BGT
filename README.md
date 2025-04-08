## Get Started

### 0.Software Requirements
- Conda 
- C++ Compiler for PyTorch extensions
- CUDA SDK 11 for PyTorch extensions (version 11.8)
- C++ Compiler and CUDA SDK must be compatible


### 1.Setup
Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate bgtriangle
```

Please do **NOT** forget to compile the CUDA kernels:
```shell
pip install submodules/brasterizer/
pip install submodules/diff-Brasterization/
```


### 2.Training

To run the optimizer, simply use
```
python demo.py  -s <path to COLMAP or NeRF Synthetic dataset>  --init_path <path to initial point cloud ply file>  --model_path <path to output folder>  --eval --disable_viewer
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```0.0.0.0``` by default.
  #### --port 
  Port to use for GUI server, ```8080``` by default.
  #### --disable_viewer
  Disable viewer during training
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.

</details>
<br>

For [Masked T&T dataset](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip), please use the provided training script (train_maskTT.sh).


### 3.Testing and Evaluation
You can render training/test sets and produce error metrics as follows:
```shell
python test.py  -s <path to COLMAP or NeRF Synthetic dataset> --model_path <path to trained model> --checkpoint <path to checkpoint file>  --eval --log_blur_radius -9 --mode 2
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

### 4.Viewer
We also provide a viewer for training and testing visualization via browsers.

If you would like to enable the training-time viewer, please do not pass '--disable_viewer' parameters to the training code. Once you have a trained model, you can visualizate it by using this command:
```
python viewer.py  -s <path to COLMAP or NeRF Synthetic dataset> --model_path <path to trained model> --start_checkpoint <path to checkpoint file> --log_blur_radius -9
```

## Initial Point Clouds
We provide the initial [point clouds](initial_pc/nerf_synthetic.zip) for NeRF Synthetic dataset, which are used in training. 


## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.
