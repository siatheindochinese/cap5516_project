# CAP5516 Project: 2D Gaussian Splatting for Medical Image Super Resolution

## 1) Requirements
Install all relevant packages using `pip install -r requirements.txt`. This repository is tested on `python==3.10`. The use of a conda environment is highly recommended.

Download the BRATS19 Brain Tumour dataset here: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2 and extract `Task01_BrainTumour/` inside this repostory.

## 2) Single Slice Rendering and Visualization
`cugs.ipynb` contains an example of a medical image slice, trained using 2D gaussians, along with code to compute PSNR and SSIM

## 3) Automated PSNR and SSIM computation.
`x1.sh`, `x2.sh` and `x4.sh` will automatically run 2D reconstruction at the same scale, half-scale and quarter-scale and rendering at native resolution (240x240), and the PSNR & SSIM results collected and stored in output json files `240.json`, `120.json` and `60.json` respectively. These 3 jsons are included in this repository for convenience.

Example: `bash x1.sh`

After obtaining all 3 jsons, run `python averages.py` to obtain the average PSNR and SSIM for all scales.

## Acknowlegdements
2D gaussian rendering functions and configurations are ripped from https://github.com/OutofAi/2D-Gaussian-Splatting.