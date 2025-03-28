# Accurate Peak Detection in Multimodal Optimization via Approximated Landscape Learning

Here we provide sourcecodes of APDMMO, which is accepted by GECCO 2025.

## Citation

```bash
@article{ma2025accurate,
  title={Accurate Peak Detection in Multimodal Optimization via Approximated Landscape Learning},
  author={Ma, Zeyuan and Lian, Hongqiao and Qiu, Wenjie and Gong, Yue-Jiao},
  journal={arXiv preprint arXiv:2503.18066},
  year={2025}
}
```

## Run the code
The optimization process on CEC2013 multimodal optimization benchmark can be  activated via the command below.
```bash
python run.py
```

## Results
The optimization process includes three stages:

1. Global Landscape Fitting(GLF): 
This stage is used to train a Landscape Learner for a problem. The Log files will be saved to `./log`, and the trained models will be saved to `./checkpoint`, while the information of mu and std for normalization are stored in `./mu_std_info`. The contour maps of 1D/2D problems are saved in `./pic`. 

2. Free-of-trial Peak Detection(FPD): 
This stage is used to detect potential peak areas. The optimized solutions are saved in `./optimization_result`.

3. Parallel Local Search(PLS): 
The last stage is designed to perform local optimization. The results of Peak Ratio(PR) and Success Rate(SR) are saved to `./result`.
