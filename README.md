<div align="center">

# FairCL: Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments

[![Paper](https://img.shields.io/badge/arXiv-2305.15700-brightgreen)](https://arxiv.org/abs/2305.15700)
[![Conference](https://img.shields.io/badge/NeurIPS-2023-blue)](https://arxiv.org/abs/2305.15700)
[![Youtube](https://img.shields.io/badge/Youtube-link-red)](https://www.youtube.com/watch?v=RSw8CJNFk94)

</div>

# Dataset

Two scripts are available to download ADE20K and Pascal-VOC 2012, please see in the `data` folder.

# Training

The implementation of the training of our proposed method will be published in the future.

# Testing

To run testing, use the scripts in the `scripts` folder.
For example, to test the performance of FairCL on the 100-50 setting of ADE20K, do
```
bash scripts/ade/FairCL_ade_100-50_segformer.sh
```

# Acknowledgements
This codebase is borrowed from [PLOP](https://github.com/arthurdouillard/CVPR2021_PLOP) and [RCIL](https://github.com/zhangchbin/RCIL).

# Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{truong2023fairness,
  title={Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments},
  author={Truong, Thanh-Dat and Nguyen, Hoang-Quan and Raj, Bhiksha and Luu, Khoa},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
