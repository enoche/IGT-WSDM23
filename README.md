# IGT-WSDM23
Official Pytorch Code for "Inductive Graph Transformer for Delivery Time Estimation", WSDM 2023

## Framework            
![](./images/IGT.png)


## How to run
Enter `IGT-main`:   
```python
python main.py -d toy -m IGT
```

## Configuration
You may configure datasets and models in `configs`.

## Datasets
Under PIPL, we are unable to provide the evaluated datasets for public use. However, we generate a toy dataset (`datasets/toy`) with the same format to demonstrate how IGT works.

## Citation
```
@article{zhou2022inductive,
  title={Inductive Graph Transformer for Delivery Time Estimation},
  author={Zhou, Xin and Wang, Jinglong and Liu, Yong and Wu, Xingyu and Shen, Zhiqi and Leung, Cyril},
  journal={arXiv preprint arXiv:2211.02863},
  year={2022}
}
```

---
No commercial use. License reserved by authors.
