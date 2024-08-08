# Transformer for BCI


## Description
This is the source code for the Transformer-based BCI competition 2005, targeted at Dataset V.
It's for research project in NCSU GEAR 2024. 
This is my first time using Github to manage a full project, any suggestions and questions on the project are welcome.

## Requirements
- Python 3.7
- Pytorch 1.7.1
- Transformers 4.6.1

## Usage
1. Download the dataset from the BCI III Competition website in .mat form.  
2. Run CV1.py and CV2.py to get the results of cross-validation.  

## Citations
This project used some code of the THP model, special thanks to the following paper:
```
@article{zuo2020transformer,
  title={Transformer Hawkes Process},
  author={Zuo, Simiao and Jiang, Haoming and Li, Zichong and Zhao, Tuo and Zha, Hongyuan},
  journal={arXiv preprint arXiv:2002.09291},
  year={2020}
}
```