
# Predicting FREEWAY INCIDENT CLEARANCE TIME from chinese text reports: A comparison of BERT and LLMs

This repository is the official implementation of [Predicting FREEWAY INCIDENT CLEARANCE TIME from chinese text reports: A comparison of BERT and LLMs](https://openreview.net/forum?id=hiGT14xrSa&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dntu.edu.tw%2FNational_Taiwan_University%2FFall_2025%2FML-MiniConf%2FAuthors%23your-submissions)). 


## Requirements

To install requirements:

```setup
git clone https://github.com/51RED/ML2025-Final_text_time_prediction.git

cd ML2025-Final_text_time_prediction

conda env create -f environment.yml

conda activate ML_env
```

>📋  Follow the command, we used conda to construct related environment.

## Training

To train the model(s) in the paper, run this command for regression or generative task:

```train
python3 train.py -m generative

python3 train.py -m regression
```

>📋  Before trainning, make sure your data exist in "data" file. After trainning, model checkpoints will storage in file "checkpoints". The pre-trained model will be upload in cloud.

## Evaluation

To evaluate my model on regression, run:

- generative
    ```eval
    python3 eval_generative.py -m generative
    ```
- regression
    ```eval
    python3 eval_regression.py -m regression
    ```
>📋  After trainning, there will be evaluation data, which split from training input data in "test data" and model checkpoint in "checkpoints" file. You need to make sure the data and model exist to run the evaluation.

## Pre-trained Models

You can download pretrained models here, and you should put into "checkpoints" file:

- [Fine-tuned Generative model(Llama-3.2-3B-Instruct)](https://drive.google.com/mymodel.pth) trained on Llama using LoRA to finetuned. 


- [Fine-tuned Regression model(Llama-3.2-3B-Instruct)](https://drive.google.com/mymodel.pth) trained on Llama using LoRA to finetuned. 

## Results

Our model achieves the following performance on :


| BERT-based models | ＭAE(min)| RSME(min)   | MAPE(%) |
| -----|---------------- | --------- |-----------|
| BERT+RF  |17.40 | 28.05 |50.72 |
| BERT+MLP |16.89 | 27.49 |46.53 |
| BERT+GBM |**16.20**|**25.62**|**46.28**|

| LLM(Llama-3.2-3B-Instruct)-Generative | ＭAE(min)| RSME(min)   | MAPE(%) |
| -----|---------------- | --------- |-----------|
| Zero-shot  |48.17 | 61.84 |181.15 |
| Few-shot |82.88 | 86.97 |383.22 |
| Fine-tuned Gen |35.28|69.77|81.02|

| LLM(Llama-3.2-Taiwan-3B-Instruct)-Generative | ＭAE(min)| RSME(min)   | MAPE(%) |
| -----|---------------- | --------- |-----------|
| Zero-shot  |21.86 | 36.58 |68.67 |
| Few-shot |65.56 | 271.82 |204.35 |
| Fine-tuned Gen |26.14|43.06|61.33|

| LLM-Regression Head(Fine-tuned only) | ＭAE(min)| RSME(min)   | MAPE(%) |
| -----|---------------- | --------- |-----------|
| Llama-3.2-3B-Instruct + Reg |**15.84** | **32.93** |38.50 |
| Llama-3.2-Taiwan-3B-Instruct + Reg |18.05 | 35.26 |**35.87** |



## Contributing

- **Code**: This repository is licensed under the [MIT License](LICENSE).
- **Model Weights**: The model weights are derived from [Meta Llama 3](https://llama.meta.com/) and are subject to the [Llama 3 Community License Agreement](https://llama.meta.com/llama3/license/).

## Acknowledgements
Built with Meta Llama 3.