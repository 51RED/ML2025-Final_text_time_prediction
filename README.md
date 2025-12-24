>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

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

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ?, run:

- generative
    ```eval
    python3 eval_generative.py -m generative
    ```
- regression
    ```eval
    python3 eval_regression.py -m regression
    ```
>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

## License
- **Code**: This repository is licensed under the [MIT License](LICENSE).
- **Model Weights**: The model weights are derived from [Meta Llama 3](https://llama.meta.com/) and are subject to the [Llama 3 Community License Agreement](https://llama.meta.com/llama3/license/).

## Acknowledgements
Built with Meta Llama 3.