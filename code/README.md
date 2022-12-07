### 1. Preparation

#### Data Preparation
a) Download m4singer.zip, then unzip this file into `data/raw`.

b) Run the following scripts to pack the dataset for training/inference.

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/m4singer/base.yaml

# `data/binary/m4singer` will be generated.
```

#### Vocoder Preparation
We use the pre-trained [Vocoder](https://drive.google.com/file/d/10LD3sq_zmAibl379yTW5M-LXy2l_xk6h/view?usp=share_link)
and [PitchExtractor](https://drive.google.com/file/d/19QtXNeqUjY3AjvVycEt3G83lXn2HwbaJ/view?usp=share_link). Please unzip this file into `checkpoints` before training your acoustic model.

### 2. Training Example
First, you need a pre-trained FFT-Singer checkpoint. You can use the pre-trained model, or train FFT-Singer from scratch, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/m4singer/fs2.yaml --exp_name m4singer_fs2_e2e --reset
```

Then, to train DiffSinger, run:

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/m4singer/diff.yaml --exp_name m4singer_diff_e2e --reset  
```


### 3. Inference from packed test set
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/m4singer/diff.yaml --exp_name m4singer_diff_e2e --infer
```

We also provide:
 - the pre-trained model of [DiffSinger](https://drive.google.com/file/d/1LsTnCNinx5tQaRlDSbbxoZYrgPwR3CgL/view?usp=share_link);
 - the pre-trained model of [FFT-Singer](https://drive.google.com/file/d/1JB1kwhQJT-hAMGF7Ykq2b7w95vuwksus/view?usp=share_link);

Remember to put the pre-trained models in `checkpoints` directory.

### 4. Inference from raw inputs
The way to generate a single utterance. The generated audio can be found at `infer_out`.
```sh
python inference/m4singer/ds_e2e.py --config usr/configs/m4singer/diff.yaml --exp_name m4singer_diff_e2e
```
The way to start the service locally.
```sh
CUDA_VISIBLE_DEVICES=0 python inference/m4singer/gradio/infer.py
```