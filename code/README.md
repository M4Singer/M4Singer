### 1. Preparation

#### Data Preparation
a) Download m4singer-short, then unzip this file into `data/processed`.

b) Run the following scripts to pack the dataset for training/inference.

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config usr/configs/m4singer/base.yaml

# `data/binary/m4singer` will be generated.
```

#### Vocoder Preparation
We use the same pre-trained [Vocoder](https://drive.google.com/file/d/12exIcy9hxvc8sJH_oXDyrL9xtxvfZtyl/view?usp=sharing)
and [PitchExtractor](https://drive.google.com/file/d/1D1Wbp9HtNljQKjriYqCEQ88UoWLEEOWp/view?usp=sharing) as [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger/blob/master/docs/README-SVS.md).
Please unzip this file into `checkpoints` before training your acoustic model.

### 2. Training Example
First, you need a pre-trained FFT-Singer checkpoint. You can use the pre-trained model, or train FFT-Singer from scratch, run:
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/m4singer/fs2.yaml --exp_name fs2_m4singer --reset
```

Then, to train DiffSinger, run:

```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/m4singer/diff.yaml --exp_name diff_m4singer --reset  
```


### 3. Inference from packed test set
```sh
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config usr/configs/m4singer/diff.yaml --exp_name diff_m4singer --infer
```

We also provide:
 - the pre-trained model of [DiffSinger](https://drive.google.com/file/d/13xkVAKCpZOyYdaQEmrfsK4196dDJgyK9/view?usp=sharing);
 - the pre-trained model of [FFT-Singer](https://drive.google.com/file/d/13xkVAKCpZOyYdaQEmrfsK4196dDJgyK9/view?usp=sharing);

Remember to put the pre-trained models in `checkpoints` directory.

More pre-trained model can be found at [here](https://drive.google.com/drive/folders/1ZkxbZTFjHroNpxmXuR3sQIgmT1mObNgw?usp=sharing). 
The rest of the code will be released after it is accepted.