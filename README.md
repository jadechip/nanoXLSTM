# nanoXLSTM

![nanoGPT](assets/nanogpt.jpg)

nanoXLSTM is a minimal codebase for playing around with language models based on the xLSTM (extended Long Short-Term Memory) architecture from the awesome research paper: xLSTM: Extended Long Short-Term Memory and heavily inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

![repro124m](assets/xlstm_loss.png)

\*\*Note: Work in progress!!!
I am working on improving the generated text.

No lofty goals here - just a simple codebase for tinkering with this innovative xLSTM technology!

Contributions are more than welcome as I continue exploring this exciting research direction.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## data prep

!python data/shakespeare_char/prepare.py

## train

```
python train.py config/train_shakespeare_char.py
```

## inference

```
python sample.py --out_dir=out-shakespeare-char
```

## todos

- Run hyperparameter sweep
