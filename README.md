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

## changelog

Model changes:

[✓] Import OneCycleLR: The OneCycleLR scheduler is imported from torch.optim.lr_scheduler.
[✓] sLSTM class: The f_bias and dropout are added to the sLSTM class.
[✓] mLSTM class: The f_bias and dropout are added to the mLSTM class.
[✓] xLSTMBlock class: The xLSTMBlock class is implemented with configurable ratio of sLSTM and mLSTM blocks, and layer normalization is applied.
[✓] GPT class: The xLSTM_blocks are used in the GPT class instead of separate sLSTM and mLSTM blocks.
[✓] configure_optimizers method: The configure_optimizers method in the GPT class is updated to use AdamW optimizer and OneCycleLR scheduler.

Training script changes:

[✓] Import statement: The OneCycleLR scheduler is imported.
[✓] Optimizer and scheduler initialization: The optimizer and scheduler are obtained from the configure_optimizers method of the GPT class.
[✓] Loading optimizer and scheduler state: The optimizer and scheduler states are loaded from the checkpoint when resuming training.
[✓] Saving scheduler state: The scheduler state is included in the checkpoint dictionary.
[✓] Stepping the scheduler: The scheduler.step() is called after each optimizer step.
[✓] Logging learning rate and MFU: The learning rate and MFU are logged using wandb (if wandb_log is enabled).
[✓] estimate_loss function: The estimate_loss function is updated to use the ctx context manager.
[✓] Training loop: The training loop is updated to use scaler.scale(loss).backward() and scaler.step(optimizer) for gradient scaling when training in fp16.
