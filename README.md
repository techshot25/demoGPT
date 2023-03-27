# demoGPT
Demo of general GPT model using PyTorch

### Set up (optional)

To set up this model simply adjust the parameters in the config.yml file.
Currently the model simply detects if you have a CUDA enabled GPU and uses that for training.
You can adjust the `block_size` and `batch_size` based on your machine's capabilities.

These are the configuration parameters:

- `hyper_params`

  - **float** `lr` Learning rate of the optimizer.

  - **str** `tokenizer` The type of tokenizer to use.

  - **int** `embed_size` Size of embedding dimension.

  - **float** `dropout` The dropout probability for regularization.

  - **int** `block_size` The size of each block (number of tokens) that is passed as memory and target to the transformer.
  
  - **int** `batch_size` The size of each batch for the dataloader.

  - **int** `num_heads` The number of attention heads in the multi-head attention model.

- `train_params`
  - **int** `num_epochs` Number of training cycles.

### Usage

To use the model, first you must install the required modules:

```python
pip install -r requirements.txt
```

Then use the `main.py` as the entrypoint.

#### To train and save

```python
python main.py --save
```

This saves the model to a file in the current working directory titled `gpt.pt` but you can rename it with the `-f` or `--file-name` argument.

#### To evaluate

```python
python main.py --cached --evaluate
```

To see a full description of the arguments, type:

```python
python main.py --help
```
