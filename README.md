# Jack's Encoder-Decoder Transformer Implementation
Like many others, I have reimplemented the original encoder-decoder transformer from the [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762) as a big first step in understanding generative AI.

I will be writing a blog post soon detailing my process writing and training this model, and what I learned from it.

## How do I run the code?
### Running in Google Colab (recommended)
1. Download the EDTransformer.ipynb file and open it in Colab (unfortunately, the "Open in Colab" button will not work).
2. Upload this repo somewhere to your Google Drive.
3. Change the cell starting with the word "folder" to point to the path of where you uploaded the repo.
4. Run the notebook! Give Colab permission to mount your Google Drive. I recommend using a GPU runtime as it will train much faster.


### Running from your desktop
1. Download or pull down the repo.
2. Run `python3 -m pip install -e`.
3. Run `python3 main.py`

## Repo Structure
### notebooks
Contains various notebooks used when developing this model.
* EDTransformer.ipynb - the final runnable made in 2024.
* EDTransformerDevNotebook.ipynb - the notebook used during development. Out of date compared to the rest of the repo.
* attention_from_scratch.ipynb - The notebook used during my original, rushed implementation in 2022.
* attention_from_scratch_may_2023.ipynb - An updated version of the 2022 implementation that has better performance.

### src
Contains the source code.

#### dataset
Contains files used for creating, tokenizing, and padding the dataset.

#### model
Contains all files used for constructing the model. 
* encoder_decoder_transformer.py is what puts everything together.
* extended_pytorch_transformer is the pytorch transformer with an embedding, positional embedding, and unembedding added to the ends of it. I used it to compare the performance of my custom implementation.

#### training
* Contains a file used for training the model.

### tst
Contains unit tests for components of the model. Most of the unit tests take in a sample input, fix the component's weights and biases, and assert an output.
