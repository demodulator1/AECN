# AECN

This is the open-source implementation of the paper "Attention-Enhanced Convolutional Networks for Short-Term Traffic Flow Prediction".

## Project Structure

```
AECN/
├── ANN.py                  # Artificial Neural Network baseline model
├── CNN.py                  # Convolutional Neural Network baseline model
├── CNN_attention.py        # The proposed AECN model in this paper
├── Flowformer.py           # Flowformer baseline model
├── GRU.py                  # GRU baseline model
├── LSTM.py                 # LSTM baseline model
├── RNN.py                  # RNN baseline model
├── Transformer.py          # Transformer baseline model
├── read_data.py            # Data reading utility
├── reshape_data.py         # Data reshaping utility
├── LICENSE                 # MIT license
├── README.md               # Project documentation
└── Ablation Study/         # Corresponding to the ablation study section in the paper
    ├── channel_halving.py
    ├── read_data.py
    ├── reshape_data.py
    ├── without_Attention.py
    ├── without_CNN.py
    ├── without_gelu.py
    └── without_transformer.py
```

## License

This project is open-sourced under the MIT License.
