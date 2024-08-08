import torch.nn as nn

from Sublayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        # Initialize the multi-head attention layer
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        # Initialize the position-wise feed-forward layer
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # Apply the multi-head attention layer to the input
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # Apply the non-padding mask to the output
        enc_output *= non_pad_mask

        # Apply the position-wise feed-forward layer to the output
        enc_output = self.pos_ffn(enc_output)
        # Apply the non-padding mask to the output
        enc_output *= non_pad_mask

        # Return the output and the attention weights
        return enc_output, enc_slf_attn