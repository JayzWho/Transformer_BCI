import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import Consts as Constants
import math
from Layers import EncoderLayer

'''Construct a simple CNN model'''
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.fc1 = nn.Linear(64 * 22, 128)  # Adjust the size according to your input dimensions
        self.fc2 = nn.Linear(128, self.num_classes)

        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        
        # Initialize biases to zero
        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv2.bias)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)


    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension, shape: [batch_size, 1, 96]
        #print(f'After unsqueeze: {x.shape}')
        x = self.pool(F.relu(self.conv1(x)))
        #print(f'After conv1 and pool: {x.shape}')
        x = self.pool(F.relu(self.conv2(x)))
        #print(f'After conv2 and pool: {x.shape}')
        x = x.view(x.size(0), -1)  # Flatten
        #print(f'After flatten: {x.shape}')
        x = F.relu(self.fc1(x))
        #print(f'After fc1: {x.shape}')
        x = self.fc2(x)
        #print(f'After fc2: {x.shape}')
        x = F.softmax(x, dim=1)
        
        return x

def get_CNN_model(num_classes):
    model = SimpleCNN(num_classes)
    return model

'''Masking for Transformer'''
def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

'''Transformer Model'''
class EEGEncoder(nn.Module):
    def __init__(self, feature_dim, d_model):
        super(EEGEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.linear = nn.Linear(feature_dim, d_model)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, features, non_pad_mask):
        # features: batch*seq_len*feature_dim
        # non_pad_mask: batch*seq_len

        # Linear transformation
        enc_output = self.linear(features)
        #print(enc_output.shape)
        #print(non_pad_mask.shape)
        # Apply non-padding mask
        enc_output *= non_pad_mask

        return enc_output

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,feature_dim,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()
        
        self.d_model = d_model
        self.feature_dim = feature_dim

        #adjusting scale difference between EEG and temp
        self.eeg_weight = nn.Parameter(torch.tensor(1.0))
        self.tem_weight = nn.Parameter(torch.tensor(1.0))
        self.init_weights()

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        #self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
        
        self.eeg_encoder = EEGEncoder(feature_dim, d_model)

    def init_weights(self):
        nn.init.constant_(self.eeg_weight, 1.0)  # 初始化为1.0
        nn.init.constant_(self.tem_weight, 1e-4)  # 初始化为1e-4
        
    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask
    


    def forward(self,event_type, features, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_time)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_time, seq_q=event_time)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        EEG_enc = self.eeg_encoder(features, non_pad_mask)
        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        EEG_enc *= self.eeg_weight
        tem_enc *= self.tem_weight
        '''Mind whether to include temporal encoding'''
        EEG_enc += tem_enc
        #Sprint("EEG_enc:",EEG_enc)
        #enc_output = self.event_emb(event_type)
        enc_output = 0
        for enc_layer in self.layer_stack:
            enc_output += EEG_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        #print("enc_output:",enc_output)
        return enc_output

class Predictor(nn.Module):
    def __init__(self, d_model, num_classes):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, encoder_output, mask):
        # encoder_output: (batch, seq_len, d_model)
        # mask: (batch, seq_len, 1)

        # Apply the fully connected layer
        output = self.fc(encoder_output)  # (batch, seq_len, num_classes)

        # Apply mask
        mask = mask.expand_as(output)  # (batch, seq_len, num_classes)

        output = F.softmax(output, dim=2)

        output = output * mask  # Mask out the padded positions
        
        return output
'''
class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out
'''

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,feature_dim=96,
            num_types=3, d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            feature_dim=feature_dim,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types
        self.predictor = Predictor(d_model, num_types)
        '''
        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)'''

    def forward(self, event_type,features, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_time)

        enc_output = self.encoder(event_type, features, event_time, non_pad_mask)
        prediction= self.predictor(enc_output, non_pad_mask)
        #print("prediction:",prediction)
        return prediction
        #enc_output = self.rnn(enc_output, non_pad_mask)

        '''time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)'''
        
def get_trans_model(
        feature_dim=96, num_types=3, d_model=256, d_inner=1024,
        n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.3):
    """
    Creates and returns a Transformer model with the specified parameters.
    Args:
        feature_dim (int): The dimension of the input features.
        num_types (int): The number of output types/classes.
        d_model (int): The dimension of the model.
        d_inner (int): The dimension of the inner feed-forward layers.
        n_layers (int): The number of encoder layers.
        n_head (int): The number of attention heads.
        d_k (int): The dimension of the key vectors.
        d_v (int): The dimension of the value vectors.
        dropout (float): The dropout rate.
    Returns:
        model (Transformer): The instantiated Transformer model.
    """
    model = Transformer(
        feature_dim=feature_dim,
        num_types=num_types,
        d_model=d_model,
        d_inner=d_inner,
        n_layers=n_layers,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout
    )
    return model