import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

# MultiHeadAttention

def scaled_dot_product(q, k, v, mask=None):
    # q, k, v = 30 * 8 * 120 * 64
    d_k = q.size()[-1] # 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # 30 * 8 * 120 * 120 (120  * 64 cross product 64 * 120)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1) 
    values = torch.matmul(attention, v) 
    return values, attention 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # output of attention unit 15
        self.num_heads = num_heads # 5
        self.head_dim = d_model // num_heads # 15 / 5 = 3
        self.qkv_layer = nn.Linear(d_model, 3 * d_model) # 15 * 45
        self.linear_layer = nn.Linear(d_model, d_model) # 15 * 15
    
    def forward(self, x, mask = None):
        batch_size, seq_len, d_model = x.size() # 30 * 120 * 15
        # print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x) # 30 * 120 * 45
        # print(f"qkv.size(): {qkv.size()}") 
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim) # 30 * 120 * 5 * 9
        # print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3) # 8 * 5 * 120 * 9
        # print(f"qkv.size(): {qkv.size()}")

        q, k, v = qkv.chunk(3, dim = -1) # each vctor is 8 * 5 * 120 * 3
        # print(f"q.size(): {q.size()} | k.size() : {k.size()} | v.size(): {v.size()}")
        values, attention = scaled_dot_product(q, k, v) 
        # print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        # print(f"values.size(): {values.size()}")
        out = self.linear_layer(values)
        # print(f"out.size(): {out.size()}")
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # output of attention unit 15
        self.num_heads = num_heads # 5
        self.head_dim = d_model // num_heads # 15 / 5 = 3
        self.kv_layer = nn.Linear(d_model, 2 * d_model) # 15 * 30
        self.q_layer = nn.Linear(d_model, d_model) # 15 * 15
        self.linear_layer = nn.Linear(d_model, d_model) # 15 * 15
    
    def forward(self, x, y, mask = None):
        batch_size, seq_len_x, d_model = x.size() # 8 * 120 * 15
        _, seq_len_y, _ = y.size() # 8 * 12 * 15
        # print(f"x.size(): {x.size()}")
        kv = self.kv_layer(x) # 8 * 120 * 30
        q = self.q_layer(y) # 8 * 12 * 15

        kv = kv.reshape(batch_size, seq_len_x, self.num_heads, 2 * self.head_dim) # 8 * 120 * 5 * 6
        q = q.reshape(batch_size, seq_len_y, self.num_heads, self.head_dim) 

        kv = kv.permute(0, 2, 1, 3) # 8 * 5 * 120 * 6
        q = q.permute(0, 2, 1, 3) # 8 * 5 * 12 * 3

        k, v = kv.chunk(2, dim = -1)
        # print(f"q.size(): {q.size()} | k.size() : {k.size()} | v.size(): {v.size()}")
        values, attention = scaled_dot_product(q, k, v, mask) 
        # print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, seq_len_y, self.num_heads * self.head_dim)
        # print(f"values.size(): {values.size()}")1
        out = self.linear_layer(values)
        # print(f"out.size(): {out.size()}")
        return out
    
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_len):
        super().__init__()
        self.max_sequence_len = max_sequence_len
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(0, self.max_sequence_len, 1).reshape(self.max_sequence_len, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim = 2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
# Layer Norm

class LayerNormalization(nn.Module):
    def __init__(self, parameter_shape, eps = 1e-5):
        super().__init__()
        self.parameter_shape = parameter_shape # can be same as d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))
    
    def forward(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        # print(f"Mean \n {mean.size()}: \n {mean}")
        var = ((inputs-mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        # print(f"Standard deviation \n ({std.size()}): \n {std}")
        y = (inputs - mean) / std
        # print(f"y \n ({y.size()}) \n {y}")
        out = self.gamma * y + self.beta
        # print(f"output \n ({out.size()}): \n {out}")
        return out

# Feed Forward Layer
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 * 2048 
        self.linear2 = nn.Linear(hidden, d_model) # 2048 * 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Encoder
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_data, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__() 
        self.input_embedding = nn.Linear(in_features=d_data, out_features=d_model)
        self.attention = MultiHeadAttention(d_model = d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameter_shape=[d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2 = nn.Dropout(p = drop_prob)
    
    def forward(self, x):
        x = self.input_embedding(x)
        resid_x = x
        x = self.attention(x, mask = None)
        x = self.dropout1(x)
        x = self.norm1(x + resid_x)
        resid_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + resid_x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, d_data, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, d_data, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.layers(x)
        return x

# Decoder
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_data, ffn_hidden, num_heads, drop_prob) -> None:
        super(DecoderLayer, self).__init__()
        self.target_embedding = nn.Linear(in_features=d_data, out_features=d_model)
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameter_shape=[d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)

        self.cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameter_shape=[d_model])
        self.dropout2 = nn.Dropout(p = drop_prob)
        
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.norm3 = LayerNormalization(parameter_shape=[d_model])
        self.dropout3 = nn.Dropout(p = drop_prob)
    
    def forward(self, x, y, decoder_mask):
        y = self.target_embedding(y)

        resid_y = y
        y = self.self_attention(y, mask = decoder_mask)
        y = self.dropout1(y)
        y = self.norm1(y + resid_y)

        resid_y = y
        y = self.cross_attention(x, y)
        y = self.dropout2(y)
        y = self.norm2(y + resid_y)

        resid_y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + resid_y)

        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, * inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, d_data, ffn_hidden, num_heads, drop_prob, num_layers) -> None:
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, d_data, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, mask):
        # x: 30 * 200 * 512
        # y: 30 * 200 * 512
        # mask: 200 * 200
        y = self.layers(x, y, mask)
        return y

# Full transformer

class Transformer(nn.Module):
    def __init__(self,
                d_enc,
                d_dec,
                d_data,
                d_temp,
                d_precip,
                ffn_hidden,
                num_heads,
                drop_prob,
                num_layers,
                num_encoders = 2
                ):
        super().__init__()
        
        self.temp_encoder = Encoder(d_enc, d_temp, ffn_hidden, num_heads, drop_prob, num_layers)
        self.precip_encoder = Encoder(d_enc, d_precip, ffn_hidden, num_heads, drop_prob, num_layers)

        self.decoder = Decoder(d_dec, d_dec, ffn_hidden, num_heads, drop_prob, num_layers)


        self.linear = nn.Linear(d_dec, d_data)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x, y):
        mask = (torch.triu(torch.ones(12, 12)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


        # x is now [batch_size * num_data_types * months * num_features]
        temp_x = x[:, :, 0, :]
        precip_x = x[:, :, 1, :]

        temp_x = self.temp_encoder(temp_x)
        precip_x = self.precip_encoder(precip_x)

        # Concatenate the two together
        combined_x = torch.cat([temp_x, precip_x], dim=-1)

        # Add any necessary processing for y if required
        temp_y = y[:, :, 0, :]
        precip_y = y[:, :, 1, :]
        
        combined_y = torch.cat([temp_y, precip_y], dim = -1)

        out = self.decoder(combined_x, combined_y, mask)
        out = self.linear(out)
        return out