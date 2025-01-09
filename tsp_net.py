
import math

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

class TmEncoder(nn.Module):
  """Configurable Tm Encoder
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, *args, **kwargs):
    super(TmEncoder, self).__init__()
    
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    
    self.time_embed = nn.Sequential(
        nn.Linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        nn.Linear(time_embed_dim, time_embed_dim),
    )
    self.out = nn.Sequential(
        # normalization(hidden_dim),
        nn.GroupNorm(32,hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GraphTransformerLayer(hidden_dim, hidden_dim, last_layer=(_==n_layers-1))
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_embed_dim,hidden_dim),
        ) for _ in range(n_layers)
    ])


  def forward(self, x, timesteps, v):
    """
    Parameters:
        v: Input node coordinates (B x V x 2)
        x: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
    Returns:
        e: Updated edge features (gs:B x V x V x 1, cg:B x V x V x 2)
    """
    # Embed edge features
    v = self.node_embed(self.pos_embed(v))
    e = self.edge_embed(self.edge_pos_embed(x))
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))

    for layer, time_layer in zip(self.layers, self.time_embed_layers):
      time_embedding = time_layer(time_emb)[:, None, None, :]
      v, e = layer(v, e, time_embedding)
      
    e = self.out(e.permute((0, 3, 1, 2)))
    return e


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.0, use_bias=False, last_layer=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.last_layer = last_layer
        
        # DiGress
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim, num_heads, use_bias, last_layer)
        

        if(not self.last_layer): 
          self.layer_norm1_h = nn.LayerNorm(out_dim)
        self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        
        # FFN for h
        if(not self.last_layer): 
          self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
          self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)


    def forward(self, h, e, time_embedding):
        """
        Transformer Encoder Layer
        Parameters:
            h: node embedding (B x V x H)
            e: edge embedding (B x V x V x H)
            timesteps: timestep embedding (B x V x V x H)
        Returns:
            h: new node embedding (B x V x H)
            e: new edge embedding (B x V x V x H)
        """
        if(not self.last_layer): 
          h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        
        # multi-head attention out
        h, e = self.attention( h, e)
        
        # add time_embedding
        e = e + time_embedding

        if(not self.last_layer): 
          h = self.layer_norm1_h(h)
        e = self.layer_norm1_e(e)


        # FFN for h
        if(not self.last_layer): 
          h = self.FFN_h_layer1(h)
          h = F.relu(h)
          h = F.dropout(h, self.dropout, training=self.training)
          h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        # residual connection
        if(not self.last_layer): 
          h = h_in1 + h # residual connection
        e = e_in1 + e # residual connection       

        return h, e

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, last_layer=False):
        super().__init__()
        assert out_dim % num_heads == 0, f"dx: {out_dim} -- nhead: {num_heads} not match"
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.df = out_dim//num_heads
        self.last_layer = last_layer

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim, bias=True)
            self.K = nn.Linear(in_dim, out_dim, bias=True)
            self.V = nn.Linear(in_dim, out_dim, bias=True)
            self.mix_y1 = nn.Linear(self.df, self.df, bias=True)
            self.mix_y2 = nn.Linear(self.df, self.df, bias=True)
            if(not self.last_layer):
              self.mix_e1 = nn.Linear(in_dim, out_dim, bias=True)
              self.mix_e2 = nn.Linear(in_dim, out_dim, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim, bias=False)
            self.K = nn.Linear(in_dim, out_dim, bias=False)
            self.V = nn.Linear(in_dim, out_dim, bias=False)
            self.mix_y1 = nn.Linear(self.df, self.df, bias=False)
            self.mix_y2 = nn.Linear(self.df, self.df, bias=False)
            if(not self.last_layer):
              self.mix_e1 = nn.Linear(in_dim, out_dim, bias=False)
              self.mix_e2 = nn.Linear(in_dim, out_dim, bias=False)
                    
    def forward(self,h, e):
        """
        Multiattention Layer
        Parameters:
            h: node embedding (B x V x H)
            e: edge embedding (B x V x V x H)
        Returns:
            h: new node embedding (B x V x dx (dx = H) )
            newE: new edge embedding (B x V x V x dx (dx = H) )
        """        
        Q_h = self.Q(h) # B V dx
        K_h = self.K(h) # B V dx
        V_h = self.V(h) # B V dx
        

        # multihead attention
        # Reshaping, dx=h*d
        h_shape = h.shape[:-1]
        e_shape = e.shape[:-1]
        Q_h = Q_h.view(*h_shape, self.num_heads, self.df) # B V h d
        K_h = K_h.view(*h_shape, self.num_heads, self.df) # B V h d
        V_h = V_h.view(*h_shape, self.num_heads, self.df) # B V h d


        Q_h = Q_h.unsqueeze(2)                              # (B, V, 1, h, d)
        K_h = K_h.unsqueeze(1)                              # (B, 1, V, h, d)

        # Compute unnormalized attentions. Y is (B, V, V, h, d)
        Y = Q_h * K_h
        Y = Y / np.sqrt(self.df)

        ## mix
        if(not self.last_layer):
          mix_e1 = self.mix_e1(e) # B V V dx
          mix_e2 = self.mix_e2(e) # B V V dx
          E1 = mix_e1.view(*e_shape, self.num_heads, self.df) # B V V h d
          E2 = mix_e2.view(*e_shape, self.num_heads, self.df) # B V V h d
        Y1 = self.mix_y1(Y)                     # B, V, V, h, d
        Y2 = self.mix_y2(Y)                     # B, V, V, h, d
        Y1 = Y1.flatten(start_dim=3)             # B, V, V, dx
        Y2 = Y2.flatten(start_dim=3)             # B, V, V, dx
        
        # Compute Mix(Y,newE) to update e 
        newE = e * (Y1 + 1) + Y2                 # (B, V, V, dx)

        if(not self.last_layer):
          # Compute Mix(E,Y) to incorporate edge features to the self attention scores.
          Y = Y * (E1 + 1) + E2                    # (B, V, V, h, d)
          
          # Compute attentions. attn is still (B, V, V, h, d)
          score = torch.softmax(Y,dim=2)           # (B, V, V, h, d)
          V_h = V_h.unsqueeze(1)                   # (B, 1, V, h, d)

          # Compute weighted values
          weighted_V = score * V_h                 # (B, V, V, h, d)
          weighted_V = weighted_V.sum(dim=2)       # (B, V, h, d)

          # Send output to input dim
          h = weighted_V.flatten(start_dim=2)          # B, V, dx

        return h, newE


class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    y_embed = x[:, :, 0]
    x_embed = x[:, :, 1]
    if self.normalize:
      # eps = 1e-6
      y_embed = y_embed * self.scale
      x_embed = x_embed * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    return pos


class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return pos_x

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  
