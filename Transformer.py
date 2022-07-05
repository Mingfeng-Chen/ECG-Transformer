# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:17:43 2022

@author: ChenMingfeng
"""

import torch
import torch.nn as nn

class LSM_IniReconNet(nn.Module):
    def __init__(self, SamplingPoints, BlockSize = 16):
        ## Smpling Points  = sampling ratios * 32 ** 2 
        super(LSM_IniReconNet, self).__init__()
        self.BlockSize = BlockSize
        self.SamplingPoints = SamplingPoints

        ## learnable LSM
        self.sampling = nn.Conv1d(1, SamplingPoints , BlockSize, stride = 16, padding = 0,bias=False)
        nn.init.normal_(self.sampling.weight, mean=0.0, std=0.028)  

        ## linear intial recon (by basic linear operator)
        self.init_bl = nn.Conv1d(SamplingPoints, BlockSize, 1, bias=False)

    def forward(self, x):
        ## cut the image into patches of pre-defined blocksize
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nb = x.size(0)

        x = self.ImageCrop(x)

        x_IR = torch.zeros(nb, self.num_patches, self.SamplingPoints, 1)

        for i in range(0, self.num_patches):
            temp = x[:, i, :, :, :]
            temp = temp.squeeze(3)
            temp = temp.to(device)
            temp = self.sampling(temp)
            x_IR[:, i, :, :] = temp

        y_IR = torch.zeros(nb, self.num_patches, self.BlockSize, 1)

        ## initial Recon
        for i in range(0, self.num_patches):
            temp_IR = x_IR[:,i,:,:]
            temp_IR = temp_IR.to(device)
            temp_IR = self.init_bl(temp_IR)
            y_IR[:, i, :, :] = temp_IR

        ## reshape and concatenation. 
        y_IR = y_IR.reshape(nb, -1, 1)
        y_IR = y_IR.to(device)

        return y_IR

    def ImageCrop(self, x, BlockSize = 16):
        H = x.size(2)
        L = x.size(3)
        nb = x.size(0)
        nc = x.size(1)
        num_patches = H * L // BlockSize
        y = torch.zeros(nb, num_patches, nc, BlockSize, 1)
        ind = range(0, H, BlockSize)
        count = 0
        for i in ind:
                temp = x[:,:,i:i+ BlockSize, :]
                temp2 = y[:,count,:,:,:,]
                y[:,count,:,:,:,] = temp
                count = count + 1
        self.oriH = H
        self.oriL = L
        self.num_patches = num_patches
        return y

'''
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv1d(
                in_chans,
                patch_size,
                kernel_size=patch_size,
                stride=patch_size
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, height, width)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, patch_size)`.
        """
        x = self.proj(
                x
            )  
        x = x.transpose(1,2)

        return x
'''
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, device):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.device = device

    def forward(self, x):
        N = x.shape[0]
        x = torch.reshape(x, [N, -1, self.patch_size]).to(self.device)
        return x
        
class SelfAttention(nn.Module):
    def __init__(self, patch_size, heads):
        super(SelfAttention,self).__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = patch_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, patch_size)
        
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values) 
        keys = self.keys(keys)  
        queries = self.queries(query)
       
        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        
        attention = torch.softmax(energy / (self.patch_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
            N, query_len, self.heads*self.head_dim)
        
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, patch_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(patch_size, heads)
        self.norm1 = nn.LayerNorm(patch_size)
        self.norm2 = nn.LayerNorm(patch_size)       
        
        self.feed_forward = nn.Sequential(
            nn.Linear(patch_size, forward_expansion*patch_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*patch_size, patch_size)
            )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(self,
                 patch_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 num_block):
        super(Encoder, self).__init__()
        self.patch_size = patch_size
        self.device = device
        self.patch_embedding = PatchEmbedding(patch_size, device)
        self.position_embedding = nn.Embedding(num_block, patch_size)
        self.num_block = num_block
        
        self.layers = nn.ModuleList(
            [TransformerBlock(patch_size, 
                              heads, 
                              dropout=dropout, 
                              forward_expansion=forward_expansion
                              )
             for _ in range(num_layers)]
            )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        N = x.shape[0]
        positions = torch.arange(0, self.num_block).expand(N, self.num_block).to(self.device)
        
        out = self.dropout(
            (self.patch_embedding(x) + self.position_embedding(positions))
            )
        
        for layer in self.layers:
            out = layer(out, out, out)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, patch_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(patch_size, heads)
        self.norm = nn.LayerNorm(patch_size)
        self.transformer_block = TransformerBlock(
            patch_size, heads, dropout, forward_expansion
            )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key):
        attention = self.attention(x, x, x)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query)
        return out
    
class Decoder(nn.Module):
    def __init__(self,
                 patch_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 num_block):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = PatchEmbedding(patch_size, device)
        self.position_embedding = nn.Embedding(num_block, patch_size)
        self.patch_size = patch_size
        self.num_block = num_block

        self.layers = nn.ModuleList(
            [DecoderBlock(patch_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out):
        enc_out = self.word_embedding(enc_out)
        N = x.shape[0]
        positions = torch.arange(0, self.num_block).expand(N, self.num_block).to(self.device)
        
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out)
        
        return x
        
class ECGTransformer(nn.Module):
    def __init__(self,
                 block_size,
                 sample_size,
                 original_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=4,
                 dropout=0.1,
                 device="cuda"):
        super(ECGTransformer, self).__init__()
        self.num_block = original_size // block_size
        
        self.lsm = LSM_IniReconNet(sample_size)
        self.encoder = Encoder(block_size, num_layers, heads, device, 
                               forward_expansion, dropout, self.num_block)
        self.decoder = Decoder(block_size, num_layers, heads, forward_expansion,
                               dropout, device, self.num_block)
        

    def forward(self, src, trg):
        N = src.shape[0]
        src = self.lsm(src).reshape(N, -1)
        
        enc_src = self.encoder(src)
        out = self.decoder(trg, enc_src).reshape(N, -1)
        return out
    