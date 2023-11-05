"""
GPT implementation by Stephane Nguyen Liem based on the paper:
"Formal Algorithms for Transformers"
by Mary Phuong and Marcus Hutter (Google Deepmind)

Notation-wise, I've replaced some notations:
    - e.g. "lmax" by "T"
    - I've transposed "everything" and used batched operations
    
I've also fixed the positional encoding to follow the "Attention
is all you need" paper
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class GPT:
    def __init__(self) -> None:
        pass

    def train(self):
        pass

    def inference(self, prompt: str) -> str:
        pass

class Decoder(nn.Module):
    def __init__(self,
                 T: int,  # max seq. length
                 vocab_size: int,
                 embed_dim: int,
                 mlp_hidden_dim: int,
                 num_layers: int = 3,
                 num_heads: int = 1) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim has to be an even number"

        # https://bobbyhadz.com/blog/python-get-all-arguments-passed-to-function
        self.__dict__.update(locals())
        
        ### Token embedding: vocab_size x embed_dim
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        ### Frozen Positional embedding: T x embed_dim
        # Each row is the embedding of a position t
        # TODO: find whether we start at 1 or 0 ?!
        num = torch.arange(T)[:, None]
        denom = torch.pow(T, 2*torch.arange(embed_dim//2)/embed_dim)[None, :]
        pos_embed_matrix = torch.stack([torch.sin(num/denom), torch.cos(num/denom)], dim=-1).flatten(start_dim=-2)  # interleaved
        self.pos_emb = nn.Embedding.from_pretrained(pos_embed_matrix, freeze=True)
        
        ### Masked Multi-Head Attention layers
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool))  # TODO uni-directional self-attention !!
        kwargs_mhas = {"Tx": T,
                       "Tz": T,
                       "Dx": embed_dim,
                       "Dz": embed_dim,
                       "Datt": embed_dim,
                       "Dmid": embed_dim,
                       "Dout": embed_dim,
                       "num_heads": num_heads,
                       "mask": mask}
        self.mhas = [MultiHeadAttention(**kwargs_mhas) for _ in range(num_layers)]
        
        ### One hidden layer MLPs
        self.mlps = [(nn.Sequential(
                        nn.Linear(embed_dim, mlp_hidden_dim),
                        nn.GELU(),
                        nn.Linear(mlp_hidden_dim, embed_dim),
                        ))
                        for _ in range(num_layers)]

        ### Pre-norm residual units
        # https://arxiv.org/pdf/1906.01787.pdf
        self.norm_add_mhas = nn.ModuleList([PreNormResUnit(embed_dim, mha) for mha in self.mhas])
        self.norm_add_mlps = nn.ModuleList([PreNormResUnit(embed_dim, mlp) for mlp in self.mlps])        
        self.final_layernorm = nn.LayerNorm(embed_dim)

        self.prob_proj = nn.Sequential(
                    nn.Linear(embed_dim, vocab_size, bias=False),
                    nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:
            x: N sequences of T ids: N x T
        intermediate:
            x or e: N sequences of T embeddings: N x T x embed_dim
        output:
            y: proba distribution over next token
               given whole past for each seq   : N x T x vocab_size
        """
        # Additive positional encoding
        e = self.token_emb(x) + self.pos_emb(torch.arange(self.T).view(1, self.T))
        x = e
        for norm_add_mha, norm_add_mlp in zip(self.norm_add_mhas,
                                              self.norm_add_mlps):
            x = norm_add_mha(x, x)
            x = norm_add_mlp(x)
        x = self.final_layernorm(x)
        y = self.prob_proj(x)  # N x T x embed_dim -> N x T x vocab_size
        return y

class PreNormResUnit(nn.Module):
    def __init__(self,
                 normalized_shape,
                 sublayer: nn.Module) -> None:
        """
        Pre-norm residual unit: https://arxiv.org/pdf/1906.01787.pdf

        Normalization then residual part
        """
        super().__init__()
        self.sublayer = sublayer  # a PyTorch nn.Module !
        self.layernorm = nn.LayerNorm(normalized_shape)

    def forward(self,
                x: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            x_tilde = self.sublayer(self.layernorm(x), self.layernorm(z))
        else:
            x_tilde = self.sublayer(self.layernorm(x))
        return x + x_tilde
    
class PostNormResUnit(nn.Module):
    def __init__(self,
                 normalized_shape,
                 sublayer: nn.Module) -> None:
        """
        Post-norm residual unit: https://arxiv.org/pdf/1906.01787.pdf

        Residual part then normalization
        """
        super().__init__()
        self.sublayer = sublayer
        self.layernorm = nn.LayerNorm(normalized_shape)

    def forward(self,
                x: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            x_tilde = self.sublayer(x, z)
        return self.layernorm(x + x_tilde)

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 Tx: int,
                 Tz: int,
                 Dx: int,
                 Dz: int,
                 Datt: int,
                 Dmid: int,  # renamed
                 Dout: int,  # new
                 num_heads: int,  # new
                 mask: torch.Tensor) -> None:
        super().__init__()
        # https://bobbyhadz.com/blog/python-get-all-arguments-passed-to-function
        self.__dict__.update(locals())

        self.attention_layers = nn.ModuleList(
            [Attention(Tx, Tz, Dx, Dz, Datt, Dout=Dmid, mask=mask) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads*Dmid, Dout)
    
    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention or cross-attention using different
        heads, concatenate the resulting representations along
        the last dim and project them.

        For each attention head:
            If x = z, it's self-attention.
            inputs:
                x: N primary seq(s) of representations: N x Tx x Dx
                z: M context seq(s) of representations: M x Tz x Dz
            output:
                y_head: N repr. with context info     : N x Tx x Dmid
        
        Then concatenate the y's:
            y_concat: N x Tx x num_heads*Dmid
        Then project:
            y       : N x Tx x Dout
        """
        y_heads = [self.attention_layers[h](x, z) for h in range(self.num_heads)]
        y_concat = torch.cat(y_heads, dim=-1)
        return self.proj(y_concat)

class Attention(nn.Module):
    def __init__(self,
                 Tx: int,
                 Tz: int,
                 Dx: int,
                 Dz: int,
                 Datt: int,
                 Dout: int,
                 mask: torch.Tensor) -> None:
        super().__init__()
        # https://bobbyhadz.com/blog/python-get-all-arguments-passed-to-function
        self.__dict__.update(locals())

        self.query_proj = nn.Linear(Dx, Datt)
        self.key_proj = nn.Linear(Dz, Datt)
        self.value_proj = nn.Linear(Dz, Dout)

    def forward(self,
                x: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
        Self-attention or cross-attention.
        if x = z, it's self-attention.

        inputs:
            x: N primary seq(s) of representations: N x Tx x Dx
            z: M context seq(s) of representations: M x Tz x Dz
        output:
            y: N repr. with context info          : N x Tx x Dout
        """
        Q = self.query_proj(x)  # N x Tx x Dx -> N x Tx x Datt
        K = self.key_proj(z)    # N x Tz x Dz -> N x Tz x Datt
        V = self.value_proj(z)  # N x Tz x Dz -> N x Tz x Dout
        S = torch.einsum("bxd,bzd->bxz", Q, K)  # -> N x Tx x Tz
        # Masking
        S[:, ~self.mask] = -torch.inf
        # Attention matrix for each pair of sequences
        A = (S/np.sqrt(self.Datt)).softmax(dim=-1)  # N x Tx x Tz
        # y = torch.einsum("bxz,bzo->bxo", A, V)
        y = torch.bmm(A, V)  # N x Tx x Dout
        return y
    
if __name__ == "__main__":
    config = {"T": 7,  # max seq. length
              "vocab_size": 10,
              "embed_dim": 4,
              "mlp_hidden_dim": 3,
              "num_layers": 2,
              "num_heads": 1}
    
    model = Decoder(**config)
    print(model)
    print(f"\nNumber of parameters: {sum(map(lambda x: x.numel(), model.parameters()))}")

    x = torch.randint(config["vocab_size"],size=(10, config["T"]))
    y = model(x)