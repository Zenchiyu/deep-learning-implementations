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
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import warnings
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.backward_compatibility import worker_init_fn
from tqdm import tqdm


warnings.filterwarnings("ignore")  # TODO: remove this, worker_init_fn

class BPETokenizer:
    # Byte-Pair Encoding Tokenizer
    def __init__(self,
                 vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.corpus = None

    def train(self,
              corpus: list[str],
              special_tokens: list[str]=["<unk>", "<eos>", "<pad>", " "]) -> None:
        pbar = tqdm(total=self.vocab_size)

        # Normalize & pre-tokenize
        self.corpus = list(map(str.lower, corpus))  # lower-case each string
        self.corpus = "".join(self.corpus).replace("\n", "<eos>").split()# + [" "]
        self.special_tokens = special_tokens

        unique_words = list(set(self.corpus).difference(set(special_tokens)))
        # Corpus made of initially made of single-character tokens
        helper_list = list(map(list, unique_words))  # will be used to merge tokens
        unique_characters = set.union(*map(set, helper_list))

        # Iteratively build the vocabulary up to the desired vocab_size
        self.unique_tokens = unique_characters
        self.merge_rules = {}  # ordered
        pbar.update(len(self.unique_tokens))

        while len(self.unique_tokens) < self.vocab_size-len(special_tokens):
            pair_frequencies = {}
            best_freq = 0
            for tokens in helper_list:
                # Slide through the word and count the pairs
                for pair in zip(tokens[:-1], tokens[1:]):
                    if pair in pair_frequencies:
                        pair_frequencies[pair] += 1
                    else:
                        pair_frequencies[pair] = 0
                    
                    # Track pair with highest freq
                    if pair_frequencies[pair] > best_freq:
                        best_pair = pair
                        best_freq = pair_frequencies[pair]
            
            # Merge the best pair to get the new token
            new_token = "".join(best_pair)
            self.merge_rules[best_pair] = new_token

            # Modify tmp
            for tokens in helper_list:
                # Slide through the word and count the pairs
                j = 0
                while j < len(tokens)-1:
                    if (tokens[j], tokens[j+1]) == best_pair:
                        tokens[j] = new_token
                        tokens.pop(j+1)
                    j += 1
            
            self.unique_tokens = self.unique_tokens.union({new_token})
            pbar.update(1)
        pbar.update(len(special_tokens))
        pbar.close()
        self.unique_tokens = special_tokens + list(self.unique_tokens)  # explicitly fixes the order although dictionaries are ordered
        self.token2id = {token: id for id, token in enumerate(self.unique_tokens)}
        self.id2token = self.unique_tokens

    def encode(self,
               texts: list[str],
               T: int,
               return_seq_lengths: bool=False) -> torch.Tensor:
        encoding = torch.empty(len(texts), T, dtype=torch.long)
        num_tokens_before_padding = []
        # pre-process the text
        for i, text in enumerate(texts):
            words_no_space = "".join(text).replace("\n", "<eos>").split()
            # Add back the spaces
            words = sum(map(list, zip(words_no_space, [" "]*len(words_no_space))), [])
            words = words[:-1]
            words = [str.lower(el) for el in words]
            
            # List of list of characters except for special tokens
            # Will contain all the tokens, including the special tokens
            helper_list = [list(word) if word not in self.special_tokens else "S" for word in words]
            special = [word for word in words if word in self.special_tokens]
            
            # Apply the merge rules to get tokens
            for (pair_rule, merged) in self.merge_rules.items():
                j = 0
                for tokens in helper_list:
                    # Slide through the word and apply the rule
                    j = 0
                    while j < len(tokens)-1:
                        if pair_rule == (tokens[j], tokens[j+1]):
                            tokens[j] = merged
                            tokens.pop(j+1)
                        j += 1
            # Put back the special tokens in a FIFO manner
            for pos_s, el in enumerate(helper_list):
                if el == "S":
                    helper_list[pos_s] = [special.pop(0)]
            
            # Merge the list of lists of chars into a list of tokens
            tokens = sum(helper_list, [])
            # List of token ids
            ids = list(map(lambda token: self.token2id[token], tokens))
            num_tokens_before_padding.append(len(ids))
            # Pad with token id of <pad>
            pad_id = self.token2id["<pad>"]
            if len(ids) < T:
                ids += [pad_id]*(T-len(ids))
            else:
                # TODO: chunk correctly
                ids = ids[:T]
            encoding[i] = torch.tensor(ids)
        
        if return_seq_lengths:
            return encoding, num_tokens_before_padding
        return encoding

    def decode(self,
               x: torch.tensor) -> list[str]:
        get_tokens_id_seq = lambda id_seq: "".join(list(map(lambda id: self.id2token[id], id_seq)))
        tokens = list(map(get_tokens_id_seq, x))
        return tokens
    
class GPT:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 tokenizer: BPETokenizer,
                 device: str,
                 T: int) -> None:
        self.model, self.criterion = model, nn.NLLLoss()
        self.optimizer = optimizer
        self.device = device

        self.model.to(device=device)
        self.criterion.to(device=device)

        self.tokenizer = tokenizer
        self.T = T  # max sequence length
    
    def train(self,
              train_loader: DataLoader,
              nb_epochs: int=10):
        
        for e in tqdm(range(nb_epochs)):
            for X in train_loader:
                # The sequences are padded
                X = self.tokenizer.encode(X, self.T)
                X = X.to(device=device)
                # Predict log-proba(next token|past)
                output = self.model(X)[:, :-1]
                # Negative log-likelihood: NNL
                # TODO: correctly mask things
                loss = self.criterion(output.permute(0, 2, 1),  # -> vocab in second dim
                                      X[:, 1:])  # shift because future token
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # TODO: remove this
            print(self.inference(["I am"], num_gen=50))
            
    def inference(self,
                  prompts: list[str],
                  num_gen: int,
                  sampling_method: str="stochastic") -> str:
        # TODO: check this method
        # Tokenize prompts
        x, num_tokens_before_padding = self.tokenizer.encode(prompts,
                                                             self.T,
                                                             return_seq_lengths=True)
        for i in range(num_gen):
            # The sequences are padded
            output = self.model(x)
            # p = output[:, num_tokens_before_padding[0]+i]  # TODO: change this
            p = torch.exp(output[:, num_tokens_before_padding[0]+i-1])
            match sampling_method:
                case "deterministic":
                    y = torch.argmax(p, dim=-1)
                case _:  # stochastic by default
                    # Sample the word based on p
                    distr = torch.distributions.categorical.Categorical(probs=p)
                    y = distr.sample()
            x[:, num_tokens_before_padding[0]+i] = y.view(x.shape[0], 1)
        # Decode tokens and ignore the paddings
        y = self.tokenizer.decode(x)
        return list(map(lambda s: s.replace("<pad>", ""), y))
    
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
        # denom = torch.pow(T, 2*torch.arange(embed_dim//2)/embed_dim)[None, :]
        denom = torch.pow(10_000, 2*torch.arange(embed_dim//2)/embed_dim)[None, :]
        pos_embed_matrix = torch.stack([torch.sin(num/denom), torch.cos(num/denom)], dim=-1).flatten(start_dim=-2)  # interleaved
        self.pos_emb = nn.Embedding.from_pretrained(pos_embed_matrix, freeze=True)
        
        ### Masked Multi-Head Attention layers
        mask = torch.tril(torch.ones((T, T), dtype=torch.bool))  # uni-directional self-attention !!
        # TODO: register_buffer
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
        self.norm_add_mhas = nn.ModuleList([PreNormResUnit([embed_dim], mha) for mha in self.mhas])
        self.norm_add_mlps = nn.ModuleList([PreNormResUnit([embed_dim], mlp) for mlp in self.mlps])        
        # TODO: correctly mask things in layer norm due to padding?
        self.final_layernorm = nn.LayerNorm([embed_dim])

        self.logprob_proj = nn.Sequential(
                    nn.Linear(embed_dim, vocab_size, bias=False),
                    nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:
            x: N sequences of T ids: N x T
        intermediate:
            x or e: N sequences of T embeddings: N x T x embed_dim
        output:
            y: log proba distribution over next token
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
        y = self.logprob_proj(x)  # N x T x embed_dim -> N x T x vocab_size
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
        else:
            x_tilde = self.sublayer(x)
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
        self.softmax = nn.Softmax(dim=-1)  # softmax on the scaled-dot prod

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
        SS = S/np.sqrt(self.Datt)  # scaled score
        # Attention matrix for each pair of sequences
        A = self.softmax(SS)  # N x Tx x Tz
        # y = torch.einsum("bxz,bzo->bxo", A, V)
        y = torch.bmm(A, V)  # N x Tx x Dout
        return y


def forward_hook_print_first_sample(module: nn.Module,
                                    input: torch.Tensor,
                                    output: torch.Tensor) -> None:
        print(f"\nModule: {module}\n"+\
            f"Output shape: {output.shape}\nOutput[0]:\n{output[0]}")
        # Also plot a heatmap :p of the top-left square of shape num x num
        with torch.no_grad():
            sns.heatmap(output[0][:num, :num])

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = False

    config = {"T": 100,  # max seq. length
              "vocab_size": 500,
              "embed_dim": 256,
              "mlp_hidden_dim": 256,
              "num_layers": 4,
              "num_heads": 4}
    
    ### Model
    model = Decoder(**config)
    print(model)
    print(f"\nNumber of parameters: {sum(map(lambda x: x.numel(), model.parameters()))}")

    # Add forward-hooks to observe attention scores
    # Here of the first-layer MHA, first head
    # layer = model.norm_add_mhas[0].sublayer.attention_layers[0].softmax
    # handle = layer.register_forward_hook(forward_hook_print_first_sample)

    # x = torch.randint(config["vocab_size"],size=(10, config["T"]))
    # y = model(x)
    
    ### Optimizer
    config["lr"] = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=config["lr"]) 

    ### Dataset: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
    dataset = torchtext.datasets.WikiText2
    dataset_name = str.lower(dataset.__name__)

    config |= {"device": device,
              "batch_size": 128}
    
    train_dp = dataset(f'.data/{dataset_name}', split="train")
    # https://pytorch.org/text/stable/datasets.html#datapipes-warnings
    train_loader = DataLoader(train_dp, num_workers=4,
                              worker_init_fn=worker_init_fn,
                              drop_last=True,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              pin_memory=torch.cuda.is_available())
    
    # XXX: testing stuffs
    # corpus = iter(train_loader).__next__()
    # tokenizer = BPETokenizer(config["vocab_size"])
    # tokenizer.train(corpus)

    # txt = corpus[87]
    # x = tokenizer.encode([txt], config["T"])
    # print(f"Encoding: {[txt]}\n{x}")
    # print("Decoded:\n", tokenizer.decode(x))
    # y = model(x)
    

    ### Tokenizer
    tokenizer = BPETokenizer(config["vocab_size"])
    # TODO: decrease memory footprint
    tokenizer.train(sum(DataLoader(train_dp, num_workers=4), []))
    
    ### Training our model
    gpt = GPT(model, optimizer, tokenizer, device, config["T"])
    if train:
        gpt.train(train_loader)
        
        torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "tokenizer_unique_tokens": tokenizer.unique_tokens,
                "tokenizer_token2id": tokenizer.token2id,
                "tokenizer_id2token": tokenizer.id2token,
                "tokenizer_corpus": tokenizer.corpus,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_merge_rules": tokenizer.merge_rules,
                "tokenizer": tokenizer,
                "config": config}, "checkpoint.pth")
    else:
        checkpoint = torch.load("checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        tokenizer = checkpoint["tokenizer"]
        config = checkpoint["config"]
        gpt = GPT(model, optimizer, tokenizer, device, config["T"])
    # x = tokenizer.encode(["I am here"], config["T"])
    x = tokenizer.encode(["An apple that had been on the tree in the garden for weeks had finally been picked up. <eos>",
                          "The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . <eos> <pad>"],
                          config["T"])
    
    layer = model.norm_add_mhas[0].sublayer.attention_layers[0].softmax
    num = x.shape[1]
    handle = layer.register_forward_hook(forward_hook_print_first_sample)
    y = model(x)
    handle.remove()

    with torch.no_grad():
        print("Stochastic sampling:")
        print(gpt.inference(["I am "], num_gen=64, sampling_method="stochastic"))
        print(gpt.inference(["Studying Deep-Learning is "], num_gen=64, sampling_method="stochastic"))
        print("Deterministic sampling:")
        print(gpt.inference(["I am "], num_gen=64, sampling_method="deterministic"))
        print(gpt.inference(["Studying Deep-Learning is "], num_gen=64, sampling_method="deterministic"))