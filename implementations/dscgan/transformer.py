import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int):
        """ multi-head attention

        Parameters
        ---------
        heads: int
            how many heads
        embedding_size: int
            vector dimension
        """
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size
        self.head_size = self.embedding_size // self.heads
        assert embedding_size % heads == 0
        self.__sqrt_embedding_size = (embedding_size / heads) ** 0.5

        self.queryTrans = nn.Linear(self.embedding_size, self.embedding_size)
        self.keyTrans   = nn.Linear(self.embedding_size, self.embedding_size)
        self.valueTrans = nn.Linear(self.embedding_size, self.embedding_size)
        self.linearOut  = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert query.shape[0] == key.shape[0] == value.shape[0]
        assert key.shape[1] == value.shape[1]
        batch_size = query.shape[0]

        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]

        # Linear Transformation
        query = self.queryTrans(query)
        key   = self.keyTrans(key)
        value = self.valueTrans(value)

        # Reshape to multi heads
        query = query.reshape(batch_size, query_len, self.heads, self.head_size)
        key   = key  .reshape(batch_size, key_len,   self.heads, self.head_size)
        value = value.reshape(batch_size, value_len, self.heads, self.head_size)

        # (QK^T / sqrt(d_k))V
        coff: Tensor = torch.einsum('bqhl,bkhl->bhqk', [query, key])
        attention = torch.softmax(torch.div(coff, self.__sqrt_embedding_size), dim = 3)

        ans = torch.einsum('bhqk,bkhl->bqhl', [attention, value])
        ans = ans.reshape(batch_size, query_len, self.embedding_size)
        ans = self.linearOut(ans)
        return ans


class EncoderBlock(nn.Module):
    def __init__(self, heads: int, embedding_size: int, expansion: int, dropout: float):
        super(EncoderBlock, self).__init__()
        self.heads = heads
        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(dropout)
        self.selfAttention = MultiHeadAttention(self.heads, self.embedding_size)
        self.norm1 = nn.LayerNorm(self.embedding_size)
        self.fcff = nn.Sequential(
                nn.Linear(self.embedding_size, expansion * self.embedding_size),
                nn.ReLU(),
                nn.Linear(expansion * self.embedding_size, self.embedding_size)
                )
        self.norm2 = nn.LayerNorm(self.embedding_size)

    def forward(self, src: Tensor) -> Tensor:
        v = self.selfAttention(src, src, src)
        v = self.dropout(self.norm1(src + v))

        b = self.fcff(v)
        b = self.dropout(self.norm2(b + v))
        return b


class TransformerEncoder(nn.Module):
    def __init__(self, heads: int, embedding_size: int,
                 expansion: int, dropout: float, layers: int):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
                EncoderBlock(
                    heads, 
                    embedding_size, 
                    expansion, 
                    dropout) for _ in range(0,layers)
                ])

    def forward(self, src: Tensor) -> Tensor:
        for layer in self.layers:
            src = layer(src)
        return src

