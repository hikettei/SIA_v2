import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding):
        return self.dropout(torch.concat([token_embedding, self.embedding_pos[: token_embedding.size(0), :].squeeze(1)], dim=1))

class SentenceEmbedding(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen):
		super().__init__()
		self.pe = PositionalEncoding(d_model, dropout, maxlen)
		self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)

	def forward(self, x):
		# x ... [[1,2,3], [4,5,6] ...]
		return self.pe(self.embedding(x))

class IndexAttentionSort(nn.Module):
	def __init__(self, bias=0.01):
		super().__init__()
		self.model = nn.CosineSimilarity(dim=3, eps=1e-6)
		self.relu  = nn.ReLU()
		self.bias  = bias

	def forward(self, xs, reference):
		weight_map = self.relu(self.model(xs.unsqueeze(1), reference)).mean(2).unsqueeze(1).mT.unsqueeze(3)
		reference = reference.squeeze(1)
		return (weight_map * reference).view(1, -1, len(reference[0][0]))

class SIAEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen, device=torch.device("cpu")):
		super().__init__()
		self.embedding = SentenceEmbedding(vocab_size, d_model, pad_idx, dropout, maxlen)
		self.iattention = IndexAttentionSort()
		self.device = device

	def forward(self, xs, ys, reference):
		# reference = [u(1), u(2), ... , u(reference_max_line)] * n
		# <=> [[1,2,3], [4,5,6] ...]
		# xs ... [[1,2,3], [4,5,6] ...]
		# ys ... [[1,2,3], [4,5,6] ...]
		reference_embedding = self.make_embedding(reference)

		xs = self.make_embedding(xs)
		ys = self.make_embedding(ys)

		reference_embedding = self.iattention(xs, reference_embedding)
		x = torch.concat([xs, reference_embedding], dim=1)

	def make_embedding(self, references):
		#L = []
		#for r in references:
		#	L.append(self.embedding(r).tolist())

		# [[[1.2.3] [4.5.6]  ...]]
		#return torch.Tensor(L, device=self.device)
		
		L = self.embedding(references[0]).unsqueeze(0)
		for i in range(len(references) - 1):
			L = torch.cat([L, self.embedding(references[i+1]).unsqueeze(0)], dim=0)
		return L

class SIADecoder(nn.Module):
	pass


class MetaLSTM(nn.Module):
	pass

class SIA(nn.Module):
	def __init__(self,
		vocab_size,
		d_model,
		pad_idx,
		reference_max_line=128):

		self.encoder = SIAEncoder(vocab_size, d_model, pad_idx, reference_max_line)
		self.decoder = SIADecoder()

	def forward(self, x, y, reference):
		x_out = self.encoder(x, y, reference)
		x_out = self.decoder(x_out)
		return x_out