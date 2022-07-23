import torch
import torch.nn as nn
import torch.nn.functional as F

from reformer_pytorch import LSHSelfAttention
import math


class SentenceEmbedding(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)

	def forward(self, x):
		# x ... [[1,2,3], [4,5,6] ...]
		return self.dropout(self.embedding(x))


class IndexAttentionSort(nn.Module):
	def __init__(self, embedding_size, bias=1.5):
		super().__init__()
		self.model = nn.CosineSimilarity(dim=3, eps=1e-6)
		self.relu  = nn.ReLU()
		self.bias  = bias#nn.Parameter(torch.tensor(bias))

	def forward(self, xs, reference):
		weight_map = self.relu(self.model(xs.unsqueeze(1), reference) * self.bias).mean(2).unsqueeze(1).mT.unsqueeze(3)
		reference = reference.squeeze(1)
		return torch.mul(weight_map, reference).view(xs.shape[0], -1, len(reference[0][0]))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))

class SIAEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen, hidden_size=256, d_ff=256, layer_norm_eps=1e-3, encoder_layer_num=6, n_heads=8, device=torch.device("cpu")):
		super().__init__()
		self.embedding = SentenceEmbedding(vocab_size, d_model, pad_idx, dropout, maxlen)
		self.iattention = IndexAttentionSort(d_model)
		self.device = device
		self.pad_idx = pad_idx
		self.train_source = True
		self.hidden_size = hidden_size

		self.encoder_layers = nn.ModuleList([
			LightEncoder(d_model, hidden_size, d_ff=d_ff, dropout=dropout, layer_norm_eps=layer_norm_eps)
			for _ in range(encoder_layer_num)])

	def forward(self, xs, ys, reference):
		# reference = [u(1), u(2), ... , u(reference_max_line)] * n
		# <=> [[1,2,3], [4,5,6] ...]
		# xs ... [[1,2,3], [4,5,6] ...]
		# ys ... [[1,2,3], [4,5,6] ...]
		reference_embedding = self.make_embedding(reference)

		x = self.make_embedding(xs)
		y = self.make_embedding(ys)

		reference_embedding = self.iattention(x, reference_embedding)
		
		# context informations
		
		x = torch.concat([x, reference_embedding, x], dim=1)


		hidden = torch.zeros(1, x.shape[0], self.hidden_size, device=self.device)

		for encoder_layer in self.encoder_layers:
			x, hidden = encoder_layer(x, hidden)
		return x, y, hidden


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
	def __init__(self, d_model, dropout, d_ff=256, layer_norm_eps=1e-3, hidden_size=256, decoder_layer_num=6, n_heads=8):
		super().__init__()

		self.hidden_size = hidden_size
		self.decoder_layers = nn.ModuleList([
			LightDecoder(d_model, hidden_size, d_ff=d_ff, dropout=dropout, layer_norm_eps=layer_norm_eps)
			for _ in range(decoder_layer_num)])

	def forward(self, x, y, hidden):
		for decoder_layer in self.decoder_layers:
			x, y, hidden = decoder_layer(x, y, hidden)
		return y, hidden

class LightEncoder(nn.Module):
	def __init__(self, d_model, hidden_size, d_ff=512, dropout=0.01, layer_norm_eps=1e-3):
		super().__init__()
		self.hidden_size = hidden_size
		self.model       = nn.GRU(d_model, hidden_size, batch_first=True)
		self.model_dropout = nn.Dropout(dropout)
		self.layer_norm_model = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.ffn = FFN(hidden_size, d_ff)

	def forward(self, i, hidden):
		#i = self.attn(i)
		i, hidden = self.model(i , hidden)
		#hidden    = self.model_dropout(hidden)
		#hidden    = self.layer_norm_model(hidden)
		return i, hidden

class LightDecoder(nn.Module):
	def __init__(self, d_model, hidden_size, d_ff=512, dropout=0.1, layer_norm_eps=1e-3):
		super().__init__()
		self.hidden_size = hidden_size
		self.model       = nn.GRU(d_model, hidden_size, batch_first=True)
		
		self.model_dropout = nn.Dropout(dropout)
		self.layer_norm_model = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.ffn = FFN(hidden_size, d_ff)

	def forward(self, x, y, state):
		# x .. x, i ... y(label), hidden ... hidden layer at x

		y, _ = self.model(y, state)
		return x, y, state

class SIA(nn.Module):
	def __init__(self,
		vocab_size,
		d_model,
		pad_idx,
		maxlen,
		d_ff=512,
		dropout=0.1,
		layer_norm_eps=1e-3,
		encoder_layer_num=6,
		decoder_layer_num=6,
		n_heads=1,
		device=torch.device("cpu")):

		super().__init__()
		self.encoder = SIAEncoder(vocab_size, d_model, pad_idx, dropout, maxlen,
			d_ff=d_ff,
			layer_norm_eps=layer_norm_eps,
			device=device,
			n_heads=n_heads,
			hidden_size=d_model,
			encoder_layer_num=encoder_layer_num)

		self.decoder = SIADecoder(d_model, dropout, d_ff=d_ff, hidden_size=d_model, layer_norm_eps=layer_norm_eps, decoder_layer_num=decoder_layer_num, n_heads=n_heads)
		self.linear  = nn.Linear(d_model, vocab_size)

	def forward(self, x, y, reference):
		x_out, y, hidden  = self.encoder(x, y, reference)
		x_out, hidden     = self.decoder(x_out, y, hidden)
		x_out             = x_out[:, :len(y[0]), :]
		return self.linear(x_out)