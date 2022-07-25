import torch
import torch.nn as nn
import torch.nn.functional as F

from reformer_pytorch import LSHAttention, LSHSelfAttention
import math

def create_mask(x, pad_idx, device):
	seq_len = x.size(1)
	mask = x.eq(pad_idx)
	return mask.to(device)

def subsequent_mask(x, device):
	batch_size = x.size(0)
	max_len = x.size(1)
	return torch.tril(torch.ones(batch_size, max_len, max_len)).eq(0).to(device)

class SentenceEmbedding(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, device):
		super().__init__()
		self.d_model = d_model
		self.device  = device
		self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

	def forward(self, x):
		# x ... [[1,2,3], [4,5,6] ...]
		#x = self.embedding(x)
		#hidden = torch.zeros(1, x.shape[0], self.d_model, device=self.device)
		#x, _   = self.lstm(x, hidden)
		#return self.dropout(x)
		return self.embedding(x)

class LSTMEncoding(nn.Module):
	def __init__(self, d_model, dropout, device):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.lstm    = nn.GRU(d_model, d_model, batch_first=True)
		self.device  = device
		self.d_model = d_model

	def forward(self, x):
		hidden = torch.zeros(1, x.shape[0], self.d_model, device=self.device)
		x, _   = self.lstm(x, hidden)
		return self.dropout(x)


class IndexAttentionSort(nn.Module):
	def __init__(self, d_model, dropout, d_ff, bucket_size, n_hashes, layer_norm_eps=1e-7):
		super().__init__()
		self.attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes)
		self.dropout = nn.Dropout(dropout)
		self.ffn = FFN(d_model, d_ff)
		self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

	def forward(self, xs, reference, input_mask, tgt_mask):
		#reference =  reference.view(xs.shape[0], -1, len(reference[0][0]))
		return self.attn(xs, reference, input_mask=input_mask, context_mask=tgt_mask)[0]


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))

class SIAEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen, hidden_size=256, d_ff=256, layer_norm_eps=1e-7, encoder_layer_num=6, bucket_size=32, n_heads=8, device=torch.device("cpu")):
		super().__init__()
		self.embedding    = SentenceEmbedding(vocab_size, d_model, pad_idx, device)
		self.pe           = LSTMEncoding(d_model, dropout, device)
		self.iattention   = IndexAttentionSort(d_model, dropout, d_ff, bucket_size, n_heads)
		self.device       = device
		self.pad_idx      = pad_idx
		self.train_source = True
		self.hidden_size  = hidden_size

		self.encoder_layers = nn.ModuleList([
			LightEncoder(d_model, hidden_size, bucket_size=bucket_size, n_hashes=n_heads, d_ff=d_ff, dropout=dropout, layer_norm_eps=layer_norm_eps)
			for _ in range(encoder_layer_num)])

	def forward(self, xs, ys, reference):
		# reference = [u(1), u(2), ... , u(reference_max_line)] * n
		# <=> [[1,2,3], [4,5,6] ...]
		# xs ... [[1,2,3], [4,5,6] ...]
		# ys ... [[1,2,3], [4,5,6] ...]


		input_mask = create_mask(xs, self.pad_idx, self.device)
		tgt_mask   = create_mask(reference, self.pad_idx, self.device)

		context_input_mask   = create_mask(torch.concat([reference, xs], dim=1), self.pad_idx, self.device)

		reference_embedding = self.make_embedding(reference)
		x                   = self.make_embedding(xs)
		y                   = self.make_embedding(ys)

		ref_attn = self.iattention(x, reference_embedding, input_mask, tgt_mask)
		
		# context informations

		x = torch.concat([ref_attn, x], dim=1)

		x = self.pe(x)
		y = self.pe(y)

		for encoder_layer in self.encoder_layers:
			x = encoder_layer(x, input_mask)
		return x, y, context_input_mask, tgt_mask


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
	def __init__(self, d_model, dropout, d_ff=256, layer_norm_eps=1e-3, decoder_layer_num=1, bucket_size=32, n_heads=4, hidden_size=256):
		super().__init__()

		self.hidden_size = hidden_size
		self.decoder_layers = nn.ModuleList([
			LightDecoder(d_model, hidden_size, d_ff=d_ff, dropout=dropout, layer_norm_eps=layer_norm_eps, bucket_size=bucket_size, n_hashes=n_heads)
			for _ in range(decoder_layer_num)])

	def forward(self, x, y, source_mask, tgt_mask):
		for decoder_layer in self.decoder_layers:
			x, y = decoder_layer(x, y, source_mask, tgt_mask)
		return x, y

class LightEncoder(nn.Module):
	def __init__(self, d_model, hidden_size, d_ff=512, dropout=0.01, bucket_size=32, n_hashes=4, layer_norm_eps=1e-3):
		super().__init__()
		self.hidden_size = hidden_size
		self.attn = LSHSelfAttention(dim=d_model, bucket_size=bucket_size, n_hashes=n_hashes)
		self.model_dropout = nn.Dropout(dropout)
		self.layer_norm_model = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.ffn = FFN(d_model, d_ff)

	def forward(self, i, mask):
		i = self.attn(i, input_mask=mask)
		i = self.model_dropout(i)
		i = self.layer_norm_model(i)
		#i = self.ffn(i)
		return i

class LightDecoder(nn.Module):
	def __init__(self, d_model, hidden_size, d_ff=512, dropout=0.1, layer_norm_eps=1e-3, bucket_size=32, n_hashes=4):
		super().__init__()
		self.hidden_size = hidden_size
		self.attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes)
		
		self.model_dropout = nn.Dropout(dropout)
		self.layer_norm_model = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.ffn = FFN(d_model, d_ff)

	def forward(self, x, y, source_mask, tgt_mask):
		# x .. x, i ... y(label), hidden ... hidden layer at x
		tgt = self.model_dropout(self.layer_norm_model(self.attn(y, y, input_mask=tgt_mask, context_mask=tgt_mask)[0]))
		x   = self.model_dropout(self.layer_norm_model(self.attn(tgt, x, input_mask=tgt_mask, context_mask=source_mask)[0]))
		return x, y

class SIA(nn.Module):
	def __init__(self,
		vocab_size,
		d_model,
		pad_idx,
		maxlen,
		d_ff=512,
		dropout=0.1,
		layer_norm_eps=1e-7,
		encoder_layer_num=6,
		decoder_layer_num=6,
		bucket_size=32,
		n_heads=4,
		device=torch.device("cpu")):

		super().__init__()
		self.encoder = SIAEncoder(vocab_size, d_model, pad_idx, dropout, maxlen,
			d_ff=d_ff,
			layer_norm_eps=layer_norm_eps,
			device=device,
			bucket_size=bucket_size,
			n_heads=n_heads,
			hidden_size=d_model,
			encoder_layer_num=encoder_layer_num)

		self.decoder = SIADecoder(d_model, dropout, d_ff=d_ff, hidden_size=d_model, bucket_size=bucket_size, layer_norm_eps=layer_norm_eps, decoder_layer_num=decoder_layer_num, n_heads=n_heads)
		self.linear  = nn.Linear(d_model, vocab_size)

	def forward(self, x, y, reference):
		x_out, y, smask, tmask = self.encoder(x, y, reference)
		x_out, y               = self.decoder(x_out, y, smask, tmask)
		x_out                  = x_out[:, :len(y[0]), :]
		return self.linear(x_out)