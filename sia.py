import torch
import torch.nn as nn
import torch.nn.functional as F
from reformer_pytorch import LSHAttention as Ref_LSHAttention
import math

# 隣接する単語だけでAtteentionを計算+Sort
# weightを後で可視化してみる.

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
    	return self.dropout(token_embedding)
        #return self.dropout(torch.concat([token_embedding, self.embedding_pos[: token_embedding.size(0), :].squeeze(1)], dim=1))


class SentenceEmbedding(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen):
		super().__init__()
		self.pe = PositionalEncoding(d_model, dropout, maxlen)
		self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)

	def forward(self, x):
		# x ... [[1,2,3], [4,5,6] ...]
		return self.pe(self.embedding(x))


class IndexAttentionSort(nn.Module):
	def __init__(self, embedding_size, bias=0.1):
		super().__init__()
		self.model = nn.CosineSimilarity(dim=3, eps=1e-6)
		self.relu  = nn.ReLU()
		self.bias  = nn.Parameter(torch.tensor(bias))#nn.Parameter(torch.tensor(bias / embedding_size))

	def forward(self, xs, reference):
		weight_map = self.relu(self.model(xs.unsqueeze(1), reference) - self.bias).mean(2).unsqueeze(1).mT.unsqueeze(3)
		reference = reference.squeeze(1)
		return (weight_map * reference).view(xs.shape[0], -1, len(reference[0][0]))

# Inputs [batch_size, seq_len, d_model]
# Outputs [batch_size, seq_len, d_model]

# if you need use it as self-attention like, for example,
# hint: set submodule == nn.Conv1d(), swap axes (1, 2), before and after of passing this module.
class WithLSHSort(nn.Module):
    def __init__(self,
            d_model=512,
            n_heads=8,
            submodule=nn.Identity(),
            eps=1e-4
            ):
        super(WithLSHSort, self).__init__()
        assert d_model % n_heads == 0, f"d_model must be able to devided by n_heads"
        self.hash = nn.ModuleList([nn.Linear(d_model // n_heads, 2) for _ in range(d_model // n_heads)])
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.mod = submodule
        self.eps = 1e-4

    def forward(self, x):
        # caluclate indexes

        projected = torch.cat([self.hash[n](head) for n, head in zip(range(self.n_heads), torch.split(x, self.d_head, dim=2))], dim=2)
        h_x, h_y = torch.split(projected, self.n_heads, dim=2) # [batch_size, seq_len, nheads] where h_x, h_y
        angles = torch.arctan(h_x / (h_y + self.eps)) # [batch_size, seq_len, n_heads] # calculate angle of vector
        indexes = torch.argsort(angles, 1) # [batch_size, seq_len, n_heads]
        indexes = torch.unsqueeze(indexes, dim=3).expand(-1, -1, -1, self.d_head) # [batch_size, seq_len, n_heads, d_head]
        
        # split heads
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        # sort heads
        x = torch.gather(x, 1, indexes)

        # concatenate heads
        x = x.reshape(x.shape[0], x.shape[1], self.d_model)
                
    
        # call module
        x = self.mod(x)
        
        # split heads
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        
        # scatter
        x = torch.scatter(x ,1, indexes, x)

        # concatenate heads
        x = x.reshape(x.shape[0], x.shape[1], self.d_model)
        return x

# convolution with swap axes.
class Conv1dForLSHSort(nn.Module):
    def __init__(self, d_model, kernel_size, stride, padding, padding_mode='circular', **kwargs):
        super(Conv1dForLSHSort, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride, padding, padding_mode=padding_mode, **kwargs)
    def forward(self, x):
        x = x.swapaxes(1,2)
        x = self.conv(x)
        x = x.swapaxes(1,2)
        return x


# convolution with swap axes.
class Conv1dForLSHSort(nn.Module):
    def __init__(self, d_model, kernel_size, stride, padding, padding_mode='circular', **kwargs):
        super(Conv1dForLSHSort, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride, padding, padding_mode=padding_mode, **kwargs)
    def forward(self, x):
        x = x.swapaxes(1,2)
        x = self.conv(x)
        x = x.swapaxes(1,2)
        return x

class MultiheadAttentionForLSHSort(nn.Module):
    def __init__(self, d_model, segment_size=4, n_heads=8, logger=nn.Identity()):
        super(MultiheadAttentionForLSHSort, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.seg_size = segment_size
        self.proj_qk = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.logger=logger
    def forward(self, x):
        # x shape = [batch_size, seq_len, d_model]
        # convert to [batch_size*seg_size, seq_len//seg_size, d_model]

        # pad
        pad_seq_len = self.seg_size - (x.shape[1] % self.seg_size)
        seq_len = x.shape[1]
        a = (seq_len + pad_seq_len) // self.seg_size
        x = torch.cat([x, x[:, 0:pad_seq_len, :]], dim=1)
        x = torch.cat(torch.chunk(x, a, dim=1), dim=0) # pack to batch
        self.logger(f"Splitted attention {a} blocks, {self.seg_size} tokens per block")
        mask = torch.diag(torch.BoolTensor(self.seg_size)).to(x.device)
        x, _ = self.attn(self.proj_qk(x), self.proj_qk(x), self.proj_v(x), attn_mask=mask)
        x = torch.cat(torch.chunk(x, a, dim=0), dim=1)
        x = x[:, 0:seq_len, :]
        return x


class LSHAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, segment_size=4, logger=nn.Identity()):
        super(LSHAttention, self).__init__()
        self.seq = WithLSHSort(d_model, n_heads, MultiheadAttentionForLSHSort(d_model, segment_size=segment_size, n_heads=n_heads, logger=logger))

    def forward(self, x):
        return self.seq(x)


class LSHConv(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size=3, stride=1, padding=1, padding_mode='circular', groups=None, bias=True):

        super(LSHConv, self).__init__()

        if not groups:
            groups = n_heads
        submodule = Conv1dForLSHSort(d_model, kernel_size, stride, padding, padding_mode, groups=groups, bias=bias)
        self.lsh_module = WithLSHSort(d_model, n_heads, submodule)

    def forward(self,x):
        return self.lsh_module(x)

    def forward(self,x):
        return self.lsh_module(x)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))

class SIAEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen, hidden_size=256, d_ff=256, layer_norm_eps=1e-3, encoder_layer_num=6, name="light", n_heads=8, device=torch.device("cpu")):
		super().__init__()
		self.embedding = SentenceEmbedding(vocab_size, d_model, pad_idx, dropout, maxlen)
		self.iattention = IndexAttentionSort(d_model)
		self.device = device
		self.pad_idx = pad_idx
		self.train_source = True
		self.name = name
		self.hidden_size = hidden_size
		def new_model(name):
			if name == "light":
				return LightEncoder(d_model, hidden_size, d_ff=d_ff, dropout=dropout, layer_norm_eps=layer_norm_eps)
			else:
				return EncoderLayer(d_model, d_ff, dropout, layer_norm_eps, n_heads=n_heads)
		self.encoder_layers = nn.ModuleList([
			new_model(name)
			for _ in range(encoder_layer_num)])

	def forward(self, xs, ys, reference):
		# reference = [u(1), u(2), ... , u(reference_max_line)] * n
		# <=> [[1,2,3], [4,5,6] ...]
		# xs ... [[1,2,3], [4,5,6] ...]
		# ys ... [[1,2,3], [4,5,6] ...]
		reference_embedding = self.make_embedding(reference)

		def make_mask(tensor, l):
			pass

		x = self.make_embedding(xs)
		y = self.make_embedding(ys)

		reference_embedding = self.iattention(x, reference_embedding)
		
		# context informations
		
		x = torch.concat([reference_embedding, x], dim=1)
		#y = torch.concat([y, reference_embedding], dim=1)

		hidden = torch.zeros(1, x.shape[0], self.hidden_size)

		if self.name == "light":
			for encoder_layer in self.encoder_layers:
				x, hidden = encoder_layer(x, hidden)
		else:
			for encoder_layer in self.encoder_layers:
				x = encoder_layer(x)
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
	def __init__(self, d_model, dropout, d_ff=256, layer_norm_eps=1e-3, hidden_size=256, decoder_layer_num=6, n_heads=8, name="light"):
		super().__init__()

		def new_model(name):
			if name == "light":
				return LightDecoder(d_model, hidden_size, d_ff=d_ff, dropout=dropout, layer_norm_eps=layer_norm_eps)
			else:
				return DecoderLayer(d_model, d_ff, dropout, layer_norm_eps, n_heads=n_heads)

		self.hidden_size = hidden_size
		self.name = name
		self.decoder_layers = nn.ModuleList([
			new_model(name)
			for _ in range(decoder_layer_num)])
	def forward(self, x, y, hidden):
		if self.name == "light":
			for decoder_layer in self.decoder_layers:
				x, y, hidden = decoder_layer(x, y, hidden)
		else:
			for decoder_layer in self.decoder_layers:
				y = decoder_layer(x, y)
		return y

class EncoderLayer(nn.Module):
	def __init__(self, d_model, d_ff, dropout_rate, layer_norm_eps, n_heads=8):
		super().__init__()

		self.attention = LSHAttention(d_model, n_heads=n_heads)
		self.dropout_attention = nn.Dropout(dropout_rate)
		self.layer_norm_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)

		self.ffn = FFN(d_model, d_ff)
		self.dropout_ffn = nn.Dropout(dropout_rate)
		self.layer_norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)

	def forward(self, x):
		x = self.layer_norm_attention(
			self.dropout_attention(
					self.attention(x)))

		x = self.layer_norm_ffn(
			self.dropout_ffn(self.ffn(x)))

		return x

class DecoderLayer(nn.Module):
	def __init__(self, d_model, d_ff, dropout_rate, layer_norm_eps, n_heads=8):
		super().__init__()

		self.src_attention = Ref_LSHAttention(n_hashes=n_heads)#nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)
		self.tgt_attention = LSHAttention(d_model, n_heads=n_heads)
		self.ffn           = FFN(d_model, d_ff)

		self.dropout_src_attention = nn.Dropout(dropout_rate)
		self.dropout_tgt_attention = nn.Dropout(dropout_rate)
		self.dropout_ffn           = nn.Dropout(dropout_rate)

		self.layer_norm_src_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.layer_norm_tgt_attention = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.layer_norm_ffn           = nn.LayerNorm(d_model, eps=layer_norm_eps)

	def forward(self, x, y):
		y = self.layer_norm_src_attention(
			y + self.dropout_src_attention(
				self.src_attention(y, x)[0])) #q, k, v=y,x,x | qk, v = x, y

		x = self.layer_norm_tgt_attention(
			y + self.dropout_tgt_attention(
					self.tgt_attention(x)))

		x = self.layer_norm_ffn(
			x + self.dropout_ffn(
				self.ffn(x)))

		return x

class LightEncoder(nn.Module):
	def __init__(self, d_model, hidden_size, d_ff=512, dropout=0.01, layer_norm_eps=1e-3):
		super().__init__()
		self.hidden_size = hidden_size
		self.model       = nn.GRU(d_model, hidden_size, batch_first=True)
		self.model_dropout = nn.Dropout(dropout)
		self.layer_norm_model = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.ffn = FFN(hidden_size, d_ff)

	def forward(self, i, hidden):
		i, hidden = self.model(i, hidden)
		hidden = self.model_dropout(hidden)
		hidden = self.layer_norm_model(hidden)
		return i, self.ffn(hidden)

class LightDecoder(nn.Module):
	def __init__(self, d_model, hidden_size, d_ff=512, dropout=0.01, layer_norm_eps=1e-3):
		super().__init__()
		self.hidden_size = hidden_size
		self.model       = nn.GRU(d_model, hidden_size, batch_first=True)
		self.attention   =  Ref_LSHAttention(n_hashes=8)
		
		self.model_dropout = nn.Dropout(dropout)
		self.layer_norm_model = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.ffn = FFN(hidden_size, d_ff)

	def forward(self, x, i, hidden):
		# x .. x, i ... y(label), hidden ... hidden layer at x

		x, _, _ = self.attention(i, x) #qk, v

		x, hidden = self.model(x, hidden)
		hidden = self.model_dropout(hidden)
		hidden = self.layer_norm_model(hidden)

		return x, i, self.ffn(hidden)

class SIA(nn.Module):
	def __init__(self,
		vocab_size,
		d_model,
		pad_idx,
		maxlen,
		d_ff=256,
		dropout=0.1,
		layer_norm_eps=1e-3,
		encoder_layer_num=3,
		decoder_layer_num=3,
		n_heads=1,
		name="light",
		device=torch.device("cpu")):

		super().__init__()
		self.encoder = SIAEncoder(vocab_size, d_model, pad_idx, dropout, maxlen,
			d_ff=d_ff,
			layer_norm_eps=layer_norm_eps,
			device=device,
			n_heads=n_heads,
			hidden_size=d_model,
			encoder_layer_num=encoder_layer_num,
			name=name)

		self.decoder = SIADecoder(d_model, dropout, d_ff=d_ff, hidden_size=d_model, layer_norm_eps=layer_norm_eps, decoder_layer_num=decoder_layer_num, n_heads=n_heads, name=name)
		self.linear  = nn.Linear(d_model, vocab_size)
	def forward(self, x, y, reference):
		x_out, ys, hidden = self.encoder(x, y, reference)
		x_out             = self.decoder(x_out, ys, hidden)
		x_out             = x_out[:, :len(y[0]), :]
		return self.linear(x_out)