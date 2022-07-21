import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])

class SentenceEmbedding(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, dropout, maxlen):
		self.pe = PositionalEncoding(d_model, dropout, maxlen)
		self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)

	def forward(self, x):
		# x ... [[1,2,3], [4,5,6] ...]
		return self.pe(self.embedding(x))

class IndexAttentionSort(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.CosineSimilarity(dim=3, eps=1e-6)
		self.relu  = nn.ReLU()

	def forward(self, xs, reference):
		xs_ = xs.unsqueeze(1)
		weight_map = self.relu(self.model(xs_, reference.unsqueeze(0))).mean(2).unsqueeze(2)
		return weight_map * xs

class SIAEncoder(nn.Module):
	def __init__(self, vocab_size, d_model, pad_idx, reference_max_line, device=torch.device("cpu")):
		self.embedding = SentenceEmbedding(vocab_size, d_model, pad_idx)
		self.iattention = IndexAttentionSort(reference_max_line)
		self.ref_max = reference_max_line
		self.device=device

	def forward(self, xs, ys, reference):
		# reference = [u(1), u(2), ... , u(reference_max_line)]
		# <=> [[1,2,3], [4,5,6] ...]
		# xs ... [[1,2,3], [4,5,6] ...]
		# ys ... [[1,2,3], [4,5,6] ...]
		reference_embedding = self.reference_embedding(reference)

		xs = self.embedding(xs)
		ys = self.embedding(ys)

		reference_embedding = self.iattention(xs, reference_embedding)


	def reference_embedding(self, references):
		L = []
		for r in references:
			L.append(self.embedding(r))
		return torch.tensor(L, device=self.device)

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