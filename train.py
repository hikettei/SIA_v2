import sia
import parser
from pprint import pprint
import torch
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

train_data = parser.parse(file_path="./dialogs/test_corpus.txt")
train_data = parser.padding(train_data, parser.get_maxlen())

model = sia.SIA(len(parser.get_dict().keys()), 512, 0, parser.get_maxlen(), device=torch.device("cpu"))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=parser.get_dict()["<PAD>"])

# Parameters

min_utterance = 5
mini_batch_f  = lambda x: 2#x * round(x * 0.1) + 1 #x * round(x * 0.05) + 1

id2word = dict(zip(parser.get_dict().values(), parser.get_dict().keys()))
#assert mini_batch < min_utterance, ""

def search_sep(seq):
	i = 0
	F = True
	for m in range(len(seq)):
		if seq[m].item() == parser.get_dict()["<SEP>"]:
			if F:
				F = False
			else:
				break
		else:
			i += 1
	return seq[:i]

def sep_index(seq):
	i = 0
	F = True
	for m in range(len(seq)):
		if seq[m].item() == parser.get_dict()["<SEP>"]:
			if F:
				F = False
			else:
				break
		else:
			i += 1
	return i

def train_epoch():
	model.train()
	for article in tqdm(train_data):
		article_ = torch.tensor(article)
		if len(article) >= min_utterance:
			mini_batch = mini_batch_f(len(article))
			for i in range(0, len(article), mini_batch):
				if i + mini_batch + 1 <= len(article):
					loss = 0.
					optimizer.zero_grad()

					x = torch.tensor(article[i:i+mini_batch])
					y = torch.tensor(article[i+1:i+mini_batch+1])

					out = model(x, y, article_)

					_, output_ids = torch.max(out, dim=-1)


					for i in range(y.size()[0]):
						m = max(sep_index(y[i]), sep_index(output_ids[i])) + 2
						loss += criterion(out[i][:m], y[i][:m])
					loss.backward()
					optimizer.step()

					print("============================")
					print("")
					print("============================")

					for b in range(mini_batch):
						m = max(sep_index(y[b]), sep_index(output_ids[b])) + 2
						sentence = [id2word[i.item()] for i in x[b][:m]]
						print("INPUT :", "".join([s for s in sentence if s != "<PAD>"]))
						print("")
						sentence = [id2word[i.item()] for i in y[b][:m]]
						print("EXCEPT:", "".join([s for s in sentence if s != "<PAD>"]))
						print("")
						sentence = [id2word[i.item()] for i in output_ids[b][:m]]
						print("ANS:", "".join([s for s in sentence if s != "<PAD>"]))
						print("")
					print(loss)

train_epoch()

train_epoch()

train_epoch()