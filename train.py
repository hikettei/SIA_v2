import sia
import parser
from pprint import pprint
import torch
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

train_data = parser.parse(file_path="./dialogs/test_code.txt")
train_data = parser.padding(train_data, parser.get_maxlen())

model = sia.SIA(len(parser.get_dict().keys()), 100, 0, parser.get_maxlen(), device=torch.device("cpu"))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=parser.get_dict()["<PAD>"])

# Parameters

min_utterance = 6
mini_batch_f  = lambda x: x * round(x * 0.05) + 1

id2word = dict(zip(parser.get_dict().values(), parser.get_dict().keys()))
#assert mini_batch < min_utterance, ""

def train_epoch():
	global c
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

					for i in range(y.size()[0]):
						loss += criterion(out[i], y[i])
					loss.backward()
					optimizer.step()

					_, output_ids = torch.max(out, dim=-1)
					print("==========")
					print("")
					print("==========")

					for b in range(mini_batch):
						sentence = [id2word[i.item()] for i in x[b]]
						print("INPUT :", "".join([s for s in sentence if s != "<PAD>"]))
						print("=====")
						sentence = [id2word[i.item()] for i in y[b]]
						print("EXCEPT:", "".join([s for s in sentence if s != "<PAD>"]))
						print("=====")
						sentence = [id2word[i.item()] for i in output_ids[b]]
						print("ANS:", "".join([s for s in sentence if s != "<PAD>"]))
						print("")
					print(loss)

train_epoch()