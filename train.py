import sia
import parser
from pprint import pprint
import torch
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

import random

dataset = parser.parse(file_path="./dialogs/test_corpus.txt", N=10000)
dataset = parser.padding(dataset, parser.get_maxlen())

m = round(len(dataset) * 0.95)

random.shuffle(dataset)

train_data, valid_data = dataset[:m], dataset[m:]

model = sia.SIA(len(parser.get_dict().keys()), 512, 0, parser.get_maxlen(), device=torch.device("cpu"))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=parser.get_dict()["<PAD>"])

# Parameters

model_name = "./models/model_1_.pth"
SAVE_MODEL_EVERY = 25
VALID_EVERY      = 100

min_utterance = 5
mini_batch_f  = lambda x: x * round(x * 0.05) + 1

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
	random.shuffle(train_data)
	for data_nth, article in tqdm(enumerate(train_data), total=len(train_data)):

		if data_nth % SAVE_MODEL_EVERY == 0:
			torch.save(model.state_dict(), model_name)

		if data_nth % VALID_EVERY == 0:
			valid_model()

		model.train()
		article_ = torch.tensor(article)

		total_loss = 0.

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

					total_loss += loss.item()

		print("Total loss as nth article: {}".format(total_loss/len(article)))


def valid_model():
	def into_str(seq):
		sentence = [id2word[i.item()] for i in seq]
		return "".join([s for s in sentence if s != "<PAD>"])

	model.eval()
	for data_nth, article in tqdm(enumerate(valid_data), total=len(valid_data)):
		article_ = torch.tensor(article)

		print("==== Contexts ==============================")
		print("")
		print("============================================")

		for i in range(0, len(article)):
			print(into_str(article_[i]))

		print("==== Predicts ==============================")
		print("")

		for i in range(0, len(article)):
			if i + 1 < len(article):
				x = torch.tensor(article[i:i+1])
				y = torch.tensor(article[i+1:i+2])

				out = model(x, x, article_)
				_, output_ids = torch.max(out, dim=-1)

				m = max(sep_index(y[0]), sep_index(output_ids[0])) + 2

				print("Input  :", into_str(x[0][:m]))
				print("")
				print("Output :", into_str(output_ids[0][:m]))
				print("")
				print("Answer :", into_str(y[0][:m]))
				print("")
				print("=======================")
				print("")


for i in range(5):
	train_epoch()
