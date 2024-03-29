import sia
import parser
from pprint import pprint
import torch
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

import random
import pickle

device_name = "cpu"
device = torch.device(device_name)

dataset = parser.parse(file_path="./dialogs/corpus_1.txt", N=5000)
dataset = parser.padding(dataset, parser.get_maxlen())

min_utterance = 5

dataset = [article for article in dataset if len(article) >= min_utterance]

m = round(len(dataset) * 0.95)

train_data, valid_data = dataset[:m], dataset[m:]

model_name = "./models/model_cpu_3_.pth"

with open("./models/model_cpu_3_.pickle", "wb") as f:
	pickle.dump(parser.get_dict(), f)

model = sia.SIA(len(parser.get_dict().keys()),
	256,
	0,
	parser.get_maxlen(),
	dropout=0.1,
	d_ff=128,
	encoder_layer_num=0,
	decoder_layer_num=0,
	device=device)

restart_from = False

if restart_from:
	model.load_state_dict(torch.load(model_name))

model.to(device_name)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=parser.get_dict()["<PAD>"])

# Parameters

SAVE_MODEL_EVERY = 50
VALID_EVERY      = 50

mini_batch_f  = lambda x: x // 3 + 1#x * round(x * 0.05) + 1

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

SOS_Tokens = []

for article in train_data[0:10]:
	for u in article:
		SOS_Tokens.append(u[0])

def train_epoch():
	random.shuffle(train_data)
	bar = tqdm(total=len(train_data))
	for data_nth, article in enumerate(train_data):
		bar.update(1)

		if (data_nth + 1) % SAVE_MODEL_EVERY == 0:
			torch.save(model.state_dict(), model_name)

		if (data_nth + 1) % VALID_EVERY == 0:
			valid_model()

		model.train()
		article_ = torch.tensor(article, device=device)

		total_loss = 0.

		if len(article) >= min_utterance:
			mini_batch = mini_batch_f(len(article))

			for i in range(0, len(article), mini_batch):
				if i + mini_batch + 1 <= len(article):
					random.shuffle(SOS_Tokens)

					loss = 0.
					optimizer.zero_grad()

					x = torch.tensor(article[i:i+mini_batch], device=device)
					y = torch.tensor(article[i+1:i+mini_batch+1], device=device)
					
					article_masked = []

					for a in range(mini_batch):
						article_masked.append(article_.view(-1).tolist())
						#article_masked.append(article_[:1+i+a].view(-1).tolist())

					article_masked = torch.tensor(article_masked, device=device)

					out = model(x, y, article_masked)

					_, output_ids = torch.max(out, dim=-1)

					for i in range(y.size()[0]):
						m = max(sep_index(y[i]), sep_index(output_ids[i])) + 1
						loss += criterion(out[i][:m-1], y[i][:m-1])

					loss.backward()
					optimizer.step()

					#for output_id in output_ids:
					#	print("".join([id2word[i.item()] for i in output_id]))

					total_loss += loss.item()

		bar.set_description('Total loss:{}'.format(total_loss/len(article)))


def valid_model():
	def into_str(seq):
		sentence = [id2word[i.item()] for i in seq]
		return "".join([s for s in sentence if s != "<PAD>"])

	model.eval()
	for data_nth, article in enumerate(valid_data):
		if len(article) >= min_utterance:
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
					x = torch.tensor(article[i:i+1], device=device)
					y = torch.tensor(article[i+1:i+2], device=device)
					y_ = torch.tensor([[article[i+1:i+2][0][0]] + [0] * 63], device=device)

					article_masked = []

					for a in range(1):
						article_masked.append(article_.view(-1).tolist())
						#article_masked.append(article_[:1+i+a].view(-1).tolist())

					article_masked = torch.tensor(article_masked, device=device)

					out = model(x, y_, article_masked)
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


for i in range(60):
	train_epoch()
