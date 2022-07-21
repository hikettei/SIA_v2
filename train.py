import sia
import parser
from pprint import pprint
import torch
from tqdm import tqdm

train_data = parser.parse(file_path="./dialogs/test_corpus.txt")
train_data = parser.padding(train_data, parser.get_maxlen())

model = sia.SIAEncoder(len(parser.get_dict().keys()), 256, 0, 0.1, parser.get_maxlen(), device=torch.device("cpu"))


# Parameters

min_utterance = 5
mini_batch    = 2

assert mini_batch < min_utterance, ""

def train_epoch():
	for article in tqdm(train_data):
		article_ = torch.tensor(article)
		if len(article) >= min_utterance:
			for i in range(0, len(article), mini_batch):
				if i + mini_batch + 1 <= len(article):
					x = torch.tensor(article[i:i+mini_batch])
					y = torch.tensor(article[i+1:i+mini_batch+1])
					model(x, y, article_)

train_epoch()