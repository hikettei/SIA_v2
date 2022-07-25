from janome.tokenizer import Tokenizer
from gensim.models import Word2Vec
from tqdm import tqdm

janome_tokenizer = Tokenizer()

def parse():
	sentences = []
	with open("./dialogs/corpus_1.txt", "r") as f:
		contents = f.read().split("\n")
		for content in tqdm(contents[-50000:], leave=False):
			if len(content.split("[SEP]")) > 2:
				_, _, utterance = content.split("[SEP]")
				utterance = [t.surface for t in janome_tokenizer.tokenize(utterance)]
				if len(utterance) > 0:
					sentences.append(utterance)

	return sentences

def train(datas):
	model = Word2Vec(datas, vector_size=128, epochs=10, min_count=0)
	model.save("./models/embedding.w2v")
	return model

def load():
	model = Word2Vec(parse(), vector_size=128, epochs=10, min_count=0)
	model.load("./models/embedding.w2v")
	return model

sentences = parse()
model     = train(sentences)