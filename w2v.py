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

def get_vector(sentence):
	return [model.wv.get_vector(w) for w in sentence]

def sentence_range(f, e):
	tmp = []
	for i in range(e):
		tmp += sentences[f+i] + ["。"]
        
	return tmp

def get_data(n, r=3):
	i = sentence_range(n, r)
	return i, sentences[n+r+1], get_vector(i), get_vector(sentences[n+r+1])

x, y, x_vec, y_vec = get_data(4902)
print("".join(x))
print("".join(y))

def calculate_attention(n):
	x, y, x_vec, y_vec = get_data(n) # x=v, y=qk yからxを重み付け

	memory = [[]]
    
	for i in range(len(x_vec)):
		pass