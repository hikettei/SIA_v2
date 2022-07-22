from janome.tokenizer import Tokenizer
from tqdm import tqdm

janome_tokenizer = Tokenizer()

words_dict = {"<PAD>":0, "<SEP>":1}
user_names = {}

max_len = 0

def get_maxlen():
	return max(256, max_len)

def get_dict():
	return words_dict

def translate_into_ids(words):
	global max_len
	l = []
	for w in words:
		if w in words_dict:
			l.append(words_dict[w])
		else:
			words_dict[w] = len(words_dict)
			l.append(words_dict[w])
	max_len = max(max_len, len(l))
	return l

def get_user_id(user_name):
	if user_name in user_names:
		return user_names[user_name]
	else:
		user_names[user_name] = len(user_names)
		return user_names[user_name]

def tokenize_word(sentence, tokenizer="janome"):
	#NAME_ID[SEP]CONTENT[SEP]NAME_ID...
	if tokenizer == "janome":
		tokens = [t.surface for t in janome_tokenizer.tokenize(sentence)]
		return translate_into_ids(tokens)
	elif tokenizer == "Bytelevel":
		pass
	else:
		pass

def read_form(line, contents):
	global max_len
	#
	columns = line.split("[SEP]")

	timestamp, author, content = columns[0], columns[1], columns[2]

	if len(contents) > 1:
		while len(contents[1].split("[SEP]")) <= 1:
			content_n = contents.pop(1)
			content += content_n
	tokenized_word = tokenize_word(content)
	max_len = max(max_len, len(tokenized_word) + 6)
	return timestamp.split("_"), get_user_id(author), tokenized_word


def collect_next_dialog(contents, interval_min):
	global max_len
	if len(contents) == 0:
		return False

	dialogs = []
	last_utterance = read_form(contents.pop(0), contents)

	def tmins(form):
		# Hour * 60 + Mins * 1
		return int(form[0][-2]) * 60 + int(form[0][-1]) * 1

	while len(contents) > 0:
		next_utterance = read_form(contents[0], contents)
		if abs(tmins(last_utterance) - tmins(next_utterance)) <= interval_min:
			dialogs.append(read_form(contents.pop(0), contents))
		else:
			break

	if len(dialogs) > 0:
		return dialogs
	else:
		if len(contents) > 0:
			next_u = read_form(contents[0], contents)
			max_len = max(max_len, len([last_utterance + next_u]))
			return [last_utterance + next_u]
		else:
			return [last_utterance]

def parse(file_path="./dialogs/corpus_1.txt", interval_min=1):

	def collect_data(x):
		return [[i[1]] + [words_dict["<SEP>"]] + i[2] + [words_dict["<SEP>"]] for i in x]

	with open(file_path, "r") as f:
		contents = f.read().split("\n")
		latest   = collect_next_dialog(contents, interval_min)
		dialogs  = [latest]

		total = len(contents) + 1
		bar = tqdm(total = total)

		while latest:
			latest = collect_next_dialog(contents, interval_min)
			if not latest:
				break
			if len(latest) > 0:
				dialogs.append(latest)
				bar.update(len(latest))
			bar.update(1)

		return [collect_data(x) for x in dialogs]


def padding(parsed_data, maxlen):
	train_data = []

	for dialogs in parsed_data:
		tmp = []
		for utterance in dialogs:
			tmp.append(utterance + [words_dict["<PAD>"]] * (maxlen - len(utterance)))
		train_data.append(tmp)
	return train_data
