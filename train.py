import sia
import parser
from pprint import pprint

r = parser.parse(file_path="./dialogs/test_corpus.txt")
pprint(r)
print(parser.get_maxlen())
r = parser.padding(r, parser.get_maxlen())
pprint(r)