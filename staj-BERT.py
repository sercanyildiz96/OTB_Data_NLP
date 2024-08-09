import nltk
import re
import torch
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertModel, BertForMaskedLM

text = open("test.txt", encoding="utf8").read()
rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ',
        '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ',
        '?':' ? ', '*':' * ',
        '(': ' ( ', ')': ' ) '}
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))
text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
words = open("toplam.txt", encoding="utf8").readlines()
words = [word.strip() for word in words]
liste = text.split(" ")
for i in liste:
    den = i.replace("I", "ı").replace("İ", "i").replace("Ö", "ö").replace("Ü", "ü").replace("Ç", "ç").replace("Ğ","ğ").replace("Ş", "ş").lower()
    if den not in words:
        liste[liste.index(i)] = "[MASK]"
text = liste[0]
for i in range(1, len(liste)):
    text = text + " " + liste[i]
print(text)
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-cased')
model = BertForMaskedLM.from_pretrained('dbmdz/bert-base-turkish-128k-cased')
model.eval()

tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']
segs = [i for i, e in enumerate(tokenized_text) if e == "."]
segments_ids = []
prev = -1
for k, s in enumerate(segs):
    segments_ids = segments_ids + [k] * (s - prev)
    prev = s

segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)


def predict_word(text_original, predictions, MASKIDS):
    for i in range(len(MASKIDS)):
        preds = torch.topk(predictions[0][0, MASKIDS[i]], k=1)
        indices = preds.indices.tolist()
        list1 = tokenizer.convert_ids_to_tokens(indices)
        simmax = 0
        predicted_token = list1[0]
        text_original = text_original.replace('[MASK]', predicted_token, 1)
    return text_original


print(predict_word(text, predictions, MASKIDS))
