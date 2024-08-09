import nltk
import re
import torch
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertModel, BertForMaskedLM

corpus = {}
with open("deneme3.txt", "r", encoding="utf8") as f:
    for line in f:
        (key, val) = line.split()
        corpus[key] = float(val)


def split(word):
    split_words = []
    for i in range(len(word) + 1):
        split_words.append((word[:i], word[i:]))
    return split_words


def delete(word):
    deleted_words = []
    for i, j in split(word):
        if j:
            deleted_words.append(i + j[1:])
    return deleted_words


def swap(word):
    swapped_words = []
    for i, j in split(word):
        if len(j) > 1:
            swapped_words.append(i + j[1] + j[0] + j[2:])
    return swapped_words


def replace(word):
    letters = 'abcçdefgğhıijklmnoöprsştuüvyz'
    replaced_words = []
    for i, j in split(word):
        for c in letters:
            if j:
                replaced_words.append(i + c + j[1:])
    return replaced_words


def insert(word):
    letters = 'abcçdefgğhıijklmnoöprsştuüvyz'
    inserted_words = []
    for i, j in split(word):
        for c in letters:
            inserted_words.append(i + c + j)
    return inserted_words


def edit1(word):
    return set(delete(word) + swap(word) + replace(word) + insert(word))


def edit2(word):
    edited2 = []
    for i in edit1(word):
        for j in edit1(i):
            edited2.append(j)
    return set(edited2)


def correct_spelling(word, corp):
    if word in corp:
        print("in corpus")
        return
    else:
        suggestions = edit1(word) or edit2(word) or [word]
        guess = []
        for i in suggestions:
            if i in corp.keys():
                guess.append(i)
        result = []
        for i in guess:
            result.append((i, corp[i]))
        return result


text2 = open("test.txt", encoding="utf8").read()
rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ',
        '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ',
        '?':' ? ', '*':' * ',
        '(': ' ( ', ')': ' ) '}
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))
text2 = pattern.sub(lambda m: rep[re.escape(m.group(0))], text2)
new_text = ""
list_text = text2.split(" ")
for i in list_text:
    if len(i) > 0 and i[0].islower() and i not in corpus.keys():
        print(i)
        try:
            a = sorted(correct_spelling(i, corpus), key=lambda x: x[1], reverse=True)[0]
            a, b = a[0], a[1]
        except Exception:
            a = i
    else:
        a = i
    new_text = new_text + a + " "
print(new_text)

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