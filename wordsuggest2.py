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

print(sorted(correct_spelling("çeçek", corpus), key=lambda x: x[1], reverse=True))