from nltk.corpus import brown

print(len(brown.words()))
raw_brown_corpus = ' '.join(brown.words())
with open("/home/wusimpl/pycharm/project1/src/data/input.txt", "w") as file:
    file.write(raw_brown_corpus)
    file.flush()
