import pickle

with open("pre_vocab.pkl","rb") as f:
    i2w = pickle.load(f)
    w2i = pickle.load(f)
    with open("vocab.txt","w") as fo:
        for n in range(len(i2w)):
            fo.write(i2w[n]+"\n")
