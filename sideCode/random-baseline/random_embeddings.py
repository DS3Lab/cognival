import random

# note: set a random seed value if you want your random embeddings to be reproducible.

words = open('allwords.txt', 'r').readlines()
print("Total words:", len(words))
print("Unique words:", len(list(set(words))))
words = list(set(words))

dimensions = [50, 100, 200, 300, 768, 850, 1024]

for dim in dimensions:
    print("Embedding dimension:", dim)
    outfile = open('random-baseline/random-embeddings-' + str(dim) + '.txt', 'w')
    header = ['x' + str(i) for i in range(dim)]
    print('word', " ".join(header), file=outfile)
    for word in words:
        word = word.strip()
        vector = []
        for x in range(dim):
            val = random.uniform(-1, 1)
            vector.append(val)
        print(word, " ".join(map(str, vector)), file=outfile)
    print("DONE.")
