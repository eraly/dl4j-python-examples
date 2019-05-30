from keras.datasets import imdb


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# save indices for later use from java
with open('word_index.txt', 'w') as f:
    for k, v in word_index.items():
        f.write(str(k.encode('utf-8'))[2:-1] + ',' + str(v) + '\n')

