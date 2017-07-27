#Run it inside the TCC folder.
#!/usr/bin/env bash

DATA_DIR=data
mkdir $DATA_DIR

# Download SQuAD
SQUAD_DIR=$DATA_DIR/squad
mkdir $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json


# Download CNN and DailyMail
# Download at: http://cs.nyu.edu/~kcho/DMQA/


# Download GloVe
GLOVE_DIR=$DATA_DIR/glove
mkdir $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download NLTK (for tokenizer)
# Make sure that nltk is installed!
export PATH_TO_NLTK_DATA=$HOME/nltk_data/
wget https://github.com/nltk/nltk_data/archive/gh-pages.zip
unzip gh-pages.zip
mv nltk_data-gh-pages $PATH_TO_NLTK_DATA
# add below code
mv $PATH_TO_NLTK_DATA/nltk_data-gh-pages/packages/* $PATH_TO_NLTK_DATA/
