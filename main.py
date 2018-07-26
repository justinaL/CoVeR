from collections import defaultdict
from CoVerModel import CoVeRModel
import string
import spacy
import pandas as pd
import numpy as np

nlp = spacy.load('en_core_web_md')

def get_corpus():
  print('getting corpus data')
  fpath = 'voa/OBV2/obv_words_v2_28-01-2017.tsv'
  df = pd.read_csv(fpath, sep='\t')

  female_speech = df.loc[(df['obc_sex'] == 'f') & (df['obc_hiscoLabel'] != 'Lawyer'),'words']
  male_speech = df.loc[df['obc_sex'] == 'm','words'] # male speech including lawyers
  # lawyer_speech = df.loc[df['obc_hiscoLabel'] == 'Lawyer', 'words']

  return female_speech,male_speech#,lawyer_speech


def get_parsed_corpus(speech,quan):
  print('getting spacy parsed corpus')
  # used spacy for the parsing and get the pos to find the similarities
  parsed_speech = [nlp(speech.iloc[i]) for i in range(quan)]
  # remove punctuations
  parsed_speech_corpus = [[word.text for word in sen if word.text not in string.punctuation] for sen in parsed_speech]

  return parsed_speech_corpus

def main():
  [female_speech, male_speech] = get_corpus()
  parsed_female_corpus = get_parsed_corpus(female_speech,10)
  parsed_male_corpus = get_parsed_corpus(male_speech,10)
  parsed_corpora = [parsed_female_corpus] + [parsed_male_corpus]
  cover = CoVeRModel(embedding_size=10, context_size=10,min_occurrences=5,learning_rate=0.05,batch_size=512)
  cover.iter_corpora(parsed_corpora) # stacked embeddings (dictionary)
  # cover.update_cooccurrence_tensor()
  cover._CoVeRModel__build_graph()