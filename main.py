from collections import defaultdict
from CoVerModel import CoVeRModel
import string
import spacy
import pandas as pd
import tensorflow as tf

nlp = spacy.load('en_core_web_md')
sess = tf.InteractiveSession()

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

def analysis(cover):
  #### ANALYSIS ####
  sim_scores = {}
  female = defaultdict(list)
  male = defaultdict(list)

  covariates = cover.covariates
  [fmodel, mmodel] = [cover.get_glove_model(model) for model in cover.models]
  common_words = list(set(fmodel.words).intersection(mmodel.words))
  common_words = sorted(common_words)

  for word in fmodel.words:
    female[word].append(fmodel.embedding_for(word))

  for word in mmodel.words:
    male[word].append(mmodel.embedding_for(word))

  for word in common_words:
    f_em = fmodel.embedding_for(word)
    f_em = tf.multiply(f_em,covariates[0])

    m_em = mmodel.embedding_for(word)
    m_em = tf.multiply(m_em,covariates[1])

    score = tf.losses.cosine_distance(tf.nn.l2_normalize(f_em,0), tf.nn.l2_normalize(m_em,0), dim=0)
    score = tf.subtract(1.0, score)

    sim_scores[word] = [sess.run(score)]

  return sim_scores,male,female,covariates

def avg(dic):
  avg_dic = {}
  for key, val in dic.items():
    avg = tf.reduce_mean(val)
    avg_dic[key] = [sess.run(avg)]

  return avg_dic

def write_file(file_path, dic):
  save_file_path = file_path
  print('writing', file_path)
  with open(save_file_path,"w") as save_file:
    for k, v in dic.items():
      save_file.write(str(k) + ' >>> ' + str(v) + '\n\n')


def main():
  [female_speech, male_speech] = get_corpus()
  parsed_female_corpus = get_parsed_corpus(female_speech,500)
  parsed_male_corpus = get_parsed_corpus(male_speech,500)
  parsed_corpora = [parsed_female_corpus] + [parsed_male_corpus]
  cover = CoVeRModel(embedding_size=300,context_size=10,min_occurrences=5,learning_rate=0.05,batch_size=512)
  cover.fit_corpora(parsed_corpora)
  cover.train()

  [similarities,male,female,covariates] = analysis(cover)
  for i in range(4):
    [sim,ma,fe,_] = analysis(cover)
    similarities = {key:value + sim[key] for key,value in similarities.items()}
    male = {key:value + ma[key] for key,value in male.items()}
    female = {key:value + fe[key] for key,value in female.items()}
    i += 1

  avg_sim = avg(similarities)
  similarities = {key:value + avg_sim[key] for key,value in similarities.items()}

  write_file("similarities_500.txt", similarities)
  write_file("female_500.txt", female)
  write_file("male_500.txt", male)

  with open("covariates_500.txt", "w") as f:
    f.write("\n\n".join(map(str, covariates)))

  with open("cover_embeddings_500.txt", "w") as f:
    f.write("\n\n".join(map(str,cover._CoVeRModel__embeddings)))

  print('\nxxxxxxxxxxxxxxxxxxx\n')
  print(avg_sim)
  print('\nxxxxxxxxxxxxxxxxxxx\n')
  print(covariates)