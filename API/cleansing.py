import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

data = pd.read_csv('../data/train_preprocess.tsv', sep='\t', header=None)
alay_dict = pd.read_csv('../data/new_kamusalay.csv', encoding='latin-1', header=None)
stopword_dict = pd.read_csv('../data/stopwordbahasa.csv', header=None)
id_stopword_dict = stopword_dict.rename(columns={0: 'stopword'})

def lowercase(text):
  return text.lower()

def remove_unnecessary_char(text):
  text = re.sub('\n',' ',text) # Remove every '\n'
  text = re.sub('rt',' ',text) # Remove every retweet symbol
  text = re.sub('user',' ',text) # Remove every username
  text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text) # Remove every URL
  text = re.sub('  +', ' ', text) # Remove extra spaces
  return text

def remove_unicode(text):
  text = re.sub(r'\bx[a-fA-F0-9]{2}\b', '', text)
  text = re.sub(r'\bx([a-fA-F0-9]{2})', '', text)
  return text

def remove_nonaplhanumeric(text):
  text = re.sub('[^0-9a-zA-Z]+', ' ', text)
  return text


def normalize_alay(text):
	alay_dict_map = dict(zip(alay_dict[0], alay_dict[1]))
	return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

def remove_stopword(text):
  text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
  text = re.sub('  +', ' ', text) # Remove extra spaces
  text = text.strip()
  return text

def stemming(text):
  return stemmer.stem(text)

def remove_extra_spaces(text):
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text

def preprocess(text):
  text = lowercase(text)
  text = remove_nonaplhanumeric(text)
  text = remove_unnecessary_char(text)
  text = normalize_alay(text)
  text = stemming(text)
  text = remove_stopword(text)
  text = remove_unicode(text)
  text = remove_extra_spaces(text)

  return text

def clean_csv(file_path):
	df = pd.read_csv(file_path, sep='\t', header=None)
	df['text'] = df[0]
	df = df.drop([0], axis=1)
  
	df.drop_duplicates(inplace=True)
	df.dropna(inplace=True)

	df['cleaned_text'] = df['text'].apply(preprocess)
	df['cleaned_text'].head()

	return df