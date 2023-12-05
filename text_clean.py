import re
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def clean_data(data):
  text = re.sub(r"\[[0-9]*\]"," ",data)
  text = re.sub('[%s]' % re.escape("""!",-.:;?`"""),' ', text)  # remove punctuations
  text = text.lower() # convert to lower case
  text = re.sub(r'\s+'," ",text)  # remove extra whitespace
  return text
