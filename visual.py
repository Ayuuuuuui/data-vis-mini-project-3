import streamlit as st
import plotly.express as px
import joblib
import pandas as pd


model = joblib.load("model.joblib")

stopwords = ["ผู้", "ที่", "ซึ่ง", "อัน"]

def tokens_to_features(tokens, i):
  word = tokens[i]

  features = {
    "bias": 1.0,
    "word.word": word,
    "word[:3]": word[:3],
    "word.isspace()": word.isspace(),
    "word.is_stopword()": word in stopwords,
    "word.isdigit()": word.isdigit(),
    "word.islen5": word.isdigit() and len(word) == 5
  }

  if i > 0:
    prevword = tokens[i - 1]
    features.update({
      "-1.word.prevword": prevword,
      "-1.word.isspace()": prevword.isspace(),
      "-1.word.is_stopword()": prevword in stopwords,
      "-1.word.isdigit()": prevword.isdigit(),
    })
  else:
    features["BOS"] = True

  if i < len(tokens) - 1:
    nextword = tokens[i + 1]
    features.update({
      "+1.word.nextword": nextword,
      "+1.word.isspace()": nextword.isspace(),
      "+1.word.is_stopword()": nextword in stopwords,
      "+1.word.isdigit()": nextword.isdigit(),
    })
  else:
    features["EOS"] = True

  return features

def parse(text):
  tokens = text.split()
  features = [tokens_to_features(tokens, i) for i in range(len(tokens))]
  return model.predict([features])[0]


import zipfile
from bs4 import BeautifulSoup
import textwrap
import pandas as pd

zipf = zipfile.ZipFile('khaosod.zip','r')
filenames = zipf.namelist()

corpus_content = list()
for filename in filenames:
  html_file = zipf.open(filename)
  html_string = html_file.read().decode('utf-8')
  html_file.close()

  html_soup = BeautifulSoup(html_string, 'html.parser')
  # print(html_soup.prettify())

  title_box = html_soup.find('h1',class_='udsg__main-title')
  title = title_box.text.strip()
  # print(title)

  content_box = html_soup.find('div', class_='udsg__content')
  content = ''
  for p in content_box.find_all('p'):
    content += p.text
  content = content.replace('\n','').replace('  ','').strip()
  # print(textwrap.fill(content,width=120))

  corpus_content.append({'title':title,
                         'content':content
                         })
zipf.close()

df_corpus = pd.DataFrame(corpus_content)

cont = []
result = []
for i in range(df_corpus.shape[0]):
    content = df_corpus['content'][i]
    cont.append(content.split())
    result.append(parse(content).tolist())

cont = sum(cont,[])
result = sum(result,[])

result_table = pd.DataFrame({'token': cont,
                             'tag': result})

tag_counts = result_table['tag'].value_counts().reset_index()

fig = px.bar(tag_counts, x='tag', y='count', title="Tag Counts", labels={'count': 'Count', 'tag': 'Tag'})

st.title("Visual Analytics for NER")
st.write("Example result from corpus 'Khasod'")

cols = st.columns(2)
with cols[0]:
    st.dataframe(result_table)
with cols[1]:
    st.plotly_chart(fig)

st.title("Corpus Details")

for index, row in df_corpus.iterrows():
  st.subheader(f'Content {index+1}: {row['title']}')
  st.write(row['content'])