from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import string, re
from slangClass import slang
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = pickle.load(open('minor_project_model.pkl', 'rb'))
tfidf = pickle.load(open('minor_project_vectorizer.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    # return "Hello world"
    return render_template('index.html')


def preprocess(txt):
    txt = re.sub(r'<.*?>', '', txt)
    txt = re.sub(r'<[^>]+>', '', txt)
    txt = txt.encode("ascii", "ignore").decode()  # removing emojis
    txt = txt.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
    txt = (slang.slang_res(slang(), txt))  # abbreviation handling
    txt = re.sub(r'\d+', '', txt)  # remove digits
    txt = re.sub(r' +', ' ', txt)  # removing extra spaces
    txt = rem_stw(txt)
    txt = stem(txt)
    return txt


def rem_stw(txt):
    return " ".join([w for w in txt.split() if w not in stopwords.words('english')])


def stem(txt):
    return " ".join([PorterStemmer().stem(w) for w in txt.split()])


@app.route('/predict', methods=['POST'])
def prediction():
    result = ""
    input = str(request.json['inp'])

    # Use re.split to split the sentence based on the . and ?
    substrings = re.split(r'[.?!]', input)

    for sentence in substrings:

        if len(sentence)==0:
            continue

        # preprocess
        input_text = preprocess(sentence)
        vectorized = tfidf.transform([input_text]).toarray()[0]

        # prediction
        p = model.predict(np.expand_dims(vectorized, axis=0))
        if p[0] == 1:
            result += sentence + ":         hate\n"
        else:
            result += sentence + ":         not hate\n"

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
