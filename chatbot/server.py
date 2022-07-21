# python standard library
import json
import string
import pickle
from datetime import datetime, timedelta
import os
import secrets

from flask.globals import session

# nltk and friends
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

# tensorflow and friends
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# numpy
import numpy as np

# flask and friends
from flask import Flask, render_template, request, redirect, url_for, make_response, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import PasswordField, SubmitField
from wtforms.validators import DataRequired

with open("questions_fw_utf8.json") as data_file:
    data = json.load(data_file)

def remove_punctuation(s):
    cur = ''
    for j in s:
        if j not in string.punctuation:
            cur += j
    return cur

lemmatizer = WordNetLemmatizer()
POS_lookup = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}
def lemmatize(s):
    words = s.split()
    lemmatized_words = []
    for word in words:
        pos_tag = nltk.pos_tag([word])[0][1][0].upper()
        lemmatized_words.append(lemmatizer.lemmatize(word, POS_lookup.get(pos_tag, wordnet.NOUN)))
    return " ".join(lemmatized_words)

stopwords_set = stopwords.words('english')
def remove_stopwords(s):
    words = s.split()
    filtered_words = []
    for word in words:
        if word not in stopwords_set:
            filtered_words.append(word)
    return " ".join(filtered_words)

def preprocess_question(question):
    question = question.lower()
    question = remove_punctuation(question)
    question = lemmatize(question)
    return question

model = load_model("chat_model2")
with open('tokenizer2.pickle', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
with open('label_encoder2.pickle', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET', 'thisisasecret')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
db = SQLAlchemy(app)
CORS(app)

# database models
class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    when = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=True)
    answered = db.Column(db.Boolean, nullable=False, default=False)
    model_version = db.Column(db.String(2), nullable=False)

    def save(self):
        db.session.add(self)
        db.session.commit()

db.create_all()

class LoginForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

sessions = {}

@app.route('/', methods=['GET', 'POST'])
def landing():
    if token := request.cookies.get("token"):
        if sessions.get(token, -1) > int(datetime.now().timestamp()):
            return redirect(url_for('dashboard'))
    form = LoginForm()
    print(form.validate_on_submit())
    if form.validate_on_submit():
        if form.password.data == os.environ.get('SECRET_PASSWORD', "testpasswd"):
            res = make_response(redirect(url_for('dashboard')))
            token = secrets.token_hex(16)
            res.set_cookie("token", token, max_age=60*60)
            sessions[token] = int((datetime.now() + timedelta(hours=1)).timestamp())
            return res
        else:
            flash('Either an incorrect password, or your session expired')
    return render_template('landing.html', title="Login", form=form)

@app.route('/dashboard')
def dashboard():
    if token := request.cookies.get("token"):
        print(token)
        if sessions.get(token, -1) > int(datetime.now().timestamp()):
            page = request.args.get('page', 1, type=int)
            logs = Log.query.order_by(Log.when.desc()).paginate(page=page, per_page=7)
            return render_template('dashboard.html', title="Dashboard", logs=logs)
    flash('Either an incorrect password, or your session expired')
    return redirect(url_for('landing'))

# model constants
sequence_length = 23

@app.route('/chat',methods=['POST'])
def chat():
    if request.method == 'POST':
        raw_inp = request.form['Question']
        inp = preprocess_question(raw_inp)
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]), maxlen=sequence_length))
        tag = label_encoder.inverse_transform([np.argmax(result)])
        for i in data['FAQ']:
            if i['tag'] == tag:
                pred = np.random.choice(i['responses'])
        Log(question=raw_inp, response=pred, answered=True, model_version="v1").save()
        return { "answer": pred }

def vectorize_sentence(s):
    sentence_vector = np.zeros(len(tokenizer.word_index))
    for idx in tokenizer.texts_to_sequences([remove_stopwords(preprocess_question(s))])[0]:
        sentence_vector[idx-1] = 1
    sentence_vector[0] = 0
    return sentence_vector

qns = []
for tag in data['FAQ']:
    for qn in tag['patterns']:
        qns.append(qn)

def get_relevance(s):
    relevance_list = []
    vs = vectorize_sentence(s)
    for qn in qns:
        vq = vectorize_sentence(qn)
        relevance_list.append(cosine_similarity([vs], [vq])[0][0])
    return max(relevance_list)

@app.route('/chatv2',methods=['POST'])
def chat_v2():
    if request.method == 'POST':
        raw_inp = request.form['Question']
        inp = preprocess_question(raw_inp)
        if get_relevance(raw_inp) > 0.5:
            result = model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]), maxlen=sequence_length))
            tag = label_encoder.inverse_transform([np.argmax(result)])
            for i in data['FAQ']:
                if i['tag'] == tag:
                    pred = np.random.choice(i['responses'])
                    Log(question=raw_inp, response=pred, answered=True, model_version="v2").save()
        else:
            pred = "Sorry, I don't know how to answer that."
            Log(question=raw_inp, response=pred, answered=False, model_version="v2").save()

        return { "answer": pred }

if __name__ == '__main__':
    app.run(debug=True)