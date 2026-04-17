from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500))
    sentiment = db.Column(db.String(50))
    confidence = db.Column(db.String(50))

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    result = ""
    confidence = ""

    if request.method == 'POST':
        text = request.form['text']
        prediction = classifier(text)[0]

        label = prediction['label']
        score = round(prediction['score'] * 100, 2)

        if label == "LABEL_2":
            result = "Positive"
        elif label == "LABEL_1":
            result = "Neutral"
        else:
            result = "Negative"

        confidence = f"{score}%"

        new_entry = History(text=text, sentiment=result, confidence=confidence)
        db.session.add(new_entry)
        db.session.commit()

    return render_template('index.html', result=result, confidence=confidence)

@app.route('/history')
def history():
    records = History.query.all()
    return render_template('history.html', records=records)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)