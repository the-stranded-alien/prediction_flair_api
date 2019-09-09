import numpy as np 
from flask import Flask, jsonify, request
from sklearn.externals import joblib
import praw 
import re 

app = Flask(__name__)
model = joblib.load("finalized_model.pkl")

@app.route('/')
def home():
    return "<h1>Flair Prediction Model As Microservice (API)</h1>"

@app.route('/predict', methods=['GET'])
def predict():
    reddit = praw.Reddit(client_id = '4olKQ-Up2BCKWQ', client_secret = 'sZsEDlw6NzhL3TpD95p1QMeH81E', user_agent = 'Sahil Gupta', username = 'the_stranded_alien', password = 'strandedalien')
    if 'post' in request.args:
        url = request.args['post']
    else:
        return "Error : No Post URL provided !"
    u = str(url)
    submission = reddit.submission(url=str(url))
    submission.comments.replace_more(limit=None)
    comment = ''
    for top_comments in submission.comments:
        comment = comment + " " + top_comments.body
    title = submission.title
    
    u = u.replace("/"," ").replace("."," ").replace("_", " ").replace("-", " ")
    u = re.sub(r"[0-9]+", "", u)
    new_url = []
    for x in (u.split()):
        if x != "reddit" and x != "comments" and x != "india" and x != "https:" and x != "http:" and len(x) > 4 and x != "www" and x != "com":
            new_url.append(x)
    final_url = " ".join(new_url)
    
    title = title.lower()
    title = re.sub("[^0-9a-z #+_]","",title)
    title = re.sub("[/(){}\[\]\|@,;]", " ", title)
    title = ' '.join(word for word in title.split())
        
    comment = comment.lower()
    comment = re.sub("[^0-9a-z #+_]","",comment)
    comment = re.sub("[/(){}\[\]\|@,;]", " ", comment)
    comment = ' '.join(word for word in comment.split())    
    
    feature = final_url + title + comment
    feature = np.array(feature).reshape((1,))
    prediction = model.predict(feature)
    
    return ' Result : {}'.format(prediction)

if __name__ == "__main__":
    app.run(debug=True)