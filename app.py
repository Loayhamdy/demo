import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from firstmodel import Firstmodel

f = Firstmodel()
# from only_for_API import vectorize_new_instance


app = Flask(__name__)
model = joblib.load('text_classification4.joblib')
model2 = joblib.load('text_classification_ar.joblib')

@app.route('/',methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    X_feature = request.form['Sentence']
    X_new_feature = np.array([X_feature])
    # X_new_final = vectorize_new_instance(X_new_feature)
    
    
    language = request.form['Language'].lower()
    
    if language == 'ar':
        prediction = model2.predict(X_new_feature)
    elif language =="x":
        prediction = f.func()
    else:
        prediction = model.predict(X_new_feature)

    output = prediction[0]

    return render_template('index.html', prediction_text='Sentence is {}'.format(output))
    

    
if __name__ == "__main__":
    app.run(debug=True)