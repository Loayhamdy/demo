import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
# from only_for_API import vectorize_new_instance




app = Flask(__name__)
model = joblib.load('text_classification.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    X_feature = request.form['Sentence']
    X_new_feature = np.array([X_feature])
    # X_new_final = vectorize_new_instance(X_new_feature)
    
    
    prediction = model.predict(X_new_feature)

    output = prediction[0]

    return render_template('index.html', prediction_text='Sentence is {}'.format(output))
    

    
if __name__ == "__main__":
    app.run(debug=True)