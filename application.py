from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("Model/scaler.pkl", "rb"))
model = pickle.load(open("Model/randomf_classifier.pkl", "rb"))



## Route for Single data point prediction
@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        age=int(request.form.get("age"))
        sex=int(request.form.get("sex"))
        on_thyroxine=int(request.form.get("on_thyroxine"))
        query_on_thyroxine=int(request.form.get("query_on_thyroxine"))
        on_antithyroid_medication=int(request.form.get("on_antithyroid_medication"))
        sick=int(request.form.get("sick"))
        pregnant=int(request.form.get("pregnant"))
        thyroid_surgery=int(request.form.get("thyroid_surgery"))
        I131_treatment=int(request.form.get("I131_treatment"))
        query_hypothyroid=int(request.form.get("query_hypothyroid"))
        query_hyperthyroid=int(request.form.get("query_hyperthyroid"))
        lithium=int(request.form.get("lithium"))
        goitre=int(request.form.get("goitre"))
        tumor=int(request.form.get("tumor"))
        hypopituitary=int(request.form.get("hypopituitary"))
        psych=int(request.form.get("psych"))
        TSH= float(request.form.get('TSH'))
        T3 = float(request.form.get('T3'))
        TT4 = float(request.form.get('TT4'))
        T4U = float(request.form.get('T4U'))

        new_data=scaler.transform([[age,sex,on_thyroxine,query_on_thyroxine,on_antithyroid_medication,sick,pregnant,thyroid_surgery,I131_treatment,query_hypothyroid,query_hyperthyroid,lithium,goitre,tumor,hypopituitary,psych,TSH,T3,TT4,T4U]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'affected by thyroid'
        else:
            result ='not effected by thyroid'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('frontend.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")