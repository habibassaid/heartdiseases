from asyncio import Condition
import logging
from os import cpu_count


#import os
#from flask import Flask,jsonify,request
#from webargs.flaskparser import parser
#from.tasks import prediction_task,prediction_task_sync
#from.request_args import request_arguments
#logging.basicConfig( 
#   format='%(asctime)'
                

logger=logging.getLogger(__name__)
#app =Flask(__name__)
#@app.route(/predict)
from flask import Flask, request, jsonify,render_template
# from asyncore import readwrite
from cProfile import run
from logging import debug
import pickle
import numpy as np
from werkzeug.exceptions import BadRequestKeyError
# import joblib
import habiba
import habiba
from habiba import make_predict
app = Flask(__name__)

@app.route('/')
def habiba():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])  
def predict():
    # هنا
    age= request.form.get('age',False)
    gender= request.form.get('gender',False)
    
    Ecg = float(request.form.get('Ecg',False))
    if Ecg=='Normal':
       Ecg=0
    if Ecg=='Having ST-Twave abnormality':
        Ecg=1
    else:
        Ecg=2
    fbs = float(request.form.get('fbs',False))
    if fbs>120:
        fbs=1
    else:
        fbs=0
    heartrate = float(request.form.get('heartrate',False))
    exang = float(request.form.get('exang',False))
    if exang=='yes':
        exang=1
    else:
        exang=0
    cp = float(request.form.get('exang',False))
    if cp=='typical angina':
        cp_1 = 1
        cp_2 = 0
        cp_3 = 0
    elif Condition:
        cp_1 = 0
        cp_2 = 1
        cp_3 = 0
    else:
        cp_1 = 0
        cp_2 = 0
        cp_3 = 1
    oldpeak = float(request.form.get('oldpeak',False))
  

    x = make_predict({'age': [age], 'thalachh': [heartrate], 'oldpeak': [oldpeak], 
                      'sex_1': [gender], 'exng_1':[exang], 'caa_1': [0.1], 'caa_2': [0.0], 
                      'caa_3': [0.0], 'caa_4': [0.0],'cp_1': [cp_1],'cp_2': [cp_2], 'cp_3': [cp_3],
                    'slp_1': [1.0], 'slp_2': [0.0], 'thall_1': [0.0], 'thall_2': [2],'thall_3': [1.0]})
    if x[0]==0:
        res_Val="No heart problem,Your Doing Good"
    else:
        res_Val="Heart Problem,You may have to give extra care towards your health" 
    return render_template('predict.html',prediction_text='Patient has {}'.format(res_Val))

if __name__=='__main__': 
    app.run(debug=True)   
