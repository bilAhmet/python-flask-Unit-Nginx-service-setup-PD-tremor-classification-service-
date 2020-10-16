#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:58:03 2020

@author: ahmet
"""
from flask import Flask,request,jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf

application = Flask(__name__)
CORS(application)



filename = '/home/app/tremor/DecisionTreeClassifier_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#converting the tri-axis data to one dimensional data with two methods

from matplotlib import pyplot as plt
import scipy.signal as sg
from datetime import datetime

def delXYZ(df):
        del df['X']
        del df['Y']
        del df['Z']
        
def del2XYZ(df):
        del df['X2']
        del df['Y2']
        del df['Z2']
        
def cal_rms(df):
    df2=df.copy()
    
    df2['X2']=df2['X']**2
    df2['Y2']=df2['Y']**2
    df2['Z2']=df2['Z']**2
    
    delXYZ(df2)
    df2['ms']=df2.mean(axis = 1, skipna = True)
    df2 = df2.apply(lambda x: np.sqrt(x) if x.name == 'ms' else x)
    df2.rename(columns={'ms':'rms'}, inplace=True)
    
    del2XYZ(df2)
    return df2



def my_plot(df):
    
    df.plot()
    plt.ylabel('magnitude')
    plt.xlabel('Samples')
    plt.title('Loaded Raw Data')
    plt.show()
    

def rem_ol(df,ws):
    df4=df.copy()
    ind=df4.index
    chunk=ws
    bag=[]
    sd_dic={'min':100,'max':0}
    for i in range(0,len(ind)-chunk-1,int(chunk/2)):
        for sub in range(i,i+chunk):
            bag.append(df4.iloc[sub])        
        sd=np.std(bag)
        if sd<sd_dic['min']:
            sd_dic['min']=sd
        if sd>sd_dic['max']:
            sd_dic['max']=sd
        bag=[]
    bag=[]
    for i in range(len(ind)-chunk-1,-1,-int(chunk/2)):
        for j in range(i,((len(df4)-1), (i+chunk))[((i+chunk) <len(df4))]):
          bag.append(df4.iloc[j])
        sd=np.std(bag)
        if (sd< 2*sd_dic['min']) or (sd>0.3*sd_dic['max']):
            df4=df4.drop(df4.index[[i,((len(df4)-1), (i+chunk))[((i+chunk) <len(df4))]]])
        bag=[]
    return df4,sd_dic,sd

def bandPassFilter(signal):
    low= 0.399
    high=0.4
    order=2
    
    b,a=scipy.signal.butter(order,[low,high],'bandpass',analog=True)
    y=filtfilt(b,a,signal,axis=0)
    return (y)

import pandas
from scipy.fftpack import fft
 
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

from scipy.signal import filtfilt
from scipy.signal import welch
#from scipy import stats
import scipy

graph = tf.get_default_graph()


        




@application.route("/")
def index():
        read = """For post request go /tremor and send file as 
    files = {'media': file, 'id':id}
     """
        return read

@application.route("/tremor", methods=["POST"])
def tremor():
        global graph
        with graph.as_default():
            try:
                
                tremorFile = request.files['media']
                df = pandas.read_table(tremorFile, delim_whitespace=True, names=('ts', 'X', 'Y', 'Z'),skiprows=14,index_col='ts',parse_dates=['ts'])
            except:
                msg="couldn't open the file"
                return msg
            try:
                date=df.index
                df2=cal_rms(df)
                
                # Remove outliers by sliding a window witha specified siz along the data frame and exclude 
                # the windows whose standard deviation is either bigger than half 
                # the maximum or lower than double the minimum
                
                df4,dic,sd=rem_ol(df2,60)
                
                date=df4.index
                mydate=[]
                for dat in date:
                    mydate.append(datetime.fromtimestamp(int(dat)/1000))
                #print(your_dt.strftime("%Y-%m-%d %H:%M:%S"))
                df4=df4['rms']
                
                
                
                # We get a triangular window with 60 samples.
                h = sg.get_window('triang', 5)
                # We convolve the signal with this window.
                fil = sg.convolve(df4, h / h.sum(),mode='same')
                # We plot the original signal..
                
                bpFil=bandPassFilter(fil)
                
                sig=pandas.DataFrame()
                sig['ts']=mydate
                sig['D']=bpFil
                sig=sig.set_index('ts')
                
                 
                t_n = 10
                N = 1000
                T = t_n / N
                f_s = 1/T
                 
                f_values, fft_values = get_fft_values(sig, T, N, f_s)
                f_values, psd_values = welch(bpFil, fs=f_s)
                
                meanFft=np.mean(fft_values)
                meanPSD=np.mean(psd_values)
                maxPSD=np.max(psd_values)
                rawSD=np.std(df4)
                finalSD=np.std(bpFil)
                
                features=[meanFft,meanPSD,maxPSD,rawSD,finalSD]
                result=loaded_model.predict([features])
                return jsonify(str(result))
            except:
                        msg="some problem has happened "
                        return msg


