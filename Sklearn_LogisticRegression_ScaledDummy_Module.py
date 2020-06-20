#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler

# create the special class that we are going to use from here on to predict new data

class Absentee_Module():
    
    def __init__(self,model_file,scaler_file):
        
        with open('Logistic_Reg_Model','rb') as model_file, open('My_Scaler','rb') as scaler_file:
            self.model = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    def load_and_clean_data(self, data_path):
        
        # import data,create a copy and drop ID column
        df = pd.read_excel(data_path)
        self.raw_data = df.copy()
        df.drop('ID',1,inplace=True)
        
        # Create seperate dataframe containing dummy values for all available reason
        reason_column = pd.get_dummies(df['Reason for Absence'],drop_first=True)
        
        # split reason_columns into 4 types
        reason_type_1 = reason_column.loc[:,1:14].sum(axis=1)
        reason_type_2 = reason_column.loc[:,15:17].sum(axis=1)
        reason_type_3 = reason_column.loc[:,18:21].sum(axis=1)
        reason_type_4 = reason_column.loc[:,22:].sum(axis=1)
        
        # To avoid multicollinearity, drop the 'Reason for Absence' column from df
        df = df.drop('Reason for Absence',axis=1)
        
        # concatenate df and the 4 types of reason for absence
        df = pd.concat([df,reason_type_1,reason_type_2,reason_type_3,reason_type_4],axis=1)
        
        # Re-order and assign names to the 4 reason type columns
        column_names =[0,1,2,3,'Date', 'Transportation Expense', 'Distance to Work', 'Age',
                       'Daily Work Load Average', 'Body Mass Index', 'Education',
                       'Children', 'Pets']
        df = df[column_names]
        df.rename({0:'Reason_1',1:'Reason_2',2:'Reason_3',3:'Reason_4'},axis=1,inplace=True)
        
        # Convert the 'Date' column into datetime
        df['Date'] = pd.to_datetime(df['Date'],format= '%Y-%m-%d')
        
        # Create a list with month values retrieved from the 'Date' column
        month_list = []
        for i in range(df['Date'].shape[0]):
            x = df['Date'][i].month
            month_list.append(x)
        
        # Insert the values in a new column in df, called 'Month Value'   
        df['Month Value'] = month_list
        
        # Create a new feature called 'Day of the Week'
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
        
        # Re-order the columns in df
        label = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date','Month Value',
                 'Day of the Week','Transportation Expense', 'Distance to Work', 'Age',
                 'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 
                 'Pets']
        df = df[label]
        
        # Drop the 'Date' column from df
        df = df.drop('Date', axis = 1)
        
        # Convert the 'Education' variables into dummy (0s & 1s) by mapping
        # 1 under eduction means 'High School', 2:'Graduate', 3:'Post-graduate',4:'Msc or Phd'
        # Group Education into 2 classes; 1) High School and lower 2) Above high school
        df['Education'] = df['Education'].map({1:0,2:1,3:1,4:1})    
        
        # drop the variables we decide we don't need
        df = df.drop(['Month Value','Distance to Work','Daily Work Load Average'],axis=1)
        
        # we have included this line of code if you want to call the 'preprocessed data'
        self.preprocessed_data = df.copy()
        
        # Standardise imput data
        scaled_input = self.scaler.transform(df)
        
        # we now have our preprocessed and standardised data
        self.data = scaled_input
        
    # Function which outputs 0 or 1 based on our model    
    def predicted_outcome(self):
        if (self.data is not None):
            pred = self.model.predict(self.data)
            return pred
    
    # Function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred_prob = self.model.predict_proba(self.dat)[:,1]
            return pred_prob
    
    # predict the outputs and the probabilities and 
    # add columns with these values at the end of the new data
    def predicted_outcome_table(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.model.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.model.predict(self.data)
            return self.preprocessed_data  
        

