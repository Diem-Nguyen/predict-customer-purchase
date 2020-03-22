import pandas as pd
import numpy as np
import json
from datetime import date

from plotnine import *
import matplotlib.pyplot as plt

# ============================================================
#  PREPROCEESING RAW DATA
# ============================================================

# split rows and separate columns
def clean_data(df):
    df['transaction_info'] =  df['transaction_info'].str.replace('[', '').str.replace(']', '')
    df['transaction_info'] = df['transaction_info'].map(lambda x: x.replace("'", '''"'''))

    # split rows 
    df = df.set_index(['csn','date']).apply(lambda x : x.str.split('},')).stack().apply(pd.Series).stack().unstack(level=2).reset_index(level=[0,1])

    df = df.reset_index(drop=True)
    df['transaction_info']=df['transaction_info'].map(lambda s: s.replace(' {', '{'))
    df['transaction_info'] = df['transaction_info'].map(lambda s: s+'}' if s[-1:]!= '}' else s )

    # turn it into a json
    df['transaction_info'] = df['transaction_info'].map(lambda x: json.loads(x))

    for col in ['article', 'salesquantity', 'price']:
        df[col] =[df['transaction_info'].iloc[i][col] for i in range(df.shape[0])]
    #df.drop('transaction_info', inplace=True)
    
    # transaction month
    
    df['date'] = df['date'].map(lambda x: pd.to_datetime(x))
    df['month'] = df['date'].map(lambda x: pd.to_datetime(x)).dt.month
    df['revenue'] = df['price'] *  df['salesquantity']
    
    return df


# ============================================================
#  MODEL ENGINEERING
# ============================================================

def extract_RFM(df, month_start):

    month=month_start
    investigated_months = [month, month+1]
    predict_month = [month+2]

    csn_in_2_months = df.loc[df['month'].isin(investigated_months)]['csn'].drop_duplicates().to_list()
    csn_in_predict_month = df.loc[df['month'].isin(predict_month)]['csn'].drop_duplicates().to_list()

    # create traning df with R, F, M
    # True/ false
    df['buy_next_month'] = False
    data = df.loc[(df['csn'].isin(csn_in_2_months)) & (df['month'].isin(investigated_months))]
    data['buy_next_month']  = data.apply(lambda row: True if row['csn'] in csn_in_predict_month else row['buy_next_month'], axis=1)

    # frequency: how many times a customer purchase in the last 2 months
    data_frequency = data[['csn', 'date']].groupby('csn').count().reset_index().rename(columns={'date':'frequency'})

    # moneytary
    data_moneytary = data[['csn', 'revenue']].groupby(['csn']).sum().reset_index().rename(columns={'revenue':'moneytary'})

    # recency
    today = pd.to_datetime(date.today())
    data['recency'] = (today - data['date'])
    data_recency =  data.loc[:, ['csn','recency']].drop_duplicates().groupby('csn').min().reset_index()

    # join df to have R; F; M
    data = data.loc[:, ['csn','buy_next_month', 'month']].drop_duplicates()\
    .merge(data_frequency, on='csn', how='left')\
    .merge(data_moneytary, on='csn', how='left')\
    .merge(data_recency, on='csn', how='left')
     
    return data

def get_data_modelling(df):
    df_all = pd.DataFrame()
    for month in range(df.month.min(), df.month.max()+1):
        if month <= df.month.max() - 2:
            df_each = extract_RFM(df, month_start=month)
            df_each['month_examined']= month+2
            print(f'Month processed: {month}')
        df_all = df_all.append(df_each)

    df_all= df_all.drop_duplicates()    
    return(df_all)


# ============================================================
#  VISUALIZATION
# ============================================================
def plot_bar(df_result, col):
    model_list = df_result.sort_values(by=col)['model'].to_list()    
    a = (ggplot(df_result, 
        aes('model', col)) +
        geom_col(fill='pink')+
        scale_x_discrete(limits=model_list)+
         coord_flip()+
        xlab('Model')+
        ylab(f'{col}') +
        ggtitle(f'{col} among models')+
        theme_bw())
    print(a)
    
def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal 
    [...] # Add axis labels and grid    
    
    
    
     