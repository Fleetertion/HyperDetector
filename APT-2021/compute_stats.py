# compute_stats.py 只需运行一次
import pandas as pd, numpy as np, pickle
cols = ['Flow Duration', 'Flow Bytes/s', 'Flow Packets/s',
        'Total Fwd Packet','Total Bwd packets',
        'Fwd Packet Length Mean','Bwd Packet Length Mean']
df = pd.read_csv('./data/SCVIC-APT-2021-Training-with-timestamp.csv', usecols=cols)
mu  = df.mean().values
std = df.std().replace(0,1e-6).values
pickle.dump({'mu': mu, 'std': std, 'cols': cols}, open('data/feat_stats.pkl','wb'))