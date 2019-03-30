import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(file_path='./', output_name='result', xlabel=None, ylabel=None):
    files= os.listdir(file_path)
    datas = []
    for file in files:
        if os.path.splitext(file)[1] != '.csv': continue
        data = pd.read_csv(os.path.join(file_path, file))
        datas.append(data)
        
    minX = min([len(data) for data in datas])
    data = pd.concat(datas)
    fig = sns.lineplot(x="Step", y="Value", data=data).get_figure()
    plt.xlim(1, minX)
    if xlabel != None: plt.xlabel(xlabel)
    if ylabel != None: plt.ylabel(ylabel)
    fig.savefig(output_name+'.pdf')