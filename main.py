import numpy as np
import pandas as pd
from matplotlib.pyplot import bone, pcolor, colorbar, plot, show

from minisom import MiniSom


def get_data(path):
    data = pd.read_csv(path)
    return data

def scale_data(df):
    ret = (df-df.min())/(df.max()-df.min())
    return np.array(ret)

def scale_label(data):
    ret=[]
    for x in data:
        if x=="BENIGN":
            ret.append(0)
        else:
            ret.append(1)
    return ret

file_path='Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
features=[41, 13, 65, 8, 42, 20, 54, 18, 67, 12, 63, 66, 52, 40, 39, 14, 22, 36, 9, 26, 55, 24, 25, 21, 2, 1, 64, 11, 16, 53, 19, 3, 37, 30, 76 ]
if __name__ == '__main__':
    data = get_data(file_path)
    x = data.iloc[:,features]
    y = data.iloc[:, -1]

    x = scale_data(x) #scalling
    y = scale_label(y)
    #print(y)

    som = MiniSom(x=10, y=10, input_len=len(features), sigma=1, learning_rate=.1)
    som.random_weights_init(x)
    som.train_random(data=x, num_iteration=1000,verbose=True)
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers=['o', 'D']
    color=['g', 'r']
    for i, a in enumerate(x):
        w=som.winner(a)
        plot(w[0]+.5,
             w[1]+.5,
             markers[y[i]],
             markeredgecolor= color[y[i]],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=1)
    show()
