import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import unitroot_adf

filename = 'data/output_populations_3-1000-0.0001.csv'
filename2 = 'data/output_pops_A.csv'
st = 300
ed = 900
maxlag = 1

# Use module 'pandas'
def read_csv_by_Pandas(file):
    df = (pd.read_csv(file, sep=",", header=0))[st:ed]
    dfIt = df['iteration']
    dfS_1 = df['1']
    dfS_2 = df['2']
    dfS_3 = df['3']
    #dfS_4 = df['4']
    #dfS_5 = df['5']

    print('\n\n1 eats 2?')
    grangercausalitytests(df[['1', '2']], maxlag=[maxlag])
    print('\n\n2 eats 1?')
    grangercausalitytests(df[['2', '1']], maxlag=[maxlag])
    print('\n\n1 eats 3?')
    grangercausalitytests(df[['1', '3']], maxlag=[maxlag])
    print('\n\n3 eats 1?')
    grangercausalitytests(df[['3', '1']], maxlag=[maxlag])
    print('\n\n2 eats 3?')
    grangercausalitytests(df[['2', '3']], maxlag=[maxlag])
    print('\n\n3 eats 2?')
    grangercausalitytests(df[['3', '2']], maxlag=[maxlag])
    #print('\n\n5 eats 3?')
    #grangercausalitytests(df[['5', '3']], maxlag=[50])
    #print('\n\n3 eats 5?')
    #grangercausalitytests(df[['3', '5']], maxlag=[50])
    #print('\n\n1 eats 4?')
    #grangercausalitytests(df[['1', '4']], maxlag=[100])
    #print('\n\n4 eats 1?')
    #grangercausalitytests(df[['4', '1']], maxlag=[100])
    #print('\n\n2 eats 4?')
    #grangercausalitytests(df[['2', '4']], maxlag=[100])
    #print('\n\n4 eats 2?')
    #grangercausalitytests(df[['4', '2']], maxlag=[100])

    ax = plt.gca()

    df.plot(x='iteration', y='1', kind='line', ax=ax)
    df.plot(x='iteration', y='2', kind='line', ax=ax)
    df.plot(x='iteration', y='3', kind='line', ax=ax)
    #df.plot(x='iteration', y='4', kind='line', ax=ax)
    #df.plot(x='iteration', y='5', kind='line', ax=ax)
    plt.show()

def read_sample_data(filename):
    pops = np.genfromtxt(filename, delimiter=',')

    pops = pops[st:,:]

    Nsp = np.shape(pops)[1]

    colours = ['g', 'r', 'b', 'y', 'c', 'k', 'm', 'o']

    for s in range(Nsp):

        plt.plot(pops[:,s], c=colours[s])

    plt.show()


def main():
    read_csv_by_Pandas(filename)
    #read_sample_data(filename2)

if __name__ == "__main__":
    main()