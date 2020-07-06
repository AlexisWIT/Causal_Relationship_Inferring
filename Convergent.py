import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyEDM

def main():
    # read_csv_by_Pandas(filename)
    # read_sample_data(filename2)

    filename = 'data/output_populations_10-5000.csv'
    filename2 = 'data/output_pops_E.csv'
    st = 1000
    ed = 2000

    file = filename
    df = (pd.read_csv(file, sep=",", header=0))[st:ed]

    print(df.head(10))

    df[['1','2','3','4']].plot()
    plt.show()

    # ax = plt.gca()

    # df.plot(x='iteration', y='1', kind='line', ax=ax)
    # df.plot(x='iteration', y='2', kind='line', ax=ax)
    # df.plot(x='iteration', y='3', kind='line', ax=ax)
    # df.plot(x='iteration', y='4', kind='line', ax=ax)
    # df.plot(x='iteration', y='5', kind='line', ax=ax)
    # plt.show()

    pyEDM.CCM(dataFrame=df, E=3, columns="1", target="2", libSizes="10 70 10", showPlot=False, sample=100)

# # Use module 'pandas'
# def read_csv_by_Pandas(file):
#     df = (pd.read_csv(file, sep=",", header=0)).head(st)
#     dfIt = df['iteration']
#     dfS_1 = df['1']
#     dfS_2 = df['2']
#     dfS_3 = df['3']

    

#     # ax = plt.gca()

#     # df.plot(x='iteration', y='1', kind='line', ax=ax)
#     # df.plot(x='iteration', y='2', kind='line', ax=ax)
#     # df.plot(x='iteration', y='3', kind='line', ax=ax)
#     # df.plot(x='iteration', y='4', kind='line', ax=ax)
#     # df.plot(x='iteration', y='5', kind='line', ax=ax)
#     # plt.show()

# def read_sample_data(filename):
#     pops = np.genfromtxt(filename2, delimiter=',')

#     pops = pops[st:,:]

#     Nsp = np.shape(pops)[1]

#     colours = ['g', 'r', 'b', 'y', 'c', 'k', 'm', 'o']

#     for s in range(Nsp):

#         plt.plot(pops[:,s], c=colours[s])

#     plt.show()

if __name__ == "__main__":
    main()