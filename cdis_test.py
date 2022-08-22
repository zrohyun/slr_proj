import time
import numpy as np
import scipy as sp
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from slr.data.graphdatagenerator import GraphDataGenerator

from slr.data.ksl.datapath import DataPath

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

if __name__ == "__main__":
    a = np.ones((128,200,137,3))
    batch_size = 32
    class_lim = 100
    x,y = DataPath(class_limit= class_lim).data

    
    st = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.3)

    gtrain = GraphDataGenerator(x_train, y_train, batch_size=batch_size,)
    # print(gtrain[0][0].shape)
    # a,_ = gtrain[0]
    # for i in range(len(gtrain)):
    #     a,_ =gtrain[i]
    a,_ =gtrain[0]
    # print(a.shape[-2:])
    print(a[0,100])
    print(a.max(), a.min())
    exit()
    # d = []
    # for i in range(batch_size):
    #     d.append(np.array([distance.cdist(a[i,w],a[i,w]) for w in range(200)]))
    # d = np.array(d)

    # print(d.shape)
    import matplotlib.pyplot as plt
    b = a[0].reshape(-1)
    print(a[0][a[0].sum((1,2))>0].shape)
    # b = a[0][a[0].sum((1,2))>0]
    print(a[0][a[0].sum((1,2))>.5].shape)
    # plt.hist(b.reshape(-1),bins=100)
    # plt.show()
    print(time.time() - st)
    print(a.shape)
