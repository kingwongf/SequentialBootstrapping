import numpy as np, pandas as pd

def getIndMatrix(barIx, t1):
    # Get Ind matrix
    indM= pd.DataFrame(0, index=barIx, columns=range(1,t1.shape[0]))
    for i, (t0,t1) in enumerate(t1.iteritems()): indM.loc[t0:t1,i] = 1.
    return indM.fillna(value=0)

def getAvgUniqueness(indM):
    # Avg uniqueness from ind matrix
    c = indM.sum()        ## still can't run
    u = indM.div(c)
    avgU = u[u>0].mean()
    return avgU

def seqBootstrap(indM, sLength=None):
    # generate a smaple via sequential bootstrap
    if sLength is None: sLength = indM.shape[1]
    phi =[]
    while len(phi)<sLength:
        avgU=pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]] #reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob=avgU/avgU.sum()    # draw prob
        phi+= [np.random.choice(indM.columns, p=prob)]
    return phi

def main():
    t1 = pd.Series([2,3,5] , index=[0,2,4])
    barIx = range(t1.max()+1)
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    phi = [1]
    print('phi', phi)
    print('Standard uniqueness: ', getAvgUniqueness(indM))
    phi = seqBootstrap(indM)
    # print(phi)
    # print('Sequential uniquess: ', getAvgUniqueness(indM[phi]))

main()