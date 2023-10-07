import numpy as np
import matplotlib.pyplot as plt

def smooth_data(p, eps = 0.0001):
    is_zeros = (p==0).astype(np.float32)
    is_nonzeros = (p!=0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    eps1 = eps*n_zeros/n_nonzeros
    hist = p.astype(np.float32)
    hist  +=  eps*is_zeros + (-eps1)*is_nonzeros
    return hist


def cal_kl(p, q):
    KL = 0.
    for i in range(len(p)):
        KL += p[i]* np.log(p[i]/(q[i]))
    return KL

def kl_test(x, kl_threshold = 0.01 ,size =10):
    y_out = []
    while True:
        y = [ np.random.uniform(1, size+1) for i in range(size)]   
        y /= np.sum(y) 
        kl_result = cal_kl(x, y)
        if kl_result < kl_threshold:
            print(kl_result)
            y_out = y
            plt.plot(x)
            plt.plot(y)
            break
    return y_out     
 
def KL_main():
    np.random.seed(1)
    size = 10
    x = [ np.random.uniform(1, size+1) for i in range(size)]
    x = x / np.sum(x)
    y_out = kl_test(x,kl_threshold = 0.01)
    plt.show()
    print(x, y_out)

if __name__  == '__main__':
    p = [1, 0, 2, 3, 5, 3, 1, 7] 
    bin = 4
    split_p = np.array_split(p, bin)
    q = []
    for arr in split_p:
        avg = np.sum(arr)/ np.count_nonzero(arr)
        for item in arr:
            if item !=0:
                q.append(avg)
                continue
            q.append(0)
    print(q)
    p /= np.sum(p)
    q /= np.sum(q)
    print(p)
    print(q)
    p = smooth_data(p)
    q = smooth_data(q)
    print(p)
    print(q)
    #cal kl
    print(cal_kl(p, q))

