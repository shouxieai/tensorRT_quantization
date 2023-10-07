import numpy as np

def saturate(x):
    return np.clip(x, -127, 127)

def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val/127

def quant_float_data(x, scale):
    xq = np.round(x / scale)
    return saturate(xq)

def dequant_data(xq, scale):
    x = (xq*scale).astype('float32')
    return x

def histgram_range(x):
    hist, range = np.histogram(x, 100)
    total = len(x)
    left  = 0
    right = len(hist) -1 
    limit = 0.99
    while True:
        cover_percent = hist[left:right].sum()/total
        if cover_percent<=limit:
            break

        if hist[left] < hist[right]:
            left+=1
        else:
            right -=1
    
    left_val = range[left]
    right_val = range[right]
    dynamic_range = max(abs(left_val), abs(right_val))
    return dynamic_range/127.


if __name__ == '__main__':
    np.random.seed(1)
    
    data_float32 = np.random.randn(1000).astype('float32')
    print('input ',data_float32)
    scale = scale_cal(data_float32)
    scale2 = histgram_range(data_float32)
    print(scale,scale2 )
    exit(1)
    xq= quant_float_data(data_float32, scale)
    print('quant result ',xq)
    xdq=dequant_data(xq, scale)
    print('dequant result ',xdq)
    print('diff ',xdq-data_float32)