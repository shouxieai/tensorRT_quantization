import numpy as np

def saturate(x, int_max, int_min):
    return np.clip(x,int_min, int_max)

def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min())/(int_max - int_min)
    z = int_max - np.round((x.max()/scale))
    return scale, z

def quant_float_data(x, scale, z, int_max, int_min):
    xq = saturate( np.round(x/scale + z), int_max, int_min)
    return xq

def dequant_data(xq, scale, z):
    x = ((xq - z)*scale).astype('float32')
    return x


if __name__ == '__main__':
    np.random.seed(1)
    data_float32 = np.random.randn(3).astype('float32')
    data_float32[0] = -0.61
    data_float32[1] = -0.52
    data_float32[2] = 1.62
    print("input",data_float32)
    int_max = 255
    int_min = 0
    
    scale, z = scale_z_cal(data_float32, int_max, int_min)
    print("scale and z ",scale, z)
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    print("quant result ",data_int8)    
    data_dequnat_float = dequant_data(data_int8, scale, z)
    print("dequant result ",data_dequnat_float)  

    print('diff',data_dequnat_float- data_float32 )