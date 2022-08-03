import numpy as np
import torch


def to_binary_string(x):
    assert type(x) == torch.Tensor

    # Tensor -> ndarray (bool)
    bin_x = x.cpu().data
    bin_x = np.array(bin_x).astype(np.bool)

    # ndarray -> string
    bin_strings = []
    for i in range(bin_x.shape[0]):
        bin_string = ""
        for j in bin_x[i]:
            if j == True:
                bin_string += "1"
            else:
                bin_string += "0"
        bin_strings.append(bin_string)

    bin_strings = np.array(bin_strings)

    return bin_strings


def to_tensor(x):
    bin_xs = []

    for i in range(x.shape[0]):
        # string -> list
        bin_x = []
        for j in range(len(x[i])):
            if x[i][j] == '1':
                bin_x.append(1)
            else:
                bin_x.append(0)

        # list -> ndarray
        bin_x = np.array(bin_x)
        bin_xs.append(bin_x)
    bin_xs = np.array(bin_xs)

    # ndarray -> torch.Tensor
    bin_xs = torch.from_numpy(bin_xs).float().cuda()

    return bin_xs


def XOR(x, y):
    assert len(x) == len(y)

    XOR_strings = []
    for i in range(len(x)):
        lenth = len(x[i])
        XOR_string = ""
        for j in range(lenth):
            if x[i][j] == y[i][j]:
                XOR_string += "0"
            else:
                XOR_string += "1"
        XOR_strings.append(XOR_string)
    XOR_strings = np.array(XOR_strings)

    return XOR_strings

def binary_to_vector(binary):
    result=[]
    for i in range(binary.shape[0]):
        vec = []
        for xx in range(3):
            tempvec=[]
            for j in range(32):
                tempval=0.0
                for k in range(8):
                    if binary[i][j*8+k+xx*128]=='1':
                        tempval+=2**k
                        #[0,256]->[-6,6]
                tempval=(tempval/256)*12-6
                tempvec.append(tempval)
            tempvec=np.array(tempvec)
            vec.append(tempvec)
        vec=np.array(vec)
        result.append(vec)
    return np.array(result)

def vector_to_binary(vector):
    bin=[]
    vector=vector.cpu().data
    for i in range(vector.shape[0]):
        tempvec=np.array(vector[i])
        resbin=""
        for j in range(32):
            tempval=(np.int)((tempvec[0][j]+6)/12*256)
            tempbin=""
            for k in range(7,-1,-1):
                if(tempval>=(2**k)):
                    tempbin+="1"
                    tempval-=(2**k)
                else:
                    tempbin+="0"
            resbin+=tempbin
        for j in range(32):
            tempval=(np.int)((tempvec[2][j]+2)/4*256)
            tempbin=""
            for k in range(7,-1,-1):
                if(tempval>=(2**k)):
                    tempbin+="1"
                    tempval-=(2**k)
                else:
                    tempbin+="0"
            resbin+=tempbin
        bin.append(resbin)
    bin=np.array(bin)
    return bin



