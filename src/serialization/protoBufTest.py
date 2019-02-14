''' 
This file is for validating that the data that was pushed into the protobuf was
saved correctly and is mostly just meant as a tool for anyone who is modifying
the protobuf schema
'''

import pickle
import sys
import numpy as np
import font_pb2
import pkl2proto
import serialization


def testProtoAndPkl(fnPkl, fnProto):
    '''
    Read a pkl file and a protobuf file and ensure that the contents match to
    validate that the serialization did not fail

    Parameters
    ----------
    fnPkl: filename of the font data contained in a pickle file 
    fnProto: filename of the font data contained in the serialized protobuf data
    
    Returns
    -------
    Returns true if all values match otherwise returns false to indicate an
    error in serialized data 
    '''

    fontDic = serialization.openPickle(
        fnPkl)  # open the pickle w font info and return dict
    pklArray = pkl2proto.parseDict(fontDic,
                                   fnPkl)  # parse the data and return np array

    proto = serialization.readSerializedProto(fnProto)
    proof = proto == pklArray  #boolean check to make sure that the arrays match
    try:
        proof = proof.flatten()
    except:
        return False

    for i in proof:
        if i == False:
            return False
    return True


def main():
    filenamepkl = sys.argv[1]  # pull file name passed to it
    # if you are using this script sequentialy with pipes in bash you have to
    # use: xargs -n 2   for python to see only two arguments at a time
    filenameproto = sys.argv[2]
    #print(filenamepkl,len(sys.argv))

    result = testProtoAndPkl(filenamepkl, filenameproto)
    print(result)


if __name__ == '__main__':
    main()
