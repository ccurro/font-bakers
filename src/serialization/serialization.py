import pickle
import sys
import numpy as np
import font_pb2


def openPickle(fn):
    '''
    Takes the filename of a pickle file, reads it, and returns the stored font
    information in a dict
    
    Parameters
    ----------
    fn: the filename of the pickle file you want to read


    Returns
    -------
    The contents of the pickle file. In our application this will typically be a
    dictionary containing the vector font information 
    '''

    f = open(fn, 'rb')
    cur = pickle.load(f)
    return cur


# read the protobuf with the closed curve information
def readSerializedProto(fn):
    '''
    Takes the filename of a serialized protobuf file, reads it, and reformats
    it. Then returns a np array.   
    
    Parameters
    ----------
    fn: the filename of the protobuf file you want to read

    Retuns
    -------
    NP array of either size [len(chars), maxClosedCurves, maxCurvePerGlyph, 3, 2]
    or [len(chars), maxCurvePerGlyph, 3, 2] containing the bezier, depending on 
    the protobuf that was supplied curve information of the given font
    '''
    WITHOUT_COUNTER = 44640
    WITH_COUNTER = 672000  #892800

    print(fn)
    sfont = font_pb2.Font()

    with open(fn, 'rb') as f:
        sfont.ParseFromString(f.read())
    readData = np.frombuffer(sfont.b, dtype=np.float64)

    if (readData.shape == (WITH_COUNTER, )):
        readArr = np.reshape(readData,
                             (sfont.v, sfont.w, sfont.x, sfont.y, sfont.z))
        return readArr
    elif (readData.shape == (WITHOUT_COUNTER, )):
        readArr = np.reshape(readData, (sfont.w, sfont.x, sfont.y, sfont.z))
        return readArr
    else:
        print("Error when reading Protobuf file: ", fn,
              'unknown shape of np array encountered', readData.shape)
        raise
