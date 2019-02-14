import pickle
import sys
import numpy as np
import font_pb2
import serialization


def parseDict(d, filename="No filename provided"):
    '''
    Read a Dict containing font curves and return the info in a standardized np array

    Parameters
    ----------
    d: Python Dictionary with key of the chars and the value of a closed curves dict, 
    the closed curve dict has keys of ints up to max num closed curves and lists
    of assosiated bez curves. Each bez curve in the list of bez curves has 3 coordinates
    which are x and y coords of the control points. 
    
    filename: Optional parameter for knowing where the reading of the data failed
    
    
    Returns
    -------
    New NP array of size len(chars), maxClosedCurves, maxCurvePerGlyph, 3, 2
    '''

    chars = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two',
        'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'exclam',
        'numbersign', 'dollar', 'percent', 'ampersand', 'asterisk', 'question',
        'at'
    ]

    # print( set(d.keys()) - set(chars))

    maxCurvePerGlyph = 80
    maxClosedCurves = 20

    # initialize the np array with all ones then multiply by 99
    final = np.ones((len(chars), maxClosedCurves, maxCurvePerGlyph, 3, 2))
    final = final * 999

    for idx, c in enumerate(chars):
        # This will run per glyph
        try:
            # getting number of closed curves in this glyph
            if not d[c]:
                print("Glyph :", c, "has no curves")
                continue
            numClosedCurves = max(
                d[c].keys()) + 1  #add one bc its indexing at 0

            # getting number of bez curves in this glyph
            curvesInGlyph = 0
            for counter in d[c]:  #counting number of bezcurves per glyph
                curvesInGlyph += len(d[c][counter])

            # use sum to check if there are too many curves or closed curves defining a glyph
            if (curvesInGlyph < maxCurvePerGlyph) and (numClosedCurves <
                                                       maxClosedCurves):
                for counter in d[c]:
                    # this runs for every closed curve

                    for idxcurve, curve in enumerate(d[c][counter]):
                        # this will loop through all the bez curves that are in one closed curve
                        # convert to np array to easily drop it into our data structure
                        bez = np.asarray(d[c][counter][idxcurve])

                        try:
                            if bez.shape != (3, 2):
                                print(
                                    "Non standard number of coordinates appeared in font:",
                                    filename, "Glyph", c, 'Curve:',
                                    d[c][counter][curve], "Shape:", bez.shape)
                            final[idx][counter][idxcurve] = bez

                        except:
                            print(
                                'error with wrong num coordinates in a curve',
                            )
                            print(sys.exc_info())

        except:
            # report the error so nohup.out will provide useful info
            print("Failed at reading char: ", c, '\tIn font:', filename)
            print(sys.exc_info())  #debug message

    return final


def serialize(arr, fn):
    '''
    Take the NP array and convert it into a byte stream and store it in a protobuf

    Parameters 
    ----------
    arr: Numpy array of size [len(chars), maxClosedCurves, maxCurvePerGlyph, 3, 2] that contains all the bezier curve data for one font
    
    fn: Parameter for knowing what font pickle that the data corresponds to and
    where to write the data to.
     
    Returns
    -------
    True if nothing failed   
    '''

    dataShape = arr.shape  #store shape of arr for reshaping
    byt = arr.tobytes()  # pack the array

    # init the protobuf
    serializedFont = font_pb2.Font()

    #shape of the file and the byte info
    serializedFont.v = dataShape[0]
    serializedFont.w = dataShape[1]
    serializedFont.x = dataShape[2]
    serializedFont.y = dataShape[3]
    serializedFont.z = dataShape[4]

    serializedFont.b = byt

    fn = fn.rstrip('.p')
    fn = fn.strip('''../../../data/fontPickles/''')

    fn = '''../../../data/fontCCProto/''' + fn + '.myproto'
    # fn = '''fontProto/''' + fn + '.myproto'

    f = open(fn, 'wb')
    f.write(serializedFont.SerializeToString())
    f.close()

    print('successfully saved font to:', fn)
    return True


def main():
    filename = sys.argv[1]  # pull file name passed to it

    fontDic = serialization.openPickle(filename)
    # open the pickle w font info and return dict

    arrForModel = parseDict(fontDic,
                            filename)  # parse the data and return np array
    serialize(arrForModel, filename)


if __name__ == '__main__':
    main()
