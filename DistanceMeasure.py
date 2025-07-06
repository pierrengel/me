import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features using mean absolute difference:
    DRxy = (1/k) * sum_i |Rx_i - Ry_i|
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similarity index (float)
    '''
    # ensure numpy arrays
    Rx = np.asarray(Rx, dtype=float)
    Ry = np.asarray(Ry, dtype=float)
    # mean absolute difference
    return float(np.mean(np.abs(Rx - Ry)))


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features using zero-mean correlation distance:
    Dtheta = (1 - corr) * k * 10, where corr = (sum((x-mean)(y-mean))) / sqrt(sum((x-mean)^2)*sum((y-mean)^2))
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similarity index (float)
    '''
    x = np.asarray(Thetax, dtype=float)
    y = np.asarray(Thetay, dtype=float)
    # zero-mean
    xm = x - x.mean()
    ym = y - y.mean()
    # denominator
    denom = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
    if denom == 0:
        return 0.0
    # correlation
    corr = float(np.dot(xm, ym) / denom)
    # distance scaled by number of features
    k = x.size
    return float((1.0 - corr) * k * 10)


# Mean Squared Error (MSE) as additional comparison
def mseDistance(imgA, imgB):
    """
    Computes the mean squared difference between two equally sized images.
    :param imgA: First image (ndarray)
    :param imgB: Second image (ndarray)
    :return: Mean squared error (float)
    """
    # convert to float for computation
    a = np.asarray(imgA, dtype=np.float32)
    b = np.asarray(imgB, dtype=np.float32)
    # compute MSE
    mse = np.mean((a - b) ** 2)
    return float(mse)
