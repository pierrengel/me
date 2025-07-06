import numpy as np
import matplotlib.pyplot as plt
# do not import more modules!


def polarToKart(shape, r, theta):
    '''
    convert polar coordinates with origin in image center to cartesian
    :param shape: shape of the image (height, width)
    :param r: radius from image center
    :param theta: angle in radians
    :return: y, x as integer tuple
    '''
    h, w = shape
    cy = h // 2
    cx = w // 2
    # theta=0→right, theta increases downwards
    y = cy + r * np.sin(theta)
    x = cx + r * np.cos(theta)
    return (int(round(y)), int(round(x)))


def calculateMagnitudeSpectrum(img) -> np.ndarray:
    '''
    compute FFT magnitude spectrum, shift to center, return in decibel
    :param img: 2D grayscale image
    :return: magnitude spectrum in dB, same shape as img
    '''
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    magnitude = np.abs(Fshift)
    eps = 1e-8
    magnitude_db = 20 * np.log10(magnitude + eps)
    return magnitude_db


def extractRingFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    """
    extract k ring features by summing intensities over concentric rings
    :param magnitude_spectrum: 2D array
    :param k: number of rings/features
    :param sampling_steps: angular samples per ring (we will sample at 0, Δθ, 2Δθ, …, π)
    :return: 1D array of length k
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)

    # step exactly π/sampling_steps, include both endpoints 0 and π
    delta_theta = np.pi / sampling_steps
    thetas = np.arange(0, sampling_steps + 1) * delta_theta  # [0, Δθ, …, π]

    features = np.zeros(k, dtype=float)

    for i in range(1, k + 1):
        # fractional radii for this ring
        r_min = (i - 1) * max_radius / k
        r_max = i       * max_radius / k
        r_start = int(np.ceil(r_min))
        r_end   = int(np.floor(r_max))

        total = 0.0
        # sum over each integer radius in [r_start, r_end]
        for r in range(r_start, r_end + 1):
            for theta in thetas:
                y, x = polarToKart((h, w), r, theta)
                if 0 <= y < h and 0 <= x < w:
                    total += magnitude_spectrum[y, x]

        features[i - 1] = total

    return features


def extractFanFeatures(magnitude_spectrum, k, sampling_steps) -> np.ndarray:
    '''
    extract k fan features by summing intensities in k angular sectors
    :param magnitude_spectrum: 2D array
    :param k: number of angular sectors
    :param sampling_steps: number of angular samples per sector
    :return: 1D array of length k
    '''
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)
    fan_width = np.pi / k
    features = np.zeros(k, dtype=float)

    for i in range(1, k + 1):
        theta_start = (i - 1) * fan_width
        theta_end   = i       * fan_width
        # evenly spaced angles in [theta_start, theta_end), endpoint=False
        thetas = np.linspace(theta_start, theta_end, sampling_steps, endpoint=False)

        total = 0.0
        # radial step of 2 for efficiency
        for theta in thetas:
            for r in range(0, max_radius + 1, 2):
                y, x = polarToKart((h, w), r, theta)
                if 0 <= y < h and 0 <= x < w:
                    total += magnitude_spectrum[y, x]

        features[i - 1] = total

    return features


def calcuateFourierParameters(img, k, sampling_steps) -> (np.ndarray, np.ndarray):
    '''
    Compute R (rings) and T (fans) features for an input image
    :param img: 2D grayscale image
    :param k: number of features
    :param sampling_steps: sampling resolution
    :return: tuple (R_features, T_features)
    '''
    mag = calculateMagnitudeSpectrum(img)
    R = extractRingFeatures(mag, k, sampling_steps)
    T = extractFanFeatures(mag, k, sampling_steps)
    return (R, T)
