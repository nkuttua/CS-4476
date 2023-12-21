#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    K = X.shape[0]
    D = feature_width * feature_width
    fvs = np.zeros((K, D))

    w, h = (feature_width // 2), (feature_width // 2)
    pad = np.pad(image_bw, ((h, h), (w, w)), mode='constant')

    x_patch = Y + feature_width // 2
    y_patch = X + feature_width // 2

    for i in range(K):
        x_val = x_patch[i]
        y_val = y_patch[i]
        if feature_width % 2 == 0:
            x_coordinate = x_val - (feature_width // 2 - 1)
            y_coordinate = y_val - (feature_width // 2 - 1)
        else:
            x_coordinate = x_val - (feature_width // 2)
            y_coordinate = y_val - (feature_width // 2)

        patch = pad[x_coordinate: x_coordinate+feature_width, y_coordinate: y_coordinate+feature_width]
        norm_patch = patch / np.linalg.norm(patch)
        fvs[i] = norm_patch.flatten()

    

    
    # raise NotImplementedError('`compute_normalized_patch_descriptors` ' +
    #     'function in`part2_patch_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
