#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
from collections import namedtuple


def crossings(x):
    it = iter(x)
    last_x = next(it)
    for j, current_x in enumerate(it):
        idx = j + 1
        if current_x == 0 or np.sign(current_x) != np.sign(last_x):
            yield idx
        last_x = current_x


def find_laplacian_edges(input, sigma, min_magnitude=1e-3):
    gradient = scipy.ndimage.gaussian_filter1d(input, sigma, order=1)
    laplace = scipy.ndimage.gaussian_laplace(input, sigma)
    cc = []
    for idx in crossings(laplace):
        if abs(gradient[idx]) >= min_magnitude:
            cc.append(idx)
    return gradient, laplace, cc


def follow_edge(gradient, edge_index):
    peak_dir = None
    sg = np.sign(gradient[edge_index])
    if sg == 1 or sg == 0:
        # rising edge or valley
        # in case of valley, walking direction is unimportant
        peak_dir = +1  # go right
    elif sg == -1:
        # falling edge
        peak_dir = -1  # go left

    def index_valid(i):
        return 0 <= i < len(gradient)

    peak_idx = edge_index
    valley_idx = edge_index

    while index_valid(peak_idx) and np.sign(gradient[peak_idx]) in (sg, 0):
        peak_idx += peak_dir
    while index_valid(valley_idx) and np.sign(gradient[valley_idx]) in (sg, 0):
        valley_idx -= peak_dir

    # it's possible the last step of peak_idx and/or valley_idx lead them out of the valid index range
    # fix them up here rather then checking at walking time, simplifies the clauses above
    peak_idx = np.clip(peak_idx, 0, len(gradient) - 1)
    valley_idx = np.clip(valley_idx, 0, len(gradient) - 1)

    return peak_idx, valley_idx


def edge_potential_difference(input, gradient, edge_index):
    left_idx, right_idx = follow_edge(gradient, edge_index)
    potential_diff = input[right_idx] - input[left_idx]
    return potential_diff, left_idx, right_idx
