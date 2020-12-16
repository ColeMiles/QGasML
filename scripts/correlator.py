import numpy as np


# Pi-preferred
def corrL(s):
    """ Compute the correlator for this pattern:
          ↑ ↓
          ↑ ↑
        and symmetry-equivalent patterns
    """
    res = 0.0
    L = s.shape[-1] 
    for i in range(L-1): 
        for j in range(L-1): 
            res += s[0, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1] 
            res += s[1, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1] 
            res += s[1, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1] 
            res += s[0, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1] 
            res += s[1, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] 
            res += s[0, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] 
            res += s[1, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] 
            res += s[0, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] 
    return res * 2


# Could be either, slightly higher for AS
def corrAFM(s):
    """ Compute the correlator for this pattern:
          ↑ ↓
          ↓ ↑
        and symmetry-equivalent patterns
    """
    res = 0.0
    L = s.shape[-1] 
    for i in range(L-1): 
        for j in range(L-1): 
            res += s[0, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] 
            res += s[1, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] 
    return res * 8


# AS-preferred
def corrFM(s):
    """ Counts this pattern:
            ↑ ↑
            ↑ ↑
        and symmetry-equivalent patterns
    """
    res = 0.0
    L = s.shape[-1]
    for i in range(L-1):
        for j in range(L-1):
            res += s[0, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1]
            res += s[1, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1]
    return res * 8


# AS-preferred
def corrBound(s):
    """ Compute the correlator for this pattern:
          ↓ ↑ 
          ↓ ↑ 
        and symmetry-equivalent patterns
    """
    res = 0.0
    L = s.shape[-1]
    for i in range(L-1):
        for j in range(L-1):
            res += s[0, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1]
            res += s[0, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1]
            res += s[1, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1]
            res += s[1, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1]
    return res * 4


def corrHole(s): 
    res = 0.0 
    L = s.shape[-1] 
    for i in range(L-2): 
        for j in range(L-2): 
            res += s[2, i, j] * s[2, i+2, j]
            res += s[2, i, j] * s[2, i, j+2]
    return res * 8


# Pi-preferred
def corrAFMHole(s):
    """ Counts this pattern:
            ↑ ↓
            ↓ ○
        and symmetry-equivalent patterns
    """
    res = 0.0
    L = s.shape[-1]
    for i in range(L-1):
        for j in range(L-1):
            res += s[0, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[2, i+1, j+1]
            res += s[1, i, j] * s[0, i+1, j] * s[2, i, j+1] * s[1, i+1, j+1]
            res += s[1, i, j] * s[2, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1]
            res += s[2, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1]

            res += s[1, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[2, i+1, j+1]
            res += s[0, i, j] * s[1, i+1, j] * s[2, i, j+1] * s[0, i+1, j+1]
            res += s[0, i, j] * s[2, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1]
            res += s[2, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1]
    return res * 2


# AS-preferred
def corrStringHole(s):
    """ Counts this pattern:
            ↓ ↑
            ↓ ○
        and symmetry-equivalent patterns
    """
    res = 0.0
    L = s.shape[-1]
    for i in range(L-1):
        for j in range(L-1):
            res += s[1, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[2, i+1, j+1]
            res += s[0, i, j] * s[2, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1]
            res += s[2, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1]
            res += s[1, i, j] * s[1, i+1, j] * s[2, i, j+1] * s[0, i+1, j+1]
            res += s[0, i, j] * s[1, i+1, j] * s[2, i, j+1] * s[1, i+1, j+1]
            res += s[1, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[2, i+1, j+1]
            res += s[1, i, j] * s[2, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1]
            res += s[2, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1]

            res += s[0, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[2, i+1, j+1]
            res += s[1, i, j] * s[2, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1]
            res += s[2, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1]
            res += s[0, i, j] * s[0, i+1, j] * s[2, i, j+1] * s[1, i+1, j+1]
            res += s[1, i, j] * s[0, i+1, j] * s[2, i, j+1] * s[0, i+1, j+1]
            res += s[0, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[2, i+1, j+1]
            res += s[0, i, j] * s[2, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1]
            res += s[2, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1]
    return res


def corrFMHole(s):
    """ Counts this pattern:
            ↓ ↓
            ↓ ○
        and symmetry-equivalent patterns
    """
    res = 0.0
    L = s.shape[-1]
    for i in range(L-1):
        for j in range(L-1):
            res += s[1, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[2, i+1, j+1]
            res += s[1, i, j] * s[2, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1]
            res += s[1, i, j] * s[1, i+1, j] * s[2, i, j+1] * s[1, i+1, j+1]
            res += s[2, i, j] * s[1, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1]

            res += s[0, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[2, i+1, j+1]
            res += s[0, i, j] * s[2, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1]
            res += s[0, i, j] * s[0, i+1, j] * s[2, i, j+1] * s[0, i+1, j+1]
            res += s[2, i, j] * s[0, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1]
    return res * 2


# Pi-preferred
def Pattern1(s):
    """ Compute the correlator for this pattern:
            ↓ ↑
          ↓ ↑
        and symmetry-equivalent patterns
    """
    res = 0.0

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[1, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j+1]
            res += s[1, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i, j+2]

            res += s[0, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j+1]
            res += s[0, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i, j+2]
    return res * 4


# Pi-preferred
def Pattern2(s):
    """ Compute the correlator for this pattern:
            ↓ 
          ↓ ↑ ↑
        and symmetry-equivalent patterns
    """
    res = 0.0
    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[1, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j]
            res += s[1, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j+1]
            res += s[0, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i, j+2]
            res += s[0, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j]
            res += s[0, i+1, j] * s[1, i, j+2] * s[0, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j+1]
            res += s[1, i, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[0, i, j+2]

            res += s[0, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j]
            res += s[0, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j+1]
            res += s[1, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i, j+2]
            res += s[1, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j]
            res += s[1, i+1, j] * s[0, i, j+2] * s[1, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j+1]
            res += s[0, i, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[1, i, j+2]
    return res


# Could be either
def Pattern3(s):
    """ Compute the correlator for this pattern:
            ↓ ↓
          ↑ ↑
        and symmetry-equivalent patterns
    """
    res = 0.0

    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[0, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j+1]
            res += s[0, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[1, i, j+2]

            res += s[1, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j+1]
            res += s[1, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[0, i, j+2]
    return res * 4


def Pattern4(s):
    """ Compute the correlator for this pattern:
              ↓
          ○ ↑ ↓
        and symmetry-equivalent patterns
    """
    res = 0.0

    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[2, i, j] * s[0, i+1, j] * s[1, i+2, j] * s[1, i+2, j+1]
            res += s[2, i+1, j] * s[0, i+1, j+1] * s[1, i+1, j+2] * s[1, i, j+2]
            res += s[1, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[2, i+2, j+1]
            res += s[1, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[2, i, j+2]
            res += s[1, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[2, i+2, j]
            res += s[1, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+1, j+2]
            res += s[2, i, j+1] * s[0, i+1, j+1] * s[1, i+2, j] * s[1, i+2, j+1]
            res += s[2, i, j] * s[0, i, j+1] * s[1, i, j+2] * s[1, i+1, j+2]

            res += s[2, i, j] * s[1, i+1, j] * s[0, i+2, j] * s[0, i+2, j+1]
            res += s[2, i+1, j] * s[1, i+1, j+1] * s[0, i+1, j+2] * s[0, i, j+2]
            res += s[0, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[2, i+2, j+1]
            res += s[0, i, j] * s[0, i+1, j] * s[1, i, j+1] * s[2, i, j+2]
            res += s[0, i, j] * s[1, i+1, j] * s[0, i, j+1] * s[2, i+2, j]
            res += s[0, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+1, j+2]
            res += s[2, i, j+1] * s[1, i+1, j+1] * s[0, i+2, j] * s[0, i+2, j+1]
            res += s[2, i, j] * s[1, i, j+1] * s[0, i, j+2] * s[0, i+1, j+2]
    return res


# Could be either, slightly pi-preferred
def Pattern5(s):
    """ Compute the correlator for this pattern:
            ↓ ↑
          ↑ ↑ 
        and symmetry-equivalent patterns
    """
    res = 0.0

    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[0, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j+1]
            res += s[0, i+1, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[0, i, j+2]
            res += s[0, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j+1]
            res += s[0, i+1, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[0, i, j+2]
            res += s[0, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j]
            res += s[0, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j]

            res += s[1, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j+1]
            res += s[1, i+1, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[1, i, j+2]
            res += s[1, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j+1]
            res += s[1, i+1, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[1, i, j+2]
            res += s[1, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j]
            res += s[1, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j]
    return res


# AS-biased
def Pattern6(s):
    """ Compute the correlator for this pattern:
            ↑ ↓
          ↑ ↑ 
        and symmetry-equivalent patterns
    """
    res = 0.0

    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[0, i, j] * s[0, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j+1]
            res += s[0, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1] * s[1, i, j+2]
            res += s[1, i, j] * s[0, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j+1]
            res += s[1, i+1, j] * s[0, i, j+1] * s[0, i+1, j+1] * s[0, i, j+2]
            res += s[1, i, j] * s[0, i, j+1] * s[0, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[0, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j]
            res += s[0, i, j] * s[0, i, j+1] * s[0, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[0, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j]

            res += s[1, i, j] * s[1, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j+1]
            res += s[1, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1] * s[0, i, j+2]
            res += s[0, i, j] * s[1, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j+1]
            res += s[0, i+1, j] * s[1, i, j+1] * s[1, i+1, j+1] * s[1, i, j+2]
            res += s[0, i, j] * s[1, i, j+1] * s[1, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[1, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j]
            res += s[1, i, j] * s[1, i, j+1] * s[1, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[1, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j]
    return res


# Unclear, slight Pi bias
def Pattern7(s):
    """ Compute the correlator for this pattern:
            ↓ ○
          ↑ ↑ 
        and symmetry-equivalent patterns
    """
    res = 0.0

    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[0, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+2, j+1]
            res += s[0, i+1, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[2, i, j+2]
            res += s[2, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j+1]
            res += s[2, i+1, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[0, i, j+2]
            res += s[2, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[0, i+1, j+1] * s[1, i+1, j] * s[2, i+2, j]
            res += s[0, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[2, i+1, j+2]
            res += s[2, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j]

            res += s[1, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+2, j+1]
            res += s[1, i+1, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[2, i, j+2]
            res += s[2, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j+1]
            res += s[2, i+1, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[1, i, j+2]
            res += s[2, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[1, i+1, j+1] * s[0, i+1, j] * s[2, i+2, j]
            res += s[1, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[2, i+1, j+2]
            res += s[2, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j]

    return res


def Pattern8(s):
    """ Compute the correlator for this pattern:
            ↓ 
          ↓ ↑ ○
        and symmetry-equivalent patterns
    """
    res = 0.0
    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[1, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+2, j]
            res += s[1, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[2, i+1, j+2]
            res += s[2, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j+1]
            res += s[2, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i, j+2]
            res += s[2, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j]
            res += s[2, i+1, j] * s[1, i, j+2] * s[0, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+2, j+1]
            res += s[1, i, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[2, i, j+2]

            res += s[0, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+2, j]
            res += s[0, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[2, i+1, j+2]
            res += s[2, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j+1]
            res += s[2, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i, j+2]
            res += s[2, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j]
            res += s[2, i+1, j] * s[0, i, j+2] * s[1, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+2, j+1]
            res += s[0, i, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[2, i, j+2]
    return res


def Pattern9(s):
    """ Compute the correlator for this pattern:
            ↓ ○
          ↓ ↑ 
        and symmetry-equivalent patterns
    """
    res = 0.0
    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[1, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+2, j+1]
            res += s[1, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[2, i, j+2]
            res += s[2, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j+1]
            res += s[2, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i, j+2]
            res += s[2, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j]
            res += s[2, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[1, i+1, j+2]
            res += s[1, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+2, j]
            res += s[1, i, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[2, i+1, j+2]

            res += s[0, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+2, j+1]
            res += s[0, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[2, i, j+2]
            res += s[2, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j+1]
            res += s[2, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i, j+2]
            res += s[2, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j]
            res += s[2, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[0, i+1, j+2]
            res += s[0, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+2, j]
            res += s[0, i, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[2, i+1, j+2]
    return res


def Pattern10(s):
    """ Compute the correlator for this pattern:
          ↓ ↓ ○
            ↑ 
        and symmetry-equivalent patterns
    """
    res = 0.0
    s = np.pad(s, ((0, 0), (2, 2), (2, 2)))

    L = s.shape[-1]
    for i in range(L-2):
        for j in range(L-2):
            res += s[1, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+2, j+1]
            res += s[1, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[2, i, j+2]
            res += s[2, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[1, i+2, j]
            res += s[2, i+1, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[1, i+1, j+2]
            res += s[2, i, j+1] * s[0, i+1, j] * s[1, i+1, j+1] * s[1, i+2, j+1]
            res += s[2, i, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[1, i, j+2]
            res += s[1, i, j] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+2, j]
            res += s[1, i+1, j] * s[1, i+1, j+1] * s[0, i, j+1] * s[2, i+1, j+2]

            res += s[0, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[2, i+2, j+1]
            res += s[0, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[2, i, j+2]
            res += s[2, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[0, i+2, j]
            res += s[2, i+1, j] * s[1, i, j+1] * s[0, i+1, j+1] * s[0, i+1, j+2]
            res += s[2, i, j+1] * s[1, i+1, j] * s[0, i+1, j+1] * s[0, i+2, j+1]
            res += s[2, i, j] * s[0, i, j+1] * s[1, i+1, j+1] * s[0, i, j+2]
            res += s[0, i, j] * s[0, i+1, j] * s[1, i+1, j+1] * s[2, i+2, j]
            res += s[0, i+1, j] * s[0, i+1, j+1] * s[1, i, j+1] * s[2, i+1, j+2]
    return res
