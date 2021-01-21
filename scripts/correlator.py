import numpy as np

def _flip_spins(snap):
    """ Swaps the two spin channels of some snapshots
    """
    flipped = np.empty_like(snap)
    flipped[..., 0] = snap[..., 1]
    flipped[..., 1] = snap[..., 0]
    flipped[..., 2:] = snap[..., 2:]
    return flipped


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
    return res


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
    return res


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
    return res


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
    return res


def corrHole(s): 
    res = 0.0 
    L = s.shape[-1] 
    for i in range(L-2): 
        for j in range(L-2): 
            res += s[2, i, j] * s[2, i+2, j]
            res += s[2, i, j] * s[2, i, j+2]
    return res


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
    return res


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
    return res


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
    return res


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
    return res


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


def symCorr(snaps, corr_func):
    corr = corr_func(snaps)
    corr += corr_func(np.rot90(snaps, 1, axes=(-1, -2)))
    corr += corr_func(np.rot90(snaps, 2, axes=(-1, -2)))
    corr += corr_func(np.rot90(snaps, 3, axes=(-1, -2)))
    corr += corr_func(snaps[..., ::-1, :])
    corr += corr_func(np.rot90(snaps[..., ::-1, :], 1, axes=(-1, -2)))
    corr += corr_func(snaps[..., :, ::-1])
    corr += corr_func(np.rot90(snaps[..., :, ::-1], 1, axes=(-1, -2)))
    flip_snaps = _flip_spins(snaps)
    corr += corr_func(flip_snaps)
    corr += corr_func(np.rot90(flip_snaps, 1, axes=(-1, -2)))
    corr += corr_func(np.rot90(flip_snaps, 2, axes=(-1, -2)))
    corr += corr_func(np.rot90(flip_snaps, 3, axes=(-1, -2)))
    corr += corr_func(flip_snaps[..., ::-1, :])
    corr += corr_func(np.rot90(flip_snaps[..., ::-1, :], 1, axes=(-1, -2)))
    corr += corr_func(flip_snaps[..., :, ::-1])
    corr += corr_func(np.rot90(flip_snaps[..., :, ::-1], 1, axes=(-1, -2)))
    return corr


def connCorrBound(snaps):
    """ Computes the connected correlator for this pattern:
          ↓ ↑
          ↓ ↑
        given a list of snapshots
    """
    snaps = snaps.astype(np.float64)

    up_dens = np.mean(snaps[:, 0], axis=(-1, -2))
    dn_dens = np.mean(snaps[:, 1], axis=(-1, -2))

    afm_nn = np.mean(snaps[:, 0, :, :-1] * snaps[:, 1, :, 1:], axis=(-1, -2))
    up_fm_nn  = np.mean(snaps[:, 0, :-1, :] * snaps[:, 0, 1:, :], axis=(-1, -2))
    dn_fm_nn  = np.mean(snaps[:, 1, :-1, :] * snaps[:, 1, 1:, :], axis=(-1, -2))

    afm_nnn1 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 1, 1:, 1:], axis=(-1, -2))
    afm_nnn2 = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 0, :-1, 1:], axis=(-1, -2))

    # Empty top-left
    third1 = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 0, :-1, 1:] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    # Empty top-right
    third2 = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 0, :-1, 1:] * snaps[:, 1, 1:, :-1], axis=(-1, -2))
    # Empty bot-right
    third3 = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    # Empty bot-left
    third4 = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:] * snaps[:, 0, :-1, 1:], axis=(-1, -2))

    four = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 1, 1:, :-1]
                   * snaps[:, 0, :-1, 1:] * snaps[:, 0, 1:, 1:], axis=(-1, -2))

    one_pt = -6 * up_dens * up_dens * dn_dens * dn_dens
    two_one_pt = 2 * (
            2 * afm_nn * up_dens * dn_dens
            + up_fm_nn * dn_dens * dn_dens + dn_fm_nn * up_dens * up_dens
            + afm_nnn1 * up_dens * dn_dens + afm_nnn2 * up_dens * dn_dens
    )
    two_two_pt = - (
            up_fm_nn * dn_fm_nn + afm_nn * afm_nn + afm_nnn1 * afm_nnn2
    )
    three_one_pt = - (
            third1 * dn_dens + third2 * up_dens + third3 * up_dens + third4 * dn_dens
    )
    return four + three_one_pt + two_two_pt + two_one_pt + one_pt


def connCorrL(snaps):
    """ Computes the connected correlator for this pattern:
          ↓ ↑
          ↑ ↑
        given a list of snapshots
    """
    snaps = snaps.astype(np.float64)

    up_dens = np.mean(snaps[:, 0], axis=(-1, -2))
    dn_dens = np.mean(snaps[:, 1], axis=(-1, -2))

    vert_afm_nn = np.mean(snaps[:, 0, :-1, :] * snaps[:, 1, 1:, :], axis=(-1, -2))
    hori_afm_nn = np.mean(snaps[:, 1, :, :-1] * snaps[:, 0, :, 1:], axis=(-1, -2))
    vert_fm_nn  = np.mean(snaps[:, 0, :-1, :] * snaps[:, 0, 1:, :], axis=(-1, -2))
    hori_fm_nn  = np.mean(snaps[:, 0, :, :-1] * snaps[:, 0, :, 1:], axis=(-1, -2))

    fm_nnn1 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    afm_nnn2 = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 0, :-1, 1:], axis=(-1, -2))

    # Empty top-left
    third1 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 0, :-1, 1:] * snaps[:, 1, 1:, 1:], axis=(-1, -2))
    # Empty top-right
    third2 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 1, :-1, 1:] * snaps[:, 0, 1:, :-1], axis=(-1, -2))
    # Empty bot-right
    third3 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    # Empty bot-left
    third4 = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:] * snaps[:, 0, :-1, 1:], axis=(-1, -2))

    four = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 1, 1:, :-1]
                   * snaps[:, 0, :-1, 1:] * snaps[:, 0, 1:, 1:], axis=(-1, -2))

    one_pt = -6 * up_dens * up_dens * up_dens * dn_dens
    two_one_pt = 2 * (
            hori_afm_nn * up_dens * up_dens + hori_fm_nn * up_dens * dn_dens
            + vert_afm_nn * up_dens * up_dens + vert_fm_nn * up_dens * dn_dens
            + fm_nnn1 * up_dens * dn_dens + afm_nnn2 * up_dens * up_dens
    )
    two_two_pt = - (
            vert_afm_nn * vert_fm_nn + hori_afm_nn * hori_fm_nn + fm_nnn1 * afm_nnn2
    )
    three_one_pt = - (
            third1 * dn_dens + third2 * up_dens + third3 * up_dens + third4 * up_dens
    )
    return four  + three_one_pt + two_two_pt + two_one_pt + one_pt


def connCorrStringHole(snaps):
    """ Computes the connected correlator for this pattern:
            ↓ ↑
            ↓ ○
        given a list of snapshots
    """
    snaps = snaps.astype(np.float64)

    up_dens = np.mean(snaps[:, 0], axis=(-1, -2))
    dn_dens = np.mean(snaps[:, 1], axis=(-1, -2))
    hole_dens = np.mean(snaps[:, 2], axis=(-1, -2))

    hori_afm_nn = np.mean(snaps[:, 0, :, :-1] * snaps[:, 1, :, 1:], axis=(-1, -2))
    vert_fm_nn  = np.mean(snaps[:, 1, :-1, :] * snaps[:, 1, 1:, :], axis=(-1, -2))
    dn_hole_nn = np.mean(snaps[:, 1, :, :-1] * snaps[:, 2, :, 1:], axis=(-1, -2))
    up_hole_nn = np.mean(snaps[:, 2, :-1, :] * snaps[:, 0, 1:, :], axis=(-1, -2))

    dn_hole_nnn = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 2, :-1, 1:], axis=(-1, -2))
    afm_nnn = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 1, 1:, 1:], axis=(-1, -2))

    # Empty top-left
    third1 = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 2, :-1, 1:] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    # Empty top-right
    third2 = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 2, :-1, 1:] * snaps[:, 1, 1:, :-1], axis=(-1, -2))
    # Empty bot-right
    third3 = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    # Empty bot-left
    third4 = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:] * snaps[:, 2, :-1, 1:], axis=(-1, -2))

    four = np.mean(snaps[:, 1, :-1, :-1] * snaps[:, 1, 1:, :-1]
                   * snaps[:, 2, :-1, 1:] * snaps[:, 0, 1:, 1:], axis=(-1, -2))

    one_pt = -6 * dn_dens * dn_dens * up_dens * hole_dens
    two_one_pt = 2 * (
            hori_afm_nn * dn_dens * hole_dens + dn_hole_nn * dn_dens * up_dens
            + vert_fm_nn * up_dens * hole_dens + up_hole_nn * dn_dens * dn_dens
            + afm_nnn * dn_dens * hole_dens + dn_hole_nnn * up_dens * dn_dens
    )
    two_two_pt = - (
            vert_fm_nn * up_hole_nn + hori_afm_nn * dn_hole_nn + afm_nnn * dn_hole_nnn
    )
    three_one_pt = - (
            third1 * dn_dens + third2 * up_dens + third3 * hole_dens + third4 * dn_dens
    )
    return four + three_one_pt + two_two_pt + two_one_pt + one_pt


def connCorrAFMHole(snaps):
    """ Computes the connected correlator for this pattern:
            ↓ ↑
            ↑ ○
        given a list of snapshots
    """
    snaps = snaps.astype(np.float64)

    up_dens = np.mean(snaps[:, 0], axis=(-1, -2))
    dn_dens = np.mean(snaps[:, 1], axis=(-1, -2))
    hole_dens = np.mean(snaps[:, 2], axis=(-1, -2))

    hori_afm_nn = np.mean(snaps[:, 1, :, :-1] * snaps[:, 0, :, 1:], axis=(-1, -2))
    vert_afm_nn  = np.mean(snaps[:, 0, :-1, :] * snaps[:, 1, 1:, :], axis=(-1, -2))
    vert_hole_nn = np.mean(snaps[:, 0, :, :-1] * snaps[:, 2, :, 1:], axis=(-1, -2))
    hori_hole_nn = np.mean(snaps[:, 2, :-1, :] * snaps[:, 0, 1:, :], axis=(-1, -2))

    dn_hole_nnn = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 2, :-1, 1:], axis=(-1, -2))
    fm_nnn = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 0, 1:, 1:], axis=(-1, -2))

    # Empty top-left
    third1 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 2, :-1, 1:] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    # Empty top-right
    third2 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 2, :-1, 1:] * snaps[:, 1, 1:, :-1], axis=(-1, -2))
    # Empty bot-right
    third3 = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:], axis=(-1, -2))
    # Empty bot-left
    third4 = np.mean(snaps[:, 1, 1:, :-1] * snaps[:, 0, 1:, 1:] * snaps[:, 2, :-1, 1:], axis=(-1, -2))

    four = np.mean(snaps[:, 0, :-1, :-1] * snaps[:, 1, 1:, :-1]
                   * snaps[:, 2, :-1, 1:] * snaps[:, 0, 1:, 1:], axis=(-1, -2))

    one_pt = -6 * dn_dens * up_dens * up_dens * hole_dens
    two_one_pt = 2 * (
            hori_afm_nn * up_dens * hole_dens + hori_hole_nn * dn_dens * up_dens
            + vert_afm_nn * up_dens * hole_dens + vert_hole_nn * dn_dens * up_dens
            + fm_nnn * dn_dens * hole_dens + dn_hole_nnn * up_dens * up_dens
    )
    two_two_pt = - (
            vert_afm_nn * vert_hole_nn + hori_afm_nn * hori_hole_nn + fm_nnn * dn_hole_nnn
    )
    three_one_pt = - (
            third1 * dn_dens + third2 * up_dens + third3 * hole_dens + third4 * up_dens
    )
    return four + three_one_pt + two_two_pt + two_one_pt + one_pt


if __name__ == "__main__":
    print("This is not a runnable script, it simply provides functions which measure correlators")

    import sys
    sys.exit(1)
