import numpy as np
from numpy import log, arange

_FAST_FFT_SIZES = None         # populated when this module is loaded.
_FAST_FFT_SIZES_MAX = 2**20+1

__all__ = ['next_fast_fft']


def next_fast_fft(n):
    """Find the next integer greater than or equal to `n` for which the
    computation of the FFT is fast.

    This is the smallest number larger than ``n`` which is solely comprised by
    prime factors 2, 3, 5, 7.

    Parameters
    ----------
    n : array-like or scalar
        One or more integers.

    Returns
    -------
    nplus : array-like or scalar
        A vector of integers greater than or equal to ``n``.

    Reference
    ---------
    Sondergaard, P. "Next Fast FFT Size" http://ltfat.sourceforge.net/notes/ltfatnote017.pdf
    """
    n = np.asarray(n, dtype=int)

    # reduce all the numbers by powers of 2 to get below _FAST_FFT_SIZES_MAX
    reduction = np.ones_like(n)
    reduction += (n > _FAST_FFT_SIZES_MAX) * np.ceil(np.log2(n / _FAST_FFT_SIZES_MAX))
    n = n / (2**reduction)

    idx = np.searchsorted(_FAST_FFT_SIZES, n)
    out = _FAST_FFT_SIZES[idx]
    out = out * (2**reduction)
    assert np.all(out >= n)
    return out


def _build_fast_fft_sizes():
    """Build a table of all of the integers less than _FAST_FFT_SIZES
    whose prime factors are only in (2, 3, 5, 7)

    Reference
    ---------
    Sondergaard, P. "Next Fast FFT Size" http://ltfat.sourceforge.net/notes/ltfatnote017.pdf
    """
    global _FAST_FFT_SIZES
    _FAST_FFT_SIZES = set()
    logmax = np.log(_FAST_FFT_SIZES_MAX)

    for p2 in 2**arange(logmax/log(2)):
        for p3 in 3**arange((logmax-log(p2))/log(3)):
            for p5 in 5**arange((logmax-log(p2)-log(p3))/log(5)):
                for p7 in 7**arange((logmax-log(p2)-log(p3)-log(p5))/log(7)):
                    _FAST_FFT_SIZES.add(int(p2 * p3 * p5 * p7))
    _FAST_FFT_SIZES = np.array(sorted(_FAST_FFT_SIZES), dtype=np.int)
    assert len(_FAST_FFT_SIZES) == 1286

_build_fast_fft_sizes()
