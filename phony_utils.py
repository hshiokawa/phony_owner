import numpy as np
from gatspy.periodic import LombScargleMultibandFast
from scipy.signal import find_peaks_cwt

# Called in get_fp()
def get_LS( y ):
    N = len(y[0])
    model = LombScargleMultibandFast().fit(y[0], y[1])

    T = y[0][-1]-y[0][0]
    fmin = 1./T
    fmax = N/(2.*T)
    df = fmin/8.
    f = np.arange(fmin,fmax,df)
    periods = 1./f
    power = model.score(periods)

    power *= N/2 # make it equal to the normalized periodogram of NR eq. 13.8.4
    return [f, power]


def get_top3( f, p ):
    peak_idx = find_peaks_cwt( p, np.array([0.5]), min_snr=.5 )
    fpeak = f[peak_idx]
    ppeak = p[peak_idx]
    fp = sorted(zip( ppeak, fpeak ))
    #fp = zip( ppeak, fpeak )
    #fp.sort()
    fp = np.transpose(fp)
    p_top3 = fp[0]
    f_top3 = fp[1]
    return f_top3[-3:][::-1], p_top3[-3:][::-1]

def get_name(fname):
    start = 'yet'
    name = ''
    for s in fname:
        if s=='.': break
        if start=='go':
            name += s
        if s=='/': start = 'go'
    return name
