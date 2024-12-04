import numpy as np
from numba import jit

@jit
def lim(Hb, dt, A, mb, kr, mu, Ym, Yini):
    """
    Lim et al. 2022 model
    """
    gama = 0.55
    f = 1.51
    Hbm = 6
    ar = gama * A**(3/2) / 16 * (Hbm/((Hbm/gama)**0.5-f))
    
    Eb = Hb**2 / 16
    Seq = Eb / ar
    
    S = np.zeros(len(Eb))
    S[0] = np.mean(Seq) + Yini - Ym
        
    for i in range(len(S) - 1):
        S[i+1] = S[i] + dt[i]*kr/24*(Seq[i+1]-S[i])

    S = S - np.mean(S) + Hb * mu / mb - np.mean(Hb * mu / mb)
    S = Ym - S
        
    return S

def lim_njit(Hb, dt, A, mb, kr, mu, Ym, Yini):
    """
    Lim et al. 2022 model
    """
    gama = 0.55
    f = 1.51
    Hbm = 6
    ar = gama * A**(3/2) / 16 * (Hbm/((Hbm/gama)**0.5-f))
    
    Eb = Hb**2 / 16
    Seq = Eb / ar
    
    S = np.zeros(len(Eb))
    S[0] = np.mean(Seq) + Yini - Ym
        
    for i in range(len(S) - 1):
        S[i+1] = S[i] + dt[i]*kr/24*(Seq[i+1]-S[i])

    S = S - np.mean(S) + Hb * mu / mb - np.mean(Hb * mu / mb)
    S = Ym - S
        
    return S