import numba
import numpy as np
from numba.experimental import jitclass

from pynucastro.rates import Tfactors, _find_rate_file
from pynucastro.screening import PlasmaState, ScreenFactors

jp = 0
jhe4 = 1
jne20 = 2
jne21 = 3
jne22 = 4
jna21 = 5
jna22 = 6
jna23 = 7
jmg24 = 8
nnuc = 9

A = np.zeros((nnuc), dtype=np.int32)

A[jp] = 1
A[jhe4] = 4
A[jne20] = 20
A[jne21] = 21
A[jne22] = 22
A[jna21] = 21
A[jna22] = 22
A[jna23] = 23
A[jmg24] = 24

Z = np.zeros((nnuc), dtype=np.int32)

Z[jp] = 1
Z[jhe4] = 2
Z[jne20] = 10
Z[jne21] = 10
Z[jne22] = 10
Z[jna21] = 11
Z[jna22] = 11
Z[jna23] = 11
Z[jmg24] = 12

names = []
names.append("h1")
names.append("he4")
names.append("ne20")
names.append("ne21")
names.append("ne22")
names.append("na21")
names.append("na22")
names.append("na23")
names.append("mg24")

@jitclass([
    ("na21__ne21__weak__wc12", numba.float64),
    ("na22__ne22__weak__wc12", numba.float64),
    ("na21__p_ne20", numba.float64),
    ("na22__p_ne21", numba.float64),
    ("na23__p_ne22", numba.float64),
    ("mg24__p_na23", numba.float64),
    ("mg24__he4_ne20", numba.float64),
    ("p_ne20__na21", numba.float64),
    ("he4_ne20__mg24", numba.float64),
    ("p_ne21__na22", numba.float64),
    ("p_ne22__na23", numba.float64),
    ("p_na23__mg24", numba.float64),
    ("he4_ne20__p_na23", numba.float64),
    ("he4_na21__p_mg24", numba.float64),
    ("p_na23__he4_ne20", numba.float64),
    ("p_mg24__he4_na21", numba.float64),
])
class RateEval:
    def __init__(self):
        self.na21__ne21__weak__wc12 = np.nan
        self.na22__ne22__weak__wc12 = np.nan
        self.na21__p_ne20 = np.nan
        self.na22__p_ne21 = np.nan
        self.na23__p_ne22 = np.nan
        self.mg24__p_na23 = np.nan
        self.mg24__he4_ne20 = np.nan
        self.p_ne20__na21 = np.nan
        self.he4_ne20__mg24 = np.nan
        self.p_ne21__na22 = np.nan
        self.p_ne22__na23 = np.nan
        self.p_na23__mg24 = np.nan
        self.he4_ne20__p_na23 = np.nan
        self.he4_na21__p_mg24 = np.nan
        self.p_na23__he4_ne20 = np.nan
        self.p_mg24__he4_na21 = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def na21__ne21__weak__wc12(rate_eval, tf):
    # na21 --> ne21
    rate = 0.0

    # wc12w
    rate += np.exp(  -3.48003)

    rate_eval.na21__ne21__weak__wc12 = rate

@numba.njit()
def na22__ne22__weak__wc12(rate_eval, tf):
    # na22 --> ne22
    rate = 0.0

    # wc12w
    rate += np.exp(  -18.59)

    rate_eval.na22__ne22__weak__wc12 = rate

@numba.njit()
def na21__p_ne20(rate_eval, tf):
    # na21 --> p + ne20
    rate = 0.0

    # ly18 
    rate += np.exp(  195320.0 + -89.3596*tf.T9i + 21894.7*tf.T913i + -319153.0*tf.T913
                  + 224369.0*tf.T9 + -188049.0*tf.T953 + 48704.9*tf.lnT9)
    # ly18 
    rate += np.exp(  230.123 + -28.3722*tf.T9i + 15.325*tf.T913i + -294.859*tf.T913
                  + 107.692*tf.T9 + -46.2072*tf.T953 + 59.3398*tf.lnT9)
    # ly18 
    rate += np.exp(  28.0772 + -37.0575*tf.T9i + 20.5893*tf.T913i + -17.5841*tf.T913
                  + 0.243226*tf.T9 + -0.000231418*tf.T953 + 14.3398*tf.lnT9)
    # ly18 
    rate += np.exp(  252.265 + -32.6731*tf.T9i + 258.57*tf.T913i + -506.387*tf.T913
                  + 22.1576*tf.T9 + -0.721182*tf.T953 + 231.788*tf.lnT9)

    rate_eval.na21__p_ne20 = rate

@numba.njit()
def na22__p_ne21(rate_eval, tf):
    # na22 --> p + ne21
    rate = 0.0

    # il10r
    rate += np.exp(  -16.4098 + -82.4235*tf.T9i + 21.1176*tf.T913i + 34.0411*tf.T913
                  + -4.45593*tf.T9 + 0.328613*tf.T953)
    # il10r
    rate += np.exp(  24.8334 + -79.6093*tf.T9i)
    # il10r
    rate += np.exp(  -24.579 + -78.4059*tf.T9i)
    # il10n
    rate += np.exp(  42.146 + -78.2097*tf.T9i + -19.2096*tf.T913i
                  + -1.0*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.na22__p_ne21 = rate

@numba.njit()
def na23__p_ne22(rate_eval, tf):
    # na23 --> p + ne22
    rate = 0.0

    # ke17r
    rate += np.exp(  18.2467 + -104.673*tf.T9i
                  + -2.79964*tf.lnT9)
    # ke17r
    rate += np.exp(  21.6534 + -103.776*tf.T9i
                  + 1.18923*tf.lnT9)
    # ke17r
    rate += np.exp(  0.818178 + -102.466*tf.T9i
                  + 0.009812*tf.lnT9)
    # ke17r
    rate += np.exp(  18.1624 + -102.855*tf.T9i
                  + 4.73558*tf.lnT9)
    # ke17r
    rate += np.exp(  36.29 + -110.779*tf.T9i
                  + 0.732533*tf.lnT9)
    # ke17r
    rate += np.exp(  33.8935 + -106.655*tf.T9i
                  + 1.65623*tf.lnT9)

    rate_eval.na23__p_ne22 = rate

@numba.njit()
def mg24__p_na23(rate_eval, tf):
    # mg24 --> p + na23
    rate = 0.0

    # il10r
    rate += np.exp(  34.0876 + -138.968*tf.T9i + -0.360588*tf.T913
                  + 1.4187*tf.T9 + -0.184061*tf.T953)
    # il10r
    rate += np.exp(  20.0024 + -137.3*tf.T9i)
    # il10n
    rate += np.exp(  43.9357 + -135.688*tf.T9i + -20.6428*tf.T913i + 1.52954*tf.T913
                  + 2.7487*tf.T9 + -1.0*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.mg24__p_na23 = rate

@numba.njit()
def mg24__he4_ne20(rate_eval, tf):
    # mg24 --> he4 + ne20
    rate = 0.0

    # il10n
    rate += np.exp(  49.3244 + -108.114*tf.T9i + -46.2525*tf.T913i + 5.58901*tf.T913
                  + 7.61843*tf.T9 + -3.683*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  16.0203 + -120.895*tf.T9i + 16.9229*tf.T913
                  + -2.57325*tf.T9 + 0.208997*tf.T953)
    # il10r
    rate += np.exp(  26.8017 + -117.334*tf.T9i)
    # il10r
    rate += np.exp(  -13.8869 + -110.62*tf.T9i)

    rate_eval.mg24__he4_ne20 = rate

@numba.njit()
def p_ne20__na21(rate_eval, tf):
    # ne20 + p --> na21
    rate = 0.0

    # ly18 
    rate += np.exp(  230.019 + -4.45358*tf.T9i + 258.57*tf.T913i + -506.387*tf.T913
                  + 22.1576*tf.T9 + -0.721182*tf.T953 + 230.288*tf.lnT9)
    # ly18 
    rate += np.exp(  195297.0 + -61.14*tf.T9i + 21894.7*tf.T913i + -319153.0*tf.T913
                  + 224369.0*tf.T9 + -188049.0*tf.T953 + 48703.4*tf.lnT9)
    # ly18 
    rate += np.exp(  207.877 + -0.152711*tf.T9i + 15.325*tf.T913i + -294.859*tf.T913
                  + 107.692*tf.T9 + -46.2072*tf.T953 + 57.8398*tf.lnT9)
    # ly18 
    rate += np.exp(  5.83103 + -8.838*tf.T9i + 20.5893*tf.T913i + -17.5841*tf.T913
                  + 0.243226*tf.T9 + -0.000231418*tf.T953 + 12.8398*tf.lnT9)

    rate_eval.p_ne20__na21 = rate

@numba.njit()
def he4_ne20__mg24(rate_eval, tf):
    # ne20 + he4 --> mg24
    rate = 0.0

    # il10r
    rate += np.exp(  -38.7055 + -2.50605*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  24.5058 + -46.2525*tf.T913i + 5.58901*tf.T913
                  + 7.61843*tf.T9 + -3.683*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -8.79827 + -12.7809*tf.T9i + 16.9229*tf.T913
                  + -2.57325*tf.T9 + 0.208997*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  1.98307 + -9.22026*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_ne20__mg24 = rate

@numba.njit()
def p_ne21__na22(rate_eval, tf):
    # ne21 + p --> na22
    rate = 0.0

    # il10r
    rate += np.exp(  -47.6554 + -0.19618*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.0696 + -19.2096*tf.T913i
                  + -1.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -39.4862 + -4.21385*tf.T9i + 21.1176*tf.T913i + 34.0411*tf.T913
                  + -4.45593*tf.T9 + 0.328613*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  1.75704 + -1.39957*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_ne21__na22 = rate

@numba.njit()
def p_ne22__na23(rate_eval, tf):
    # ne22 + p --> na23
    rate = 0.0

    # ke17r
    rate += np.exp(  -4.00597 + -2.6179*tf.T9i
                  + -4.29964*tf.lnT9)
    # ke17r
    rate += np.exp(  -0.599331 + -1.72007*tf.T9i
                  + -0.310765*tf.lnT9)
    # ke17r
    rate += np.exp(  -21.4345 + -0.410962*tf.T9i
                  + -1.49019*tf.lnT9)
    # ke17r
    rate += np.exp(  -4.09035 + -0.799756*tf.T9i
                  + 3.23558*tf.lnT9)
    # ke17r
    rate += np.exp(  14.0373 + -8.72377*tf.T9i
                  + -0.767467*tf.lnT9)
    # ke17r
    rate += np.exp(  11.6408 + -4.59936*tf.T9i
                  + 0.156226*tf.lnT9)

    rate_eval.p_ne22__na23 = rate

@numba.njit()
def p_na23__mg24(rate_eval, tf):
    # na23 + p --> mg24
    rate = 0.0

    # il10n
    rate += np.exp(  18.9075 + -20.6428*tf.T913i + 1.52954*tf.T913
                  + 2.7487*tf.T9 + -1.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  9.0594 + -3.28029*tf.T9i + -0.360588*tf.T913
                  + 1.4187*tf.T9 + -0.184061*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -5.02585 + -1.61219*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_na23__mg24 = rate

@numba.njit()
def he4_ne20__p_na23(rate_eval, tf):
    # ne20 + he4 --> p + na23
    rate = 0.0

    # il10r
    rate += np.exp(  0.227472 + -29.4348*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.1852 + -27.5738*tf.T9i + -20.0024*tf.T913i + 11.5988*tf.T913
                  + -1.37398*tf.T9 + -1.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -6.37772 + -29.8896*tf.T9i + 19.7297*tf.T913
                  + -2.20987*tf.T9 + 0.153374*tf.T953 + -1.5*tf.lnT9)

    rate_eval.he4_ne20__p_na23 = rate

@numba.njit()
def he4_na21__p_mg24(rate_eval, tf):
    # na21 + he4 --> p + mg24
    rate = 0.0

    # nacr 
    rate += np.exp(  39.8144 + -49.9621*tf.T913i + 5.90498*tf.T913
                  + -1.6598*tf.T9 + 0.117817*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_na21__p_mg24 = rate

@numba.njit()
def p_na23__he4_ne20(rate_eval, tf):
    # na23 + p --> he4 + ne20
    rate = 0.0

    # il10r
    rate += np.exp(  -6.58736 + -2.31577*tf.T9i + 19.7297*tf.T913
                  + -2.20987*tf.T9 + 0.153374*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  0.0178295 + -1.86103*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  18.9756 + -20.0024*tf.T913i + 11.5988*tf.T913
                  + -1.37398*tf.T9 + -1.0*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_na23__he4_ne20 = rate

@numba.njit()
def p_mg24__he4_na21(rate_eval, tf):
    # mg24 + p --> he4 + na21
    rate = 0.0

    # nacr 
    rate += np.exp(  42.3867 + -79.897*tf.T9i + -49.9621*tf.T913i + 5.90498*tf.T913
                  + -1.6598*tf.T9 + 0.117817*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_mg24__he4_na21 = rate

def rhs(t, Y, rho, T, screen_func=None):
    return rhs_eq(t, Y, rho, T, screen_func)

@numba.njit()
def rhs_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    na21__ne21__weak__wc12(rate_eval, tf)
    na22__ne22__weak__wc12(rate_eval, tf)
    na21__p_ne20(rate_eval, tf)
    na22__p_ne21(rate_eval, tf)
    na23__p_ne22(rate_eval, tf)
    mg24__p_na23(rate_eval, tf)
    mg24__he4_ne20(rate_eval, tf)
    p_ne20__na21(rate_eval, tf)
    he4_ne20__mg24(rate_eval, tf)
    p_ne21__na22(rate_eval, tf)
    p_ne22__na23(rate_eval, tf)
    p_na23__mg24(rate_eval, tf)
    he4_ne20__p_na23(rate_eval, tf)
    he4_na21__p_mg24(rate_eval, tf)
    p_na23__he4_ne20(rate_eval, tf)
    p_mg24__he4_na21(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne20__na21 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_ne20__mg24 *= scor
        rate_eval.he4_ne20__p_na23 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne21__na22 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne22__na23 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_na23__mg24 *= scor
        rate_eval.p_na23__he4_ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_na21__p_mg24 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_mg24__he4_na21 *= scor

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jp] = (
       -rho*Y[jp]*Y[jne20]*rate_eval.p_ne20__na21
       -rho*Y[jp]*Y[jne21]*rate_eval.p_ne21__na22
       -rho*Y[jp]*Y[jne22]*rate_eval.p_ne22__na23
       -rho*Y[jp]*Y[jna23]*rate_eval.p_na23__mg24
       -rho*Y[jp]*Y[jna23]*rate_eval.p_na23__he4_ne20
       -rho*Y[jp]*Y[jmg24]*rate_eval.p_mg24__he4_na21
       +Y[jna21]*rate_eval.na21__p_ne20
       +Y[jna22]*rate_eval.na22__p_ne21
       +Y[jna23]*rate_eval.na23__p_ne22
       +Y[jmg24]*rate_eval.mg24__p_na23
       +rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__p_na23
       +rho*Y[jhe4]*Y[jna21]*rate_eval.he4_na21__p_mg24
       )

    dYdt[jhe4] = (
       -rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__mg24
       -rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__p_na23
       -rho*Y[jhe4]*Y[jna21]*rate_eval.he4_na21__p_mg24
       +Y[jmg24]*rate_eval.mg24__he4_ne20
       +rho*Y[jp]*Y[jna23]*rate_eval.p_na23__he4_ne20
       +rho*Y[jp]*Y[jmg24]*rate_eval.p_mg24__he4_na21
       )

    dYdt[jne20] = (
       -rho*Y[jp]*Y[jne20]*rate_eval.p_ne20__na21
       -rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__mg24
       -rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__p_na23
       +Y[jna21]*rate_eval.na21__p_ne20
       +Y[jmg24]*rate_eval.mg24__he4_ne20
       +rho*Y[jp]*Y[jna23]*rate_eval.p_na23__he4_ne20
       )

    dYdt[jne21] = (
       -rho*Y[jp]*Y[jne21]*rate_eval.p_ne21__na22
       +Y[jna21]*rate_eval.na21__ne21__weak__wc12
       +Y[jna22]*rate_eval.na22__p_ne21
       )

    dYdt[jne22] = (
       -rho*Y[jp]*Y[jne22]*rate_eval.p_ne22__na23
       +Y[jna22]*rate_eval.na22__ne22__weak__wc12
       +Y[jna23]*rate_eval.na23__p_ne22
       )

    dYdt[jna21] = (
       -Y[jna21]*rate_eval.na21__ne21__weak__wc12
       -Y[jna21]*rate_eval.na21__p_ne20
       -rho*Y[jhe4]*Y[jna21]*rate_eval.he4_na21__p_mg24
       +rho*Y[jp]*Y[jne20]*rate_eval.p_ne20__na21
       +rho*Y[jp]*Y[jmg24]*rate_eval.p_mg24__he4_na21
       )

    dYdt[jna22] = (
       -Y[jna22]*rate_eval.na22__ne22__weak__wc12
       -Y[jna22]*rate_eval.na22__p_ne21
       +rho*Y[jp]*Y[jne21]*rate_eval.p_ne21__na22
       )

    dYdt[jna23] = (
       -Y[jna23]*rate_eval.na23__p_ne22
       -rho*Y[jp]*Y[jna23]*rate_eval.p_na23__mg24
       -rho*Y[jp]*Y[jna23]*rate_eval.p_na23__he4_ne20
       +Y[jmg24]*rate_eval.mg24__p_na23
       +rho*Y[jp]*Y[jne22]*rate_eval.p_ne22__na23
       +rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__p_na23
       )

    dYdt[jmg24] = (
       -Y[jmg24]*rate_eval.mg24__p_na23
       -Y[jmg24]*rate_eval.mg24__he4_ne20
       -rho*Y[jp]*Y[jmg24]*rate_eval.p_mg24__he4_na21
       +rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__mg24
       +rho*Y[jp]*Y[jna23]*rate_eval.p_na23__mg24
       +rho*Y[jhe4]*Y[jna21]*rate_eval.he4_na21__p_mg24
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    na21__ne21__weak__wc12(rate_eval, tf)
    na22__ne22__weak__wc12(rate_eval, tf)
    na21__p_ne20(rate_eval, tf)
    na22__p_ne21(rate_eval, tf)
    na23__p_ne22(rate_eval, tf)
    mg24__p_na23(rate_eval, tf)
    mg24__he4_ne20(rate_eval, tf)
    p_ne20__na21(rate_eval, tf)
    he4_ne20__mg24(rate_eval, tf)
    p_ne21__na22(rate_eval, tf)
    p_ne22__na23(rate_eval, tf)
    p_na23__mg24(rate_eval, tf)
    he4_ne20__p_na23(rate_eval, tf)
    he4_na21__p_mg24(rate_eval, tf)
    p_na23__he4_ne20(rate_eval, tf)
    p_mg24__he4_na21(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne20__na21 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_ne20__mg24 *= scor
        rate_eval.he4_ne20__p_na23 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne21__na22 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 22)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne22__na23 *= scor

        scn_fac = ScreenFactors(1, 1, 11, 23)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_na23__mg24 *= scor
        rate_eval.p_na23__he4_ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 11, 21)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_na21__p_mg24 *= scor

        scn_fac = ScreenFactors(1, 1, 12, 24)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_mg24__he4_na21 *= scor

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jp, jp] = (
       -rho*Y[jne20]*rate_eval.p_ne20__na21
       -rho*Y[jne21]*rate_eval.p_ne21__na22
       -rho*Y[jne22]*rate_eval.p_ne22__na23
       -rho*Y[jna23]*rate_eval.p_na23__mg24
       -rho*Y[jna23]*rate_eval.p_na23__he4_ne20
       -rho*Y[jmg24]*rate_eval.p_mg24__he4_na21
       )

    jac[jp, jhe4] = (
       +rho*Y[jne20]*rate_eval.he4_ne20__p_na23
       +rho*Y[jna21]*rate_eval.he4_na21__p_mg24
       )

    jac[jp, jne20] = (
       -rho*Y[jp]*rate_eval.p_ne20__na21
       +rho*Y[jhe4]*rate_eval.he4_ne20__p_na23
       )

    jac[jp, jne21] = (
       -rho*Y[jp]*rate_eval.p_ne21__na22
       )

    jac[jp, jne22] = (
       -rho*Y[jp]*rate_eval.p_ne22__na23
       )

    jac[jp, jna21] = (
       +rate_eval.na21__p_ne20
       +rho*Y[jhe4]*rate_eval.he4_na21__p_mg24
       )

    jac[jp, jna22] = (
       +rate_eval.na22__p_ne21
       )

    jac[jp, jna23] = (
       -rho*Y[jp]*rate_eval.p_na23__mg24
       -rho*Y[jp]*rate_eval.p_na23__he4_ne20
       +rate_eval.na23__p_ne22
       )

    jac[jp, jmg24] = (
       -rho*Y[jp]*rate_eval.p_mg24__he4_na21
       +rate_eval.mg24__p_na23
       )

    jac[jhe4, jp] = (
       +rho*Y[jna23]*rate_eval.p_na23__he4_ne20
       +rho*Y[jmg24]*rate_eval.p_mg24__he4_na21
       )

    jac[jhe4, jhe4] = (
       -rho*Y[jne20]*rate_eval.he4_ne20__mg24
       -rho*Y[jne20]*rate_eval.he4_ne20__p_na23
       -rho*Y[jna21]*rate_eval.he4_na21__p_mg24
       )

    jac[jhe4, jne20] = (
       -rho*Y[jhe4]*rate_eval.he4_ne20__mg24
       -rho*Y[jhe4]*rate_eval.he4_ne20__p_na23
       )

    jac[jhe4, jna21] = (
       -rho*Y[jhe4]*rate_eval.he4_na21__p_mg24
       )

    jac[jhe4, jna23] = (
       +rho*Y[jp]*rate_eval.p_na23__he4_ne20
       )

    jac[jhe4, jmg24] = (
       +rate_eval.mg24__he4_ne20
       +rho*Y[jp]*rate_eval.p_mg24__he4_na21
       )

    jac[jne20, jp] = (
       -rho*Y[jne20]*rate_eval.p_ne20__na21
       +rho*Y[jna23]*rate_eval.p_na23__he4_ne20
       )

    jac[jne20, jhe4] = (
       -rho*Y[jne20]*rate_eval.he4_ne20__mg24
       -rho*Y[jne20]*rate_eval.he4_ne20__p_na23
       )

    jac[jne20, jne20] = (
       -rho*Y[jp]*rate_eval.p_ne20__na21
       -rho*Y[jhe4]*rate_eval.he4_ne20__mg24
       -rho*Y[jhe4]*rate_eval.he4_ne20__p_na23
       )

    jac[jne20, jna21] = (
       +rate_eval.na21__p_ne20
       )

    jac[jne20, jna23] = (
       +rho*Y[jp]*rate_eval.p_na23__he4_ne20
       )

    jac[jne20, jmg24] = (
       +rate_eval.mg24__he4_ne20
       )

    jac[jne21, jp] = (
       -rho*Y[jne21]*rate_eval.p_ne21__na22
       )

    jac[jne21, jne21] = (
       -rho*Y[jp]*rate_eval.p_ne21__na22
       )

    jac[jne21, jna21] = (
       +rate_eval.na21__ne21__weak__wc12
       )

    jac[jne21, jna22] = (
       +rate_eval.na22__p_ne21
       )

    jac[jne22, jp] = (
       -rho*Y[jne22]*rate_eval.p_ne22__na23
       )

    jac[jne22, jne22] = (
       -rho*Y[jp]*rate_eval.p_ne22__na23
       )

    jac[jne22, jna22] = (
       +rate_eval.na22__ne22__weak__wc12
       )

    jac[jne22, jna23] = (
       +rate_eval.na23__p_ne22
       )

    jac[jna21, jp] = (
       +rho*Y[jne20]*rate_eval.p_ne20__na21
       +rho*Y[jmg24]*rate_eval.p_mg24__he4_na21
       )

    jac[jna21, jhe4] = (
       -rho*Y[jna21]*rate_eval.he4_na21__p_mg24
       )

    jac[jna21, jne20] = (
       +rho*Y[jp]*rate_eval.p_ne20__na21
       )

    jac[jna21, jna21] = (
       -rate_eval.na21__ne21__weak__wc12
       -rate_eval.na21__p_ne20
       -rho*Y[jhe4]*rate_eval.he4_na21__p_mg24
       )

    jac[jna21, jmg24] = (
       +rho*Y[jp]*rate_eval.p_mg24__he4_na21
       )

    jac[jna22, jp] = (
       +rho*Y[jne21]*rate_eval.p_ne21__na22
       )

    jac[jna22, jne21] = (
       +rho*Y[jp]*rate_eval.p_ne21__na22
       )

    jac[jna22, jna22] = (
       -rate_eval.na22__ne22__weak__wc12
       -rate_eval.na22__p_ne21
       )

    jac[jna23, jp] = (
       -rho*Y[jna23]*rate_eval.p_na23__mg24
       -rho*Y[jna23]*rate_eval.p_na23__he4_ne20
       +rho*Y[jne22]*rate_eval.p_ne22__na23
       )

    jac[jna23, jhe4] = (
       +rho*Y[jne20]*rate_eval.he4_ne20__p_na23
       )

    jac[jna23, jne20] = (
       +rho*Y[jhe4]*rate_eval.he4_ne20__p_na23
       )

    jac[jna23, jne22] = (
       +rho*Y[jp]*rate_eval.p_ne22__na23
       )

    jac[jna23, jna23] = (
       -rate_eval.na23__p_ne22
       -rho*Y[jp]*rate_eval.p_na23__mg24
       -rho*Y[jp]*rate_eval.p_na23__he4_ne20
       )

    jac[jna23, jmg24] = (
       +rate_eval.mg24__p_na23
       )

    jac[jmg24, jp] = (
       -rho*Y[jmg24]*rate_eval.p_mg24__he4_na21
       +rho*Y[jna23]*rate_eval.p_na23__mg24
       )

    jac[jmg24, jhe4] = (
       +rho*Y[jne20]*rate_eval.he4_ne20__mg24
       +rho*Y[jna21]*rate_eval.he4_na21__p_mg24
       )

    jac[jmg24, jne20] = (
       +rho*Y[jhe4]*rate_eval.he4_ne20__mg24
       )

    jac[jmg24, jna21] = (
       +rho*Y[jhe4]*rate_eval.he4_na21__p_mg24
       )

    jac[jmg24, jna23] = (
       +rho*Y[jp]*rate_eval.p_na23__mg24
       )

    jac[jmg24, jmg24] = (
       -rate_eval.mg24__p_na23
       -rate_eval.mg24__he4_ne20
       -rho*Y[jp]*rate_eval.p_mg24__he4_na21
       )

    return jac
