import numba
import numpy as np
from numba.experimental import jitclass

from pynucastro.rates import Tfactors, _find_rate_file
from pynucastro.screening import PlasmaState, ScreenFactors

jp = 0
jhe4 = 1
jc12 = 2
jc13 = 3
jn13 = 4
jn14 = 5
jn15 = 6
jo15 = 7
jo16 = 8
jo17 = 9
jf17 = 10
jf18 = 11
jf19 = 12
jne20 = 13
nnuc = 14

A = np.zeros((nnuc), dtype=np.int32)

A[jp] = 1
A[jhe4] = 4
A[jc12] = 12
A[jc13] = 13
A[jn13] = 13
A[jn14] = 14
A[jn15] = 15
A[jo15] = 15
A[jo16] = 16
A[jo17] = 17
A[jf17] = 17
A[jf18] = 18
A[jf19] = 19
A[jne20] = 20

Z = np.zeros((nnuc), dtype=np.int32)

Z[jp] = 1
Z[jhe4] = 2
Z[jc12] = 6
Z[jc13] = 6
Z[jn13] = 7
Z[jn14] = 7
Z[jn15] = 7
Z[jo15] = 8
Z[jo16] = 8
Z[jo17] = 8
Z[jf17] = 9
Z[jf18] = 9
Z[jf19] = 9
Z[jne20] = 10

names = []
names.append("h1")
names.append("he4")
names.append("c12")
names.append("c13")
names.append("n13")
names.append("n14")
names.append("n15")
names.append("o15")
names.append("o16")
names.append("o17")
names.append("f17")
names.append("f18")
names.append("f19")
names.append("ne20")

@jitclass([
    ("n13__c13__weak__wc12", numba.float64),
    ("o15__n15__weak__wc12", numba.float64),
    ("f17__o17__weak__wc12", numba.float64),
    ("n13__p_c12", numba.float64),
    ("n14__p_c13", numba.float64),
    ("o15__p_n14", numba.float64),
    ("o16__p_n15", numba.float64),
    ("o16__he4_c12", numba.float64),
    ("f17__p_o16", numba.float64),
    ("f18__p_o17", numba.float64),
    ("f18__he4_n14", numba.float64),
    ("f19__he4_n15", numba.float64),
    ("ne20__p_f19", numba.float64),
    ("ne20__he4_o16", numba.float64),
    ("c12__he4_he4_he4", numba.float64),
    ("p_c12__n13", numba.float64),
    ("he4_c12__o16", numba.float64),
    ("p_c13__n14", numba.float64),
    ("p_n14__o15", numba.float64),
    ("he4_n14__f18", numba.float64),
    ("p_n15__o16", numba.float64),
    ("he4_n15__f19", numba.float64),
    ("p_o16__f17", numba.float64),
    ("he4_o16__ne20", numba.float64),
    ("p_o17__f18", numba.float64),
    ("p_f19__ne20", numba.float64),
    ("he4_c12__p_n15", numba.float64),
    ("c12_c12__he4_ne20", numba.float64),
    ("he4_n13__p_o16", numba.float64),
    ("he4_n14__p_o17", numba.float64),
    ("p_n15__he4_c12", numba.float64),
    ("he4_o15__p_f18", numba.float64),
    ("p_o16__he4_n13", numba.float64),
    ("he4_o16__p_f19", numba.float64),
    ("p_o17__he4_n14", numba.float64),
    ("he4_f17__p_ne20", numba.float64),
    ("p_f18__he4_o15", numba.float64),
    ("p_f19__he4_o16", numba.float64),
    ("p_ne20__he4_f17", numba.float64),
    ("he4_ne20__c12_c12", numba.float64),
    ("he4_he4_he4__c12", numba.float64),
])
class RateEval:
    def __init__(self):
        self.n13__c13__weak__wc12 = np.nan
        self.o15__n15__weak__wc12 = np.nan
        self.f17__o17__weak__wc12 = np.nan
        self.n13__p_c12 = np.nan
        self.n14__p_c13 = np.nan
        self.o15__p_n14 = np.nan
        self.o16__p_n15 = np.nan
        self.o16__he4_c12 = np.nan
        self.f17__p_o16 = np.nan
        self.f18__p_o17 = np.nan
        self.f18__he4_n14 = np.nan
        self.f19__he4_n15 = np.nan
        self.ne20__p_f19 = np.nan
        self.ne20__he4_o16 = np.nan
        self.c12__he4_he4_he4 = np.nan
        self.p_c12__n13 = np.nan
        self.he4_c12__o16 = np.nan
        self.p_c13__n14 = np.nan
        self.p_n14__o15 = np.nan
        self.he4_n14__f18 = np.nan
        self.p_n15__o16 = np.nan
        self.he4_n15__f19 = np.nan
        self.p_o16__f17 = np.nan
        self.he4_o16__ne20 = np.nan
        self.p_o17__f18 = np.nan
        self.p_f19__ne20 = np.nan
        self.he4_c12__p_n15 = np.nan
        self.c12_c12__he4_ne20 = np.nan
        self.he4_n13__p_o16 = np.nan
        self.he4_n14__p_o17 = np.nan
        self.p_n15__he4_c12 = np.nan
        self.he4_o15__p_f18 = np.nan
        self.p_o16__he4_n13 = np.nan
        self.he4_o16__p_f19 = np.nan
        self.p_o17__he4_n14 = np.nan
        self.he4_f17__p_ne20 = np.nan
        self.p_f18__he4_o15 = np.nan
        self.p_f19__he4_o16 = np.nan
        self.p_ne20__he4_f17 = np.nan
        self.he4_ne20__c12_c12 = np.nan
        self.he4_he4_he4__c12 = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def n13__c13__weak__wc12(rate_eval, tf):
    # n13 --> c13
    rate = 0.0

    # wc12w
    rate += np.exp(  -6.7601)

    rate_eval.n13__c13__weak__wc12 = rate

@numba.njit()
def o15__n15__weak__wc12(rate_eval, tf):
    # o15 --> n15
    rate = 0.0

    # wc12w
    rate += np.exp(  -5.17053)

    rate_eval.o15__n15__weak__wc12 = rate

@numba.njit()
def f17__o17__weak__wc12(rate_eval, tf):
    # f17 --> o17
    rate = 0.0

    # wc12w
    rate += np.exp(  -4.53318)

    rate_eval.f17__o17__weak__wc12 = rate

@numba.njit()
def n13__p_c12(rate_eval, tf):
    # n13 --> p + c12
    rate = 0.0

    # ls09r
    rate += np.exp(  40.4354 + -26.326*tf.T9i + -5.10735*tf.T913i + -2.24111*tf.T913
                  + 0.148883*tf.T9)
    # ls09n
    rate += np.exp(  40.0408 + -22.5475*tf.T9i + -13.692*tf.T913i + -0.230881*tf.T913
                  + 4.44362*tf.T9 + -3.15898*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.n13__p_c12 = rate

@numba.njit()
def n14__p_c13(rate_eval, tf):
    # n14 --> p + c13
    rate = 0.0

    # nacrr
    rate += np.exp(  37.1528 + -93.4071*tf.T9i + -0.196703*tf.T913
                  + 0.142126*tf.T9 + -0.0238912*tf.T953)
    # nacrr
    rate += np.exp(  38.3716 + -101.18*tf.T9i)
    # nacrn
    rate += np.exp(  41.7046 + -87.6256*tf.T9i + -13.72*tf.T913i + -0.450018*tf.T913
                  + 3.70823*tf.T9 + -1.70545*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.n14__p_c13 = rate

@numba.njit()
def o15__p_n14(rate_eval, tf):
    # o15 --> p + n14
    rate = 0.0

    # im05r
    rate += np.exp(  30.7435 + -89.5667*tf.T9i
                  + 1.5682*tf.lnT9)
    # im05r
    rate += np.exp(  31.6622 + -87.6737*tf.T9i)
    # im05n
    rate += np.exp(  44.1246 + -84.6757*tf.T9i + -15.193*tf.T913i + -4.63975*tf.T913
                  + 9.73458*tf.T9 + -9.55051*tf.T953 + 1.83333*tf.lnT9)
    # im05n
    rate += np.exp(  41.0177 + -84.6757*tf.T9i + -15.193*tf.T913i + -0.161954*tf.T913
                  + -7.52123*tf.T9 + -0.987565*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.o15__p_n14 = rate

@numba.njit()
def o16__p_n15(rate_eval, tf):
    # o16 --> p + n15
    rate = 0.0

    # li10r
    rate += np.exp(  38.8465 + -150.962*tf.T9i
                  + 0.0459037*tf.T9)
    # li10r
    rate += np.exp(  30.8927 + -143.656*tf.T9i)
    # li10n
    rate += np.exp(  44.3197 + -140.732*tf.T9i + -15.24*tf.T913i + 0.334926*tf.T913
                  + 4.59088*tf.T9 + -4.78468*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.o16__p_n15 = rate

@numba.njit()
def o16__he4_c12(rate_eval, tf):
    # o16 --> he4 + c12
    rate = 0.0

    # nac2 
    rate += np.exp(  279.295 + -84.9515*tf.T9i + 103.411*tf.T913i + -420.567*tf.T913
                  + 64.0874*tf.T9 + -12.4624*tf.T953 + 138.803*tf.lnT9)
    # nac2 
    rate += np.exp(  94.3131 + -84.503*tf.T9i + 58.9128*tf.T913i + -148.273*tf.T913
                  + 9.08324*tf.T9 + -0.541041*tf.T953 + 71.8554*tf.lnT9)

    rate_eval.o16__he4_c12 = rate

@numba.njit()
def f17__p_o16(rate_eval, tf):
    # f17 --> p + o16
    rate = 0.0

    # ia08n
    rate += np.exp(  40.9135 + -6.96583*tf.T9i + -16.696*tf.T913i + -1.16252*tf.T913
                  + 0.267703*tf.T9 + -0.0338411*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.f17__p_o16 = rate

@numba.njit()
def f18__p_o17(rate_eval, tf):
    # f18 --> p + o17
    rate = 0.0

    # il10r
    rate += np.exp(  33.7037 + -71.2889*tf.T9i + 2.31435*tf.T913
                  + -0.302835*tf.T9 + 0.020133*tf.T953)
    # il10r
    rate += np.exp(  11.2362 + -65.8069*tf.T9i)
    # il10n
    rate += np.exp(  40.2061 + -65.0606*tf.T9i + -16.4035*tf.T913i + 4.31885*tf.T913
                  + -0.709921*tf.T9 + -2.0*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.f18__p_o17 = rate

@numba.njit()
def f18__he4_n14(rate_eval, tf):
    # f18 --> he4 + n14
    rate = 0.0

    # il10n
    rate += np.exp(  46.249 + -51.2292*tf.T9i + -36.2504*tf.T913i
                  + -5.0*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  38.6146 + -62.1948*tf.T9i + -5.6227*tf.T913i)
    # il10r
    rate += np.exp(  24.9119 + -56.3896*tf.T9i)

    rate_eval.f18__he4_n14 = rate

@numba.njit()
def f19__he4_n15(rate_eval, tf):
    # f19 --> he4 + n15
    rate = 0.0

    # il10r
    rate += np.exp(  15.3186 + -50.7554*tf.T9i)
    # il10n
    rate += np.exp(  50.1291 + -46.5774*tf.T9i + -36.2324*tf.T913i
                  + -2.0*tf.T953 + 0.833333*tf.lnT9)
    # il10r
    rate += np.exp(  -4.06142 + -50.7773*tf.T9i + 35.4292*tf.T913
                  + -5.5767*tf.T9 + 0.441293*tf.T953)
    # il10r
    rate += np.exp(  28.2717 + -53.5621*tf.T9i)

    rate_eval.f19__he4_n15 = rate

@numba.njit()
def ne20__p_f19(rate_eval, tf):
    # ne20 --> p + f19
    rate = 0.0

    # nacrr
    rate += np.exp(  18.691 + -156.781*tf.T9i + 31.6442*tf.T913i + -58.6563*tf.T913
                  + 67.7365*tf.T9 + -22.9721*tf.T953)
    # nacrr
    rate += np.exp(  36.7036 + -150.75*tf.T9i + -11.3832*tf.T913i + 5.47872*tf.T913
                  + -1.07203*tf.T9 + 0.11196*tf.T953)
    # nacrn
    rate += np.exp(  42.6027 + -149.037*tf.T9i + -18.116*tf.T913i + -1.4622*tf.T913
                  + 6.95113*tf.T9 + -2.90366*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.ne20__p_f19 = rate

@numba.njit()
def ne20__he4_o16(rate_eval, tf):
    # ne20 --> he4 + o16
    rate = 0.0

    # co10r
    rate += np.exp(  34.2658 + -67.6518*tf.T9i + -3.65925*tf.T913
                  + 0.714224*tf.T9 + -0.00107508*tf.T953)
    # co10r
    rate += np.exp(  28.6431 + -65.246*tf.T9i)
    # co10n
    rate += np.exp(  48.6604 + -54.8875*tf.T9i + -39.7262*tf.T913i + -0.210799*tf.T913
                  + 0.442879*tf.T9 + -0.0797753*tf.T953 + 0.833333*tf.lnT9)

    rate_eval.ne20__he4_o16 = rate

@numba.njit()
def c12__he4_he4_he4(rate_eval, tf):
    # c12 --> he4 + he4 + he4
    rate = 0.0

    # fy05n
    rate += np.exp(  45.7734 + -84.4227*tf.T9i + -37.06*tf.T913i + 29.3493*tf.T913
                  + -115.507*tf.T9 + -10.0*tf.T953 + 1.66667*tf.lnT9)
    # fy05r
    rate += np.exp(  22.394 + -88.5493*tf.T9i + -13.49*tf.T913i + 21.4259*tf.T913
                  + -1.34769*tf.T9 + 0.0879816*tf.T953 + -10.1653*tf.lnT9)
    # fy05r
    rate += np.exp(  34.9561 + -85.4472*tf.T9i + -23.57*tf.T913i + 20.4886*tf.T913
                  + -12.9882*tf.T9 + -20.0*tf.T953 + 0.83333*tf.lnT9)

    rate_eval.c12__he4_he4_he4 = rate

@numba.njit()
def p_c12__n13(rate_eval, tf):
    # c12 + p --> n13
    rate = 0.0

    # ls09n
    rate += np.exp(  17.1482 + -13.692*tf.T913i + -0.230881*tf.T913
                  + 4.44362*tf.T9 + -3.15898*tf.T953 + -0.666667*tf.lnT9)
    # ls09r
    rate += np.exp(  17.5428 + -3.77849*tf.T9i + -5.10735*tf.T913i + -2.24111*tf.T913
                  + 0.148883*tf.T9 + -1.5*tf.lnT9)

    rate_eval.p_c12__n13 = rate

@numba.njit()
def he4_c12__o16(rate_eval, tf):
    # c12 + he4 --> o16
    rate = 0.0

    # nac2 
    rate += np.exp(  254.634 + -1.84097*tf.T9i + 103.411*tf.T913i + -420.567*tf.T913
                  + 64.0874*tf.T9 + -12.4624*tf.T953 + 137.303*tf.lnT9)
    # nac2 
    rate += np.exp(  69.6526 + -1.39254*tf.T9i + 58.9128*tf.T913i + -148.273*tf.T913
                  + 9.08324*tf.T9 + -0.541041*tf.T953 + 70.3554*tf.lnT9)

    rate_eval.he4_c12__o16 = rate

@numba.njit()
def p_c13__n14(rate_eval, tf):
    # c13 + p --> n14
    rate = 0.0

    # nacrr
    rate += np.exp(  15.1825 + -13.5543*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  18.5155 + -13.72*tf.T913i + -0.450018*tf.T913
                  + 3.70823*tf.T9 + -1.70545*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  13.9637 + -5.78147*tf.T9i + -0.196703*tf.T913
                  + 0.142126*tf.T9 + -0.0238912*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_c13__n14 = rate

@numba.njit()
def p_n14__o15(rate_eval, tf):
    # n14 + p --> o15
    rate = 0.0

    # im05n
    rate += np.exp(  17.01 + -15.193*tf.T913i + -0.161954*tf.T913
                  + -7.52123*tf.T9 + -0.987565*tf.T953 + -0.666667*tf.lnT9)
    # im05r
    rate += np.exp(  6.73578 + -4.891*tf.T9i
                  + 0.0682*tf.lnT9)
    # im05r
    rate += np.exp(  7.65444 + -2.998*tf.T9i
                  + -1.5*tf.lnT9)
    # im05n
    rate += np.exp(  20.1169 + -15.193*tf.T913i + -4.63975*tf.T913
                  + 9.73458*tf.T9 + -9.55051*tf.T953 + 0.333333*tf.lnT9)

    rate_eval.p_n14__o15 = rate

@numba.njit()
def he4_n14__f18(rate_eval, tf):
    # n14 + he4 --> f18
    rate = 0.0

    # il10n
    rate += np.exp(  21.5339 + -36.2504*tf.T913i
                  + -5.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  13.8995 + -10.9656*tf.T9i + -5.6227*tf.T913i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  0.196838 + -5.16034*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_n14__f18 = rate

@numba.njit()
def p_n15__o16(rate_eval, tf):
    # n15 + p --> o16
    rate = 0.0

    # li10n
    rate += np.exp(  20.0176 + -15.24*tf.T913i + 0.334926*tf.T913
                  + 4.59088*tf.T9 + -4.78468*tf.T953 + -0.666667*tf.lnT9)
    # li10r
    rate += np.exp(  14.5444 + -10.2295*tf.T9i
                  + 0.0459037*tf.T9 + -1.5*tf.lnT9)
    # li10r
    rate += np.exp(  6.59056 + -2.92315*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_n15__o16 = rate

@numba.njit()
def he4_n15__f19(rate_eval, tf):
    # n15 + he4 --> f19
    rate = 0.0

    # il10r
    rate += np.exp(  -9.41892 + -4.17795*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  25.3916 + -36.2324*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  -28.7989 + -4.19986*tf.T9i + 35.4292*tf.T913
                  + -5.5767*tf.T9 + 0.441293*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  3.5342 + -6.98462*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_n15__f19 = rate

@numba.njit()
def p_o16__f17(rate_eval, tf):
    # o16 + p --> f17
    rate = 0.0

    # ia08n
    rate += np.exp(  19.0904 + -16.696*tf.T913i + -1.16252*tf.T913
                  + 0.267703*tf.T9 + -0.0338411*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_o16__f17 = rate

@numba.njit()
def he4_o16__ne20(rate_eval, tf):
    # o16 + he4 --> ne20
    rate = 0.0

    # co10r
    rate += np.exp(  9.50848 + -12.7643*tf.T9i + -3.65925*tf.T913
                  + 0.714224*tf.T9 + -0.00107508*tf.T953 + -1.5*tf.lnT9)
    # co10r
    rate += np.exp(  3.88571 + -10.3585*tf.T9i
                  + -1.5*tf.lnT9)
    # co10n
    rate += np.exp(  23.903 + -39.7262*tf.T913i + -0.210799*tf.T913
                  + 0.442879*tf.T9 + -0.0797753*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_o16__ne20 = rate

@numba.njit()
def p_o17__f18(rate_eval, tf):
    # o17 + p --> f18
    rate = 0.0

    # il10n
    rate += np.exp(  15.8929 + -16.4035*tf.T913i + 4.31885*tf.T913
                  + -0.709921*tf.T9 + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  9.39048 + -6.22828*tf.T9i + 2.31435*tf.T913
                  + -0.302835*tf.T9 + 0.020133*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -13.077 + -0.746296*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_o17__f18 = rate

@numba.njit()
def p_f19__ne20(rate_eval, tf):
    # f19 + p --> ne20
    rate = 0.0

    # nacrr
    rate += np.exp(  -5.63093 + -7.74414*tf.T9i + 31.6442*tf.T913i + -58.6563*tf.T913
                  + 67.7365*tf.T9 + -22.9721*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  12.3816 + -1.71383*tf.T9i + -11.3832*tf.T913i + 5.47872*tf.T913
                  + -1.07203*tf.T9 + 0.11196*tf.T953 + -1.5*tf.lnT9)
    # nacrn
    rate += np.exp(  18.2807 + -18.116*tf.T913i + -1.4622*tf.T913
                  + 6.95113*tf.T9 + -2.90366*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_f19__ne20 = rate

@numba.njit()
def he4_c12__p_n15(rate_eval, tf):
    # c12 + he4 --> p + n15
    rate = 0.0

    # nacrn
    rate += np.exp(  27.118 + -57.6279*tf.T9i + -15.253*tf.T913i + 1.59318*tf.T913
                  + 2.4479*tf.T9 + -2.19708*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  -6.93365 + -58.7917*tf.T9i + 22.7105*tf.T913
                  + -2.90707*tf.T9 + 0.205754*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  20.5388 + -65.034*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  -5.2319 + -59.6491*tf.T9i + 30.8497*tf.T913
                  + -8.50433*tf.T9 + -1.54426*tf.T953 + -1.5*tf.lnT9)

    rate_eval.he4_c12__p_n15 = rate

@numba.njit()
def c12_c12__he4_ne20(rate_eval, tf):
    # c12 + c12 --> he4 + ne20
    rate = 0.0

    # cf88r
    rate += np.exp(  61.2863 + -84.165*tf.T913i + -1.56627*tf.T913
                  + -0.0736084*tf.T9 + -0.072797*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.c12_c12__he4_ne20 = rate

@numba.njit()
def he4_n13__p_o16(rate_eval, tf):
    # n13 + he4 --> p + o16
    rate = 0.0

    # cf88n
    rate += np.exp(  40.4644 + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_n13__p_o16 = rate

@numba.njit()
def he4_n14__p_o17(rate_eval, tf):
    # n14 + he4 --> p + o17
    rate = 0.0

    # il10r
    rate += np.exp(  -7.60954 + -14.5839*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.1771 + -13.8305*tf.T9i + -16.9078*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  9.77209 + -18.7891*tf.T9i + 5.10182*tf.T913
                  + 0.379373*tf.T9 + -0.0672515*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  5.13169 + -15.9452*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_n14__p_o17 = rate

@numba.njit()
def p_n15__he4_c12(rate_eval, tf):
    # n15 + p --> he4 + c12
    rate = 0.0

    # nacrn
    rate += np.exp(  27.4764 + -15.253*tf.T913i + 1.59318*tf.T913
                  + 2.4479*tf.T9 + -2.19708*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  -6.57522 + -1.1638*tf.T9i + 22.7105*tf.T913
                  + -2.90707*tf.T9 + 0.205754*tf.T953 + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  20.8972 + -7.406*tf.T9i
                  + -1.5*tf.lnT9)
    # nacrr
    rate += np.exp(  -4.87347 + -2.02117*tf.T9i + 30.8497*tf.T913
                  + -8.50433*tf.T9 + -1.54426*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_n15__he4_c12 = rate

@numba.njit()
def he4_o15__p_f18(rate_eval, tf):
    # o15 + he4 --> p + f18
    rate = 0.0

    # il10r
    rate += np.exp(  1.04969 + -36.4627*tf.T9i + 13.3223*tf.T913
                  + -1.36696*tf.T9 + 0.0757363*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -32.4461 + -33.8223*tf.T9i + 61.738*tf.T913
                  + -108.29*tf.T9 + -34.2365*tf.T953 + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  61.2985 + -33.4459*tf.T9i + -21.4023*tf.T913i + -80.8891*tf.T913
                  + 134.6*tf.T9 + -126.504*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_o15__p_f18 = rate

@numba.njit()
def p_o16__he4_n13(rate_eval, tf):
    # o16 + p --> he4 + n13
    rate = 0.0

    # cf88n
    rate += np.exp(  42.2324 + -60.5523*tf.T9i + -35.829*tf.T913i + -0.530275*tf.T913
                  + -0.982462*tf.T9 + 0.0808059*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_o16__he4_n13 = rate

@numba.njit()
def he4_o16__p_f19(rate_eval, tf):
    # o16 + he4 --> p + f19
    rate = 0.0

    # nacr 
    rate += np.exp(  -53.1397 + -94.2866*tf.T9i
                  + -1.5*tf.lnT9)
    # nacr 
    rate += np.exp(  25.8562 + -94.1589*tf.T9i + -18.116*tf.T913i
                  + 1.86674*tf.T9 + -7.5666*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  13.9232 + -97.4449*tf.T9i
                  + -0.21103*tf.T9 + 2.87702*tf.lnT9)
    # nacr 
    rate += np.exp(  14.7601 + -97.9108*tf.T9i
                  + -1.5*tf.lnT9)
    # nacr 
    rate += np.exp(  7.80363 + -96.6272*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.he4_o16__p_f19 = rate

@numba.njit()
def p_o17__he4_n14(rate_eval, tf):
    # o17 + p --> he4 + n14
    rate = 0.0

    # il10r
    rate += np.exp(  5.5336 + -2.11477*tf.T9i
                  + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -7.20763 + -0.753395*tf.T9i
                  + -1.5*tf.lnT9)
    # il10n
    rate += np.exp(  19.579 + -16.9078*tf.T913i
                  + -2.0*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  10.174 + -4.95865*tf.T9i + 5.10182*tf.T913
                  + 0.379373*tf.T9 + -0.0672515*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_o17__he4_n14 = rate

@numba.njit()
def he4_f17__p_ne20(rate_eval, tf):
    # f17 + he4 --> p + ne20
    rate = 0.0

    # nacr 
    rate += np.exp(  38.6287 + -43.18*tf.T913i + 4.46827*tf.T913
                  + -1.63915*tf.T9 + 0.123483*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_f17__p_ne20 = rate

@numba.njit()
def p_f18__he4_o15(rate_eval, tf):
    # f18 + p --> he4 + o15
    rate = 0.0

    # il10n
    rate += np.exp(  62.0058 + -21.4023*tf.T913i + -80.8891*tf.T913
                  + 134.6*tf.T9 + -126.504*tf.T953 + -0.666667*tf.lnT9)
    # il10r
    rate += np.exp(  1.75704 + -3.01675*tf.T9i + 13.3223*tf.T913
                  + -1.36696*tf.T9 + 0.0757363*tf.T953 + -1.5*tf.lnT9)
    # il10r
    rate += np.exp(  -31.7388 + -0.376432*tf.T9i + 61.738*tf.T913
                  + -108.29*tf.T9 + -34.2365*tf.T953 + -1.5*tf.lnT9)

    rate_eval.p_f18__he4_o15 = rate

@numba.njit()
def p_f19__he4_o16(rate_eval, tf):
    # f19 + p --> he4 + o16
    rate = 0.0

    # nacr 
    rate += np.exp(  8.239 + -2.46828*tf.T9i
                  + -1.5*tf.lnT9)
    # nacr 
    rate += np.exp(  -52.7043 + -0.12765*tf.T9i
                  + -1.5*tf.lnT9)
    # nacr 
    rate += np.exp(  26.2916 + -18.116*tf.T913i
                  + 1.86674*tf.T9 + -7.5666*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  14.3586 + -3.286*tf.T9i
                  + -0.21103*tf.T9 + 2.87702*tf.lnT9)
    # nacr 
    rate += np.exp(  15.1955 + -3.75185*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_f19__he4_o16 = rate

@numba.njit()
def p_ne20__he4_f17(rate_eval, tf):
    # ne20 + p --> he4 + f17
    rate = 0.0

    # nacr 
    rate += np.exp(  41.563 + -47.9266*tf.T9i + -43.18*tf.T913i + 4.46827*tf.T913
                  + -1.63915*tf.T9 + 0.123483*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_ne20__he4_f17 = rate

@numba.njit()
def he4_ne20__c12_c12(rate_eval, tf):
    # ne20 + he4 --> c12 + c12
    rate = 0.0

    # cf88r
    rate += np.exp(  61.4748 + -53.6267*tf.T9i + -84.165*tf.T913i + -1.56627*tf.T913
                  + -0.0736084*tf.T9 + -0.072797*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_ne20__c12_c12 = rate

@numba.njit()
def he4_he4_he4__c12(rate_eval, tf):
    # he4 + he4 + he4 --> c12
    rate = 0.0

    # fy05r
    rate += np.exp(  -24.3505 + -4.12656*tf.T9i + -13.49*tf.T913i + 21.4259*tf.T913
                  + -1.34769*tf.T9 + 0.0879816*tf.T953 + -13.1653*tf.lnT9)
    # fy05r
    rate += np.exp(  -11.7884 + -1.02446*tf.T9i + -23.57*tf.T913i + 20.4886*tf.T913
                  + -12.9882*tf.T9 + -20.0*tf.T953 + -2.16667*tf.lnT9)
    # fy05n
    rate += np.exp(  -0.971052 + -37.06*tf.T913i + 29.3493*tf.T913
                  + -115.507*tf.T9 + -10.0*tf.T953 + -1.33333*tf.lnT9)

    rate_eval.he4_he4_he4__c12 = rate

def rhs(t, Y, rho, T, screen_func=None):
    return rhs_eq(t, Y, rho, T, screen_func)

@numba.njit()
def rhs_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    n13__c13__weak__wc12(rate_eval, tf)
    o15__n15__weak__wc12(rate_eval, tf)
    f17__o17__weak__wc12(rate_eval, tf)
    n13__p_c12(rate_eval, tf)
    n14__p_c13(rate_eval, tf)
    o15__p_n14(rate_eval, tf)
    o16__p_n15(rate_eval, tf)
    o16__he4_c12(rate_eval, tf)
    f17__p_o16(rate_eval, tf)
    f18__p_o17(rate_eval, tf)
    f18__he4_n14(rate_eval, tf)
    f19__he4_n15(rate_eval, tf)
    ne20__p_f19(rate_eval, tf)
    ne20__he4_o16(rate_eval, tf)
    c12__he4_he4_he4(rate_eval, tf)
    p_c12__n13(rate_eval, tf)
    he4_c12__o16(rate_eval, tf)
    p_c13__n14(rate_eval, tf)
    p_n14__o15(rate_eval, tf)
    he4_n14__f18(rate_eval, tf)
    p_n15__o16(rate_eval, tf)
    he4_n15__f19(rate_eval, tf)
    p_o16__f17(rate_eval, tf)
    he4_o16__ne20(rate_eval, tf)
    p_o17__f18(rate_eval, tf)
    p_f19__ne20(rate_eval, tf)
    he4_c12__p_n15(rate_eval, tf)
    c12_c12__he4_ne20(rate_eval, tf)
    he4_n13__p_o16(rate_eval, tf)
    he4_n14__p_o17(rate_eval, tf)
    p_n15__he4_c12(rate_eval, tf)
    he4_o15__p_f18(rate_eval, tf)
    p_o16__he4_n13(rate_eval, tf)
    he4_o16__p_f19(rate_eval, tf)
    p_o17__he4_n14(rate_eval, tf)
    he4_f17__p_ne20(rate_eval, tf)
    p_f18__he4_o15(rate_eval, tf)
    p_f19__he4_o16(rate_eval, tf)
    p_ne20__he4_f17(rate_eval, tf)
    he4_ne20__c12_c12(rate_eval, tf)
    he4_he4_he4__c12(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c12__n13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c12__o16 *= scor
        rate_eval.he4_c12__p_n15 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c13__n14 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n14__o15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n14__f18 *= scor
        rate_eval.he4_n14__p_o17 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n15__o16 *= scor
        rate_eval.p_n15__he4_c12 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n15__f19 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o16__f17 *= scor
        rate_eval.p_o16__he4_n13 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_o16__ne20 *= scor
        rate_eval.he4_o16__p_f19 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o17__f18 *= scor
        rate_eval.p_o17__he4_n14 *= scor

        scn_fac = ScreenFactors(1, 1, 9, 19)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_f19__ne20 *= scor
        rate_eval.p_f19__he4_o16 *= scor

        scn_fac = ScreenFactors(6, 12, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.c12_c12__he4_ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n13__p_o16 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_o15__p_f18 *= scor

        scn_fac = ScreenFactors(2, 4, 9, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_f17__p_ne20 *= scor

        scn_fac = ScreenFactors(1, 1, 9, 18)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_f18__he4_o15 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne20__he4_f17 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_ne20__c12_c12 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.he4_he4_he4__c12 *= scor * scor2

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jp] = (
       -rho*Y[jp]*Y[jc12]*rate_eval.p_c12__n13
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n14
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__o15
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jp]*Y[jo16]*rate_eval.p_o16__f17
       -rho*Y[jp]*Y[jo17]*rate_eval.p_o17__f18
       -rho*Y[jp]*Y[jf19]*rate_eval.p_f19__ne20
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       -rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       -rho*Y[jp]*Y[jo17]*rate_eval.p_o17__he4_n14
       -rho*Y[jp]*Y[jf18]*rate_eval.p_f18__he4_o15
       -rho*Y[jp]*Y[jf19]*rate_eval.p_f19__he4_o16
       -rho*Y[jp]*Y[jne20]*rate_eval.p_ne20__he4_f17
       +Y[jn13]*rate_eval.n13__p_c12
       +Y[jn14]*rate_eval.n14__p_c13
       +Y[jo15]*rate_eval.o15__p_n14
       +Y[jo16]*rate_eval.o16__p_n15
       +Y[jf17]*rate_eval.f17__p_o16
       +Y[jf18]*rate_eval.f18__p_o17
       +Y[jne20]*rate_eval.ne20__p_f19
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       +rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
       +rho*Y[jhe4]*Y[jn14]*rate_eval.he4_n14__p_o17
       +rho*Y[jhe4]*Y[jo15]*rate_eval.he4_o15__p_f18
       +rho*Y[jhe4]*Y[jo16]*rate_eval.he4_o16__p_f19
       +rho*Y[jhe4]*Y[jf17]*rate_eval.he4_f17__p_ne20
       )

    dYdt[jhe4] = (
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jhe4]*Y[jn14]*rate_eval.he4_n14__f18
       -rho*Y[jhe4]*Y[jn15]*rate_eval.he4_n15__f19
       -rho*Y[jhe4]*Y[jo16]*rate_eval.he4_o16__ne20
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       -rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
       -rho*Y[jhe4]*Y[jn14]*rate_eval.he4_n14__p_o17
       -rho*Y[jhe4]*Y[jo15]*rate_eval.he4_o15__p_f18
       -rho*Y[jhe4]*Y[jo16]*rate_eval.he4_o16__p_f19
       -rho*Y[jhe4]*Y[jf17]*rate_eval.he4_f17__p_ne20
       -rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__c12_c12
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__c12
       +Y[jo16]*rate_eval.o16__he4_c12
       +Y[jf18]*rate_eval.f18__he4_n14
       +Y[jf19]*rate_eval.f19__he4_n15
       +Y[jne20]*rate_eval.ne20__he4_o16
       +3*Y[jc12]*rate_eval.c12__he4_he4_he4
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.c12_c12__he4_ne20
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       +rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       +rho*Y[jp]*Y[jo17]*rate_eval.p_o17__he4_n14
       +rho*Y[jp]*Y[jf18]*rate_eval.p_f18__he4_o15
       +rho*Y[jp]*Y[jf19]*rate_eval.p_f19__he4_o16
       +rho*Y[jp]*Y[jne20]*rate_eval.p_ne20__he4_f17
       )

    dYdt[jc12] = (
       -Y[jc12]*rate_eval.c12__he4_he4_he4
       -rho*Y[jp]*Y[jc12]*rate_eval.p_c12__n13
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       -2*5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.c12_c12__he4_ne20
       +Y[jn13]*rate_eval.n13__p_c12
       +Y[jo16]*rate_eval.o16__he4_c12
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       +2*rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__c12_c12
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__c12
       )

    dYdt[jc13] = (
       -rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n14
       +Y[jn13]*rate_eval.n13__c13__weak__wc12
       +Y[jn14]*rate_eval.n14__p_c13
       )

    dYdt[jn13] = (
       -Y[jn13]*rate_eval.n13__c13__weak__wc12
       -Y[jn13]*rate_eval.n13__p_c12
       -rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
       +rho*Y[jp]*Y[jc12]*rate_eval.p_c12__n13
       +rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       )

    dYdt[jn14] = (
       -Y[jn14]*rate_eval.n14__p_c13
       -rho*Y[jp]*Y[jn14]*rate_eval.p_n14__o15
       -rho*Y[jhe4]*Y[jn14]*rate_eval.he4_n14__f18
       -rho*Y[jhe4]*Y[jn14]*rate_eval.he4_n14__p_o17
       +Y[jo15]*rate_eval.o15__p_n14
       +Y[jf18]*rate_eval.f18__he4_n14
       +rho*Y[jp]*Y[jc13]*rate_eval.p_c13__n14
       +rho*Y[jp]*Y[jo17]*rate_eval.p_o17__he4_n14
       )

    dYdt[jn15] = (
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jhe4]*Y[jn15]*rate_eval.he4_n15__f19
       -rho*Y[jp]*Y[jn15]*rate_eval.p_n15__he4_c12
       +Y[jo15]*rate_eval.o15__n15__weak__wc12
       +Y[jo16]*rate_eval.o16__p_n15
       +Y[jf19]*rate_eval.f19__he4_n15
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__p_n15
       )

    dYdt[jo15] = (
       -Y[jo15]*rate_eval.o15__n15__weak__wc12
       -Y[jo15]*rate_eval.o15__p_n14
       -rho*Y[jhe4]*Y[jo15]*rate_eval.he4_o15__p_f18
       +rho*Y[jp]*Y[jn14]*rate_eval.p_n14__o15
       +rho*Y[jp]*Y[jf18]*rate_eval.p_f18__he4_o15
       )

    dYdt[jo16] = (
       -Y[jo16]*rate_eval.o16__p_n15
       -Y[jo16]*rate_eval.o16__he4_c12
       -rho*Y[jp]*Y[jo16]*rate_eval.p_o16__f17
       -rho*Y[jhe4]*Y[jo16]*rate_eval.he4_o16__ne20
       -rho*Y[jp]*Y[jo16]*rate_eval.p_o16__he4_n13
       -rho*Y[jhe4]*Y[jo16]*rate_eval.he4_o16__p_f19
       +Y[jf17]*rate_eval.f17__p_o16
       +Y[jne20]*rate_eval.ne20__he4_o16
       +rho*Y[jhe4]*Y[jc12]*rate_eval.he4_c12__o16
       +rho*Y[jp]*Y[jn15]*rate_eval.p_n15__o16
       +rho*Y[jhe4]*Y[jn13]*rate_eval.he4_n13__p_o16
       +rho*Y[jp]*Y[jf19]*rate_eval.p_f19__he4_o16
       )

    dYdt[jo17] = (
       -rho*Y[jp]*Y[jo17]*rate_eval.p_o17__f18
       -rho*Y[jp]*Y[jo17]*rate_eval.p_o17__he4_n14
       +Y[jf17]*rate_eval.f17__o17__weak__wc12
       +Y[jf18]*rate_eval.f18__p_o17
       +rho*Y[jhe4]*Y[jn14]*rate_eval.he4_n14__p_o17
       )

    dYdt[jf17] = (
       -Y[jf17]*rate_eval.f17__o17__weak__wc12
       -Y[jf17]*rate_eval.f17__p_o16
       -rho*Y[jhe4]*Y[jf17]*rate_eval.he4_f17__p_ne20
       +rho*Y[jp]*Y[jo16]*rate_eval.p_o16__f17
       +rho*Y[jp]*Y[jne20]*rate_eval.p_ne20__he4_f17
       )

    dYdt[jf18] = (
       -Y[jf18]*rate_eval.f18__p_o17
       -Y[jf18]*rate_eval.f18__he4_n14
       -rho*Y[jp]*Y[jf18]*rate_eval.p_f18__he4_o15
       +rho*Y[jhe4]*Y[jn14]*rate_eval.he4_n14__f18
       +rho*Y[jp]*Y[jo17]*rate_eval.p_o17__f18
       +rho*Y[jhe4]*Y[jo15]*rate_eval.he4_o15__p_f18
       )

    dYdt[jf19] = (
       -Y[jf19]*rate_eval.f19__he4_n15
       -rho*Y[jp]*Y[jf19]*rate_eval.p_f19__ne20
       -rho*Y[jp]*Y[jf19]*rate_eval.p_f19__he4_o16
       +Y[jne20]*rate_eval.ne20__p_f19
       +rho*Y[jhe4]*Y[jn15]*rate_eval.he4_n15__f19
       +rho*Y[jhe4]*Y[jo16]*rate_eval.he4_o16__p_f19
       )

    dYdt[jne20] = (
       -Y[jne20]*rate_eval.ne20__p_f19
       -Y[jne20]*rate_eval.ne20__he4_o16
       -rho*Y[jp]*Y[jne20]*rate_eval.p_ne20__he4_f17
       -rho*Y[jhe4]*Y[jne20]*rate_eval.he4_ne20__c12_c12
       +rho*Y[jhe4]*Y[jo16]*rate_eval.he4_o16__ne20
       +rho*Y[jp]*Y[jf19]*rate_eval.p_f19__ne20
       +5.00000000000000e-01*rho*Y[jc12]**2*rate_eval.c12_c12__he4_ne20
       +rho*Y[jhe4]*Y[jf17]*rate_eval.he4_f17__p_ne20
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    n13__c13__weak__wc12(rate_eval, tf)
    o15__n15__weak__wc12(rate_eval, tf)
    f17__o17__weak__wc12(rate_eval, tf)
    n13__p_c12(rate_eval, tf)
    n14__p_c13(rate_eval, tf)
    o15__p_n14(rate_eval, tf)
    o16__p_n15(rate_eval, tf)
    o16__he4_c12(rate_eval, tf)
    f17__p_o16(rate_eval, tf)
    f18__p_o17(rate_eval, tf)
    f18__he4_n14(rate_eval, tf)
    f19__he4_n15(rate_eval, tf)
    ne20__p_f19(rate_eval, tf)
    ne20__he4_o16(rate_eval, tf)
    c12__he4_he4_he4(rate_eval, tf)
    p_c12__n13(rate_eval, tf)
    he4_c12__o16(rate_eval, tf)
    p_c13__n14(rate_eval, tf)
    p_n14__o15(rate_eval, tf)
    he4_n14__f18(rate_eval, tf)
    p_n15__o16(rate_eval, tf)
    he4_n15__f19(rate_eval, tf)
    p_o16__f17(rate_eval, tf)
    he4_o16__ne20(rate_eval, tf)
    p_o17__f18(rate_eval, tf)
    p_f19__ne20(rate_eval, tf)
    he4_c12__p_n15(rate_eval, tf)
    c12_c12__he4_ne20(rate_eval, tf)
    he4_n13__p_o16(rate_eval, tf)
    he4_n14__p_o17(rate_eval, tf)
    p_n15__he4_c12(rate_eval, tf)
    he4_o15__p_f18(rate_eval, tf)
    p_o16__he4_n13(rate_eval, tf)
    he4_o16__p_f19(rate_eval, tf)
    p_o17__he4_n14(rate_eval, tf)
    he4_f17__p_ne20(rate_eval, tf)
    p_f18__he4_o15(rate_eval, tf)
    p_f19__he4_o16(rate_eval, tf)
    p_ne20__he4_f17(rate_eval, tf)
    he4_ne20__c12_c12(rate_eval, tf)
    he4_he4_he4__c12(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c12__n13 *= scor

        scn_fac = ScreenFactors(2, 4, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_c12__o16 *= scor
        rate_eval.he4_c12__p_n15 *= scor

        scn_fac = ScreenFactors(1, 1, 6, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_c13__n14 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n14__o15 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 14)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n14__f18 *= scor
        rate_eval.he4_n14__p_o17 *= scor

        scn_fac = ScreenFactors(1, 1, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_n15__o16 *= scor
        rate_eval.p_n15__he4_c12 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n15__f19 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o16__f17 *= scor
        rate_eval.p_o16__he4_n13 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 16)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_o16__ne20 *= scor
        rate_eval.he4_o16__p_f19 *= scor

        scn_fac = ScreenFactors(1, 1, 8, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_o17__f18 *= scor
        rate_eval.p_o17__he4_n14 *= scor

        scn_fac = ScreenFactors(1, 1, 9, 19)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_f19__ne20 *= scor
        rate_eval.p_f19__he4_o16 *= scor

        scn_fac = ScreenFactors(6, 12, 6, 12)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.c12_c12__he4_ne20 *= scor

        scn_fac = ScreenFactors(2, 4, 7, 13)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_n13__p_o16 *= scor

        scn_fac = ScreenFactors(2, 4, 8, 15)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_o15__p_f18 *= scor

        scn_fac = ScreenFactors(2, 4, 9, 17)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_f17__p_ne20 *= scor

        scn_fac = ScreenFactors(1, 1, 9, 18)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_f18__he4_o15 *= scor

        scn_fac = ScreenFactors(1, 1, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_ne20__he4_f17 *= scor

        scn_fac = ScreenFactors(2, 4, 10, 20)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_ne20__c12_c12 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.he4_he4_he4__c12 *= scor * scor2

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jp, jp] = (
       -rho*Y[jc12]*rate_eval.p_c12__n13
       -rho*Y[jc13]*rate_eval.p_c13__n14
       -rho*Y[jn14]*rate_eval.p_n14__o15
       -rho*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jo16]*rate_eval.p_o16__f17
       -rho*Y[jo17]*rate_eval.p_o17__f18
       -rho*Y[jf19]*rate_eval.p_f19__ne20
       -rho*Y[jn15]*rate_eval.p_n15__he4_c12
       -rho*Y[jo16]*rate_eval.p_o16__he4_n13
       -rho*Y[jo17]*rate_eval.p_o17__he4_n14
       -rho*Y[jf18]*rate_eval.p_f18__he4_o15
       -rho*Y[jf19]*rate_eval.p_f19__he4_o16
       -rho*Y[jne20]*rate_eval.p_ne20__he4_f17
       )

    jac[jp, jhe4] = (
       +rho*Y[jc12]*rate_eval.he4_c12__p_n15
       +rho*Y[jn13]*rate_eval.he4_n13__p_o16
       +rho*Y[jn14]*rate_eval.he4_n14__p_o17
       +rho*Y[jo15]*rate_eval.he4_o15__p_f18
       +rho*Y[jo16]*rate_eval.he4_o16__p_f19
       +rho*Y[jf17]*rate_eval.he4_f17__p_ne20
       )

    jac[jp, jc12] = (
       -rho*Y[jp]*rate_eval.p_c12__n13
       +rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       )

    jac[jp, jc13] = (
       -rho*Y[jp]*rate_eval.p_c13__n14
       )

    jac[jp, jn13] = (
       +rate_eval.n13__p_c12
       +rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       )

    jac[jp, jn14] = (
       -rho*Y[jp]*rate_eval.p_n14__o15
       +rate_eval.n14__p_c13
       +rho*Y[jhe4]*rate_eval.he4_n14__p_o17
       )

    jac[jp, jn15] = (
       -rho*Y[jp]*rate_eval.p_n15__o16
       -rho*Y[jp]*rate_eval.p_n15__he4_c12
       )

    jac[jp, jo15] = (
       +rate_eval.o15__p_n14
       +rho*Y[jhe4]*rate_eval.he4_o15__p_f18
       )

    jac[jp, jo16] = (
       -rho*Y[jp]*rate_eval.p_o16__f17
       -rho*Y[jp]*rate_eval.p_o16__he4_n13
       +rate_eval.o16__p_n15
       +rho*Y[jhe4]*rate_eval.he4_o16__p_f19
       )

    jac[jp, jo17] = (
       -rho*Y[jp]*rate_eval.p_o17__f18
       -rho*Y[jp]*rate_eval.p_o17__he4_n14
       )

    jac[jp, jf17] = (
       +rate_eval.f17__p_o16
       +rho*Y[jhe4]*rate_eval.he4_f17__p_ne20
       )

    jac[jp, jf18] = (
       -rho*Y[jp]*rate_eval.p_f18__he4_o15
       +rate_eval.f18__p_o17
       )

    jac[jp, jf19] = (
       -rho*Y[jp]*rate_eval.p_f19__ne20
       -rho*Y[jp]*rate_eval.p_f19__he4_o16
       )

    jac[jp, jne20] = (
       -rho*Y[jp]*rate_eval.p_ne20__he4_f17
       +rate_eval.ne20__p_f19
       )

    jac[jhe4, jp] = (
       +rho*Y[jn15]*rate_eval.p_n15__he4_c12
       +rho*Y[jo16]*rate_eval.p_o16__he4_n13
       +rho*Y[jo17]*rate_eval.p_o17__he4_n14
       +rho*Y[jf18]*rate_eval.p_f18__he4_o15
       +rho*Y[jf19]*rate_eval.p_f19__he4_o16
       +rho*Y[jne20]*rate_eval.p_ne20__he4_f17
       )

    jac[jhe4, jhe4] = (
       -rho*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jn14]*rate_eval.he4_n14__f18
       -rho*Y[jn15]*rate_eval.he4_n15__f19
       -rho*Y[jo16]*rate_eval.he4_o16__ne20
       -rho*Y[jc12]*rate_eval.he4_c12__p_n15
       -rho*Y[jn13]*rate_eval.he4_n13__p_o16
       -rho*Y[jn14]*rate_eval.he4_n14__p_o17
       -rho*Y[jo15]*rate_eval.he4_o15__p_f18
       -rho*Y[jo16]*rate_eval.he4_o16__p_f19
       -rho*Y[jf17]*rate_eval.he4_f17__p_ne20
       -rho*Y[jne20]*rate_eval.he4_ne20__c12_c12
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__c12
       )

    jac[jhe4, jc12] = (
       -rho*Y[jhe4]*rate_eval.he4_c12__o16
       -rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       +3*rate_eval.c12__he4_he4_he4
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.c12_c12__he4_ne20
       )

    jac[jhe4, jn13] = (
       -rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       )

    jac[jhe4, jn14] = (
       -rho*Y[jhe4]*rate_eval.he4_n14__f18
       -rho*Y[jhe4]*rate_eval.he4_n14__p_o17
       )

    jac[jhe4, jn15] = (
       -rho*Y[jhe4]*rate_eval.he4_n15__f19
       +rho*Y[jp]*rate_eval.p_n15__he4_c12
       )

    jac[jhe4, jo15] = (
       -rho*Y[jhe4]*rate_eval.he4_o15__p_f18
       )

    jac[jhe4, jo16] = (
       -rho*Y[jhe4]*rate_eval.he4_o16__ne20
       -rho*Y[jhe4]*rate_eval.he4_o16__p_f19
       +rate_eval.o16__he4_c12
       +rho*Y[jp]*rate_eval.p_o16__he4_n13
       )

    jac[jhe4, jo17] = (
       +rho*Y[jp]*rate_eval.p_o17__he4_n14
       )

    jac[jhe4, jf17] = (
       -rho*Y[jhe4]*rate_eval.he4_f17__p_ne20
       )

    jac[jhe4, jf18] = (
       +rate_eval.f18__he4_n14
       +rho*Y[jp]*rate_eval.p_f18__he4_o15
       )

    jac[jhe4, jf19] = (
       +rate_eval.f19__he4_n15
       +rho*Y[jp]*rate_eval.p_f19__he4_o16
       )

    jac[jhe4, jne20] = (
       -rho*Y[jhe4]*rate_eval.he4_ne20__c12_c12
       +rate_eval.ne20__he4_o16
       +rho*Y[jp]*rate_eval.p_ne20__he4_f17
       )

    jac[jc12, jp] = (
       -rho*Y[jc12]*rate_eval.p_c12__n13
       +rho*Y[jn15]*rate_eval.p_n15__he4_c12
       )

    jac[jc12, jhe4] = (
       -rho*Y[jc12]*rate_eval.he4_c12__o16
       -rho*Y[jc12]*rate_eval.he4_c12__p_n15
       +2*rho*Y[jne20]*rate_eval.he4_ne20__c12_c12
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__c12
       )

    jac[jc12, jc12] = (
       -rate_eval.c12__he4_he4_he4
       -rho*Y[jp]*rate_eval.p_c12__n13
       -rho*Y[jhe4]*rate_eval.he4_c12__o16
       -rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       -2*5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.c12_c12__he4_ne20
       )

    jac[jc12, jn13] = (
       +rate_eval.n13__p_c12
       )

    jac[jc12, jn15] = (
       +rho*Y[jp]*rate_eval.p_n15__he4_c12
       )

    jac[jc12, jo16] = (
       +rate_eval.o16__he4_c12
       )

    jac[jc12, jne20] = (
       +2*rho*Y[jhe4]*rate_eval.he4_ne20__c12_c12
       )

    jac[jc13, jp] = (
       -rho*Y[jc13]*rate_eval.p_c13__n14
       )

    jac[jc13, jc13] = (
       -rho*Y[jp]*rate_eval.p_c13__n14
       )

    jac[jc13, jn13] = (
       +rate_eval.n13__c13__weak__wc12
       )

    jac[jc13, jn14] = (
       +rate_eval.n14__p_c13
       )

    jac[jn13, jp] = (
       +rho*Y[jc12]*rate_eval.p_c12__n13
       +rho*Y[jo16]*rate_eval.p_o16__he4_n13
       )

    jac[jn13, jhe4] = (
       -rho*Y[jn13]*rate_eval.he4_n13__p_o16
       )

    jac[jn13, jc12] = (
       +rho*Y[jp]*rate_eval.p_c12__n13
       )

    jac[jn13, jn13] = (
       -rate_eval.n13__c13__weak__wc12
       -rate_eval.n13__p_c12
       -rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       )

    jac[jn13, jo16] = (
       +rho*Y[jp]*rate_eval.p_o16__he4_n13
       )

    jac[jn14, jp] = (
       -rho*Y[jn14]*rate_eval.p_n14__o15
       +rho*Y[jc13]*rate_eval.p_c13__n14
       +rho*Y[jo17]*rate_eval.p_o17__he4_n14
       )

    jac[jn14, jhe4] = (
       -rho*Y[jn14]*rate_eval.he4_n14__f18
       -rho*Y[jn14]*rate_eval.he4_n14__p_o17
       )

    jac[jn14, jc13] = (
       +rho*Y[jp]*rate_eval.p_c13__n14
       )

    jac[jn14, jn14] = (
       -rate_eval.n14__p_c13
       -rho*Y[jp]*rate_eval.p_n14__o15
       -rho*Y[jhe4]*rate_eval.he4_n14__f18
       -rho*Y[jhe4]*rate_eval.he4_n14__p_o17
       )

    jac[jn14, jo15] = (
       +rate_eval.o15__p_n14
       )

    jac[jn14, jo17] = (
       +rho*Y[jp]*rate_eval.p_o17__he4_n14
       )

    jac[jn14, jf18] = (
       +rate_eval.f18__he4_n14
       )

    jac[jn15, jp] = (
       -rho*Y[jn15]*rate_eval.p_n15__o16
       -rho*Y[jn15]*rate_eval.p_n15__he4_c12
       )

    jac[jn15, jhe4] = (
       -rho*Y[jn15]*rate_eval.he4_n15__f19
       +rho*Y[jc12]*rate_eval.he4_c12__p_n15
       )

    jac[jn15, jc12] = (
       +rho*Y[jhe4]*rate_eval.he4_c12__p_n15
       )

    jac[jn15, jn15] = (
       -rho*Y[jp]*rate_eval.p_n15__o16
       -rho*Y[jhe4]*rate_eval.he4_n15__f19
       -rho*Y[jp]*rate_eval.p_n15__he4_c12
       )

    jac[jn15, jo15] = (
       +rate_eval.o15__n15__weak__wc12
       )

    jac[jn15, jo16] = (
       +rate_eval.o16__p_n15
       )

    jac[jn15, jf19] = (
       +rate_eval.f19__he4_n15
       )

    jac[jo15, jp] = (
       +rho*Y[jn14]*rate_eval.p_n14__o15
       +rho*Y[jf18]*rate_eval.p_f18__he4_o15
       )

    jac[jo15, jhe4] = (
       -rho*Y[jo15]*rate_eval.he4_o15__p_f18
       )

    jac[jo15, jn14] = (
       +rho*Y[jp]*rate_eval.p_n14__o15
       )

    jac[jo15, jo15] = (
       -rate_eval.o15__n15__weak__wc12
       -rate_eval.o15__p_n14
       -rho*Y[jhe4]*rate_eval.he4_o15__p_f18
       )

    jac[jo15, jf18] = (
       +rho*Y[jp]*rate_eval.p_f18__he4_o15
       )

    jac[jo16, jp] = (
       -rho*Y[jo16]*rate_eval.p_o16__f17
       -rho*Y[jo16]*rate_eval.p_o16__he4_n13
       +rho*Y[jn15]*rate_eval.p_n15__o16
       +rho*Y[jf19]*rate_eval.p_f19__he4_o16
       )

    jac[jo16, jhe4] = (
       -rho*Y[jo16]*rate_eval.he4_o16__ne20
       -rho*Y[jo16]*rate_eval.he4_o16__p_f19
       +rho*Y[jc12]*rate_eval.he4_c12__o16
       +rho*Y[jn13]*rate_eval.he4_n13__p_o16
       )

    jac[jo16, jc12] = (
       +rho*Y[jhe4]*rate_eval.he4_c12__o16
       )

    jac[jo16, jn13] = (
       +rho*Y[jhe4]*rate_eval.he4_n13__p_o16
       )

    jac[jo16, jn15] = (
       +rho*Y[jp]*rate_eval.p_n15__o16
       )

    jac[jo16, jo16] = (
       -rate_eval.o16__p_n15
       -rate_eval.o16__he4_c12
       -rho*Y[jp]*rate_eval.p_o16__f17
       -rho*Y[jhe4]*rate_eval.he4_o16__ne20
       -rho*Y[jp]*rate_eval.p_o16__he4_n13
       -rho*Y[jhe4]*rate_eval.he4_o16__p_f19
       )

    jac[jo16, jf17] = (
       +rate_eval.f17__p_o16
       )

    jac[jo16, jf19] = (
       +rho*Y[jp]*rate_eval.p_f19__he4_o16
       )

    jac[jo16, jne20] = (
       +rate_eval.ne20__he4_o16
       )

    jac[jo17, jp] = (
       -rho*Y[jo17]*rate_eval.p_o17__f18
       -rho*Y[jo17]*rate_eval.p_o17__he4_n14
       )

    jac[jo17, jhe4] = (
       +rho*Y[jn14]*rate_eval.he4_n14__p_o17
       )

    jac[jo17, jn14] = (
       +rho*Y[jhe4]*rate_eval.he4_n14__p_o17
       )

    jac[jo17, jo17] = (
       -rho*Y[jp]*rate_eval.p_o17__f18
       -rho*Y[jp]*rate_eval.p_o17__he4_n14
       )

    jac[jo17, jf17] = (
       +rate_eval.f17__o17__weak__wc12
       )

    jac[jo17, jf18] = (
       +rate_eval.f18__p_o17
       )

    jac[jf17, jp] = (
       +rho*Y[jo16]*rate_eval.p_o16__f17
       +rho*Y[jne20]*rate_eval.p_ne20__he4_f17
       )

    jac[jf17, jhe4] = (
       -rho*Y[jf17]*rate_eval.he4_f17__p_ne20
       )

    jac[jf17, jo16] = (
       +rho*Y[jp]*rate_eval.p_o16__f17
       )

    jac[jf17, jf17] = (
       -rate_eval.f17__o17__weak__wc12
       -rate_eval.f17__p_o16
       -rho*Y[jhe4]*rate_eval.he4_f17__p_ne20
       )

    jac[jf17, jne20] = (
       +rho*Y[jp]*rate_eval.p_ne20__he4_f17
       )

    jac[jf18, jp] = (
       -rho*Y[jf18]*rate_eval.p_f18__he4_o15
       +rho*Y[jo17]*rate_eval.p_o17__f18
       )

    jac[jf18, jhe4] = (
       +rho*Y[jn14]*rate_eval.he4_n14__f18
       +rho*Y[jo15]*rate_eval.he4_o15__p_f18
       )

    jac[jf18, jn14] = (
       +rho*Y[jhe4]*rate_eval.he4_n14__f18
       )

    jac[jf18, jo15] = (
       +rho*Y[jhe4]*rate_eval.he4_o15__p_f18
       )

    jac[jf18, jo17] = (
       +rho*Y[jp]*rate_eval.p_o17__f18
       )

    jac[jf18, jf18] = (
       -rate_eval.f18__p_o17
       -rate_eval.f18__he4_n14
       -rho*Y[jp]*rate_eval.p_f18__he4_o15
       )

    jac[jf19, jp] = (
       -rho*Y[jf19]*rate_eval.p_f19__ne20
       -rho*Y[jf19]*rate_eval.p_f19__he4_o16
       )

    jac[jf19, jhe4] = (
       +rho*Y[jn15]*rate_eval.he4_n15__f19
       +rho*Y[jo16]*rate_eval.he4_o16__p_f19
       )

    jac[jf19, jn15] = (
       +rho*Y[jhe4]*rate_eval.he4_n15__f19
       )

    jac[jf19, jo16] = (
       +rho*Y[jhe4]*rate_eval.he4_o16__p_f19
       )

    jac[jf19, jf19] = (
       -rate_eval.f19__he4_n15
       -rho*Y[jp]*rate_eval.p_f19__ne20
       -rho*Y[jp]*rate_eval.p_f19__he4_o16
       )

    jac[jf19, jne20] = (
       +rate_eval.ne20__p_f19
       )

    jac[jne20, jp] = (
       -rho*Y[jne20]*rate_eval.p_ne20__he4_f17
       +rho*Y[jf19]*rate_eval.p_f19__ne20
       )

    jac[jne20, jhe4] = (
       -rho*Y[jne20]*rate_eval.he4_ne20__c12_c12
       +rho*Y[jo16]*rate_eval.he4_o16__ne20
       +rho*Y[jf17]*rate_eval.he4_f17__p_ne20
       )

    jac[jne20, jc12] = (
       +5.00000000000000e-01*rho*2*Y[jc12]*rate_eval.c12_c12__he4_ne20
       )

    jac[jne20, jo16] = (
       +rho*Y[jhe4]*rate_eval.he4_o16__ne20
       )

    jac[jne20, jf17] = (
       +rho*Y[jhe4]*rate_eval.he4_f17__p_ne20
       )

    jac[jne20, jf19] = (
       +rho*Y[jp]*rate_eval.p_f19__ne20
       )

    jac[jne20, jne20] = (
       -rate_eval.ne20__p_f19
       -rate_eval.ne20__he4_o16
       -rho*Y[jp]*rate_eval.p_ne20__he4_f17
       -rho*Y[jhe4]*rate_eval.he4_ne20__c12_c12
       )

    return jac
