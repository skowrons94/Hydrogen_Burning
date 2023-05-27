import numba
import numpy as np
from numba.experimental import jitclass

from pynucastro.rates import Tfactors, _find_rate_file
from pynucastro.screening import PlasmaState, ScreenFactors

jp = 0
jd = 1
jhe3 = 2
jhe4 = 3
jli7 = 4
jbe7 = 5
jbe8 = 6
jb8 = 7
jc12 = 8
nnuc = 9

A = np.zeros((nnuc), dtype=np.int32)

A[jp] = 1
A[jd] = 2
A[jhe3] = 3
A[jhe4] = 4
A[jli7] = 7
A[jbe7] = 7
A[jbe8] = 8
A[jb8] = 8
A[jc12] = 12

Z = np.zeros((nnuc), dtype=np.int32)

Z[jp] = 1
Z[jd] = 1
Z[jhe3] = 2
Z[jhe4] = 2
Z[jli7] = 3
Z[jbe7] = 4
Z[jbe8] = 4
Z[jb8] = 5
Z[jc12] = 6

names = []
names.append("h1")
names.append("h2")
names.append("he3")
names.append("he4")
names.append("li7")
names.append("be7")
names.append("be8")
names.append("b8")
names.append("c12")

@jitclass([
    ("be7__li7__weak__electron_capture", numba.float64),
    ("b8__be8__weak__wc17", numba.float64),
    ("b8__he4_he4__weak__wc12", numba.float64),
    ("p_p__d__weak__bet_pos_", numba.float64),
    ("p_p__d__weak__electron_capture", numba.float64),
    ("p_d__he3", numba.float64),
    ("d_d__he4", numba.float64),
    ("p_he3__he4__weak__bet_pos_", numba.float64),
    ("he4_he3__be7", numba.float64),
    ("p_be7__b8", numba.float64),
    ("d_he3__p_he4", numba.float64),
    ("p_li7__he4_he4", numba.float64),
    ("he3_he3__p_p_he4", numba.float64),
    ("d_be7__p_he4_he4", numba.float64),
    ("he3_be7__p_p_he4_he4", numba.float64),
    ("he4_he4_he4__c12", numba.float64),
])
class RateEval:
    def __init__(self):
        self.be7__li7__weak__electron_capture = np.nan
        self.b8__be8__weak__wc17 = np.nan
        self.b8__he4_he4__weak__wc12 = np.nan
        self.p_p__d__weak__bet_pos_ = np.nan
        self.p_p__d__weak__electron_capture = np.nan
        self.p_d__he3 = np.nan
        self.d_d__he4 = np.nan
        self.p_he3__he4__weak__bet_pos_ = np.nan
        self.he4_he3__be7 = np.nan
        self.p_be7__b8 = np.nan
        self.d_he3__p_he4 = np.nan
        self.p_li7__he4_he4 = np.nan
        self.he3_he3__p_p_he4 = np.nan
        self.d_be7__p_he4_he4 = np.nan
        self.he3_be7__p_p_he4_he4 = np.nan
        self.he4_he4_he4__c12 = np.nan

@numba.njit()
def ye(Y):
    return np.sum(Z * Y)/np.sum(A * Y)

@numba.njit()
def be7__li7__weak__electron_capture(rate_eval, tf):
    # be7 --> li7
    rate = 0.0

    #   ecw
    rate += np.exp(  -23.8328 + 3.02033*tf.T913
                  + -0.0742132*tf.T9 + -0.00792386*tf.T953 + -0.650113*tf.lnT9)

    rate_eval.be7__li7__weak__electron_capture = rate

@numba.njit()
def b8__be8__weak__wc17(rate_eval, tf):
    # b8 --> be8
    rate = 0.0

    # wc17w
    rate += np.exp(  -115.234)

    rate_eval.b8__be8__weak__wc17 = rate

@numba.njit()
def b8__he4_he4__weak__wc12(rate_eval, tf):
    # b8 --> he4 + he4
    rate = 0.0

    # wc12w
    rate += np.exp(  -0.105148)

    rate_eval.b8__he4_he4__weak__wc12 = rate

@numba.njit()
def p_p__d__weak__bet_pos_(rate_eval, tf):
    # p + p --> d
    rate = 0.0

    # bet+w
    rate += np.exp(  -34.7863 + -3.51193*tf.T913i + 3.10086*tf.T913
                  + -0.198314*tf.T9 + 0.0126251*tf.T953 + -1.02517*tf.lnT9)

    rate_eval.p_p__d__weak__bet_pos_ = rate

@numba.njit()
def p_p__d__weak__electron_capture(rate_eval, tf):
    # p + p --> d
    rate = 0.0

    #   ecw
    rate += np.exp(  -43.6499 + -0.00246064*tf.T9i + -2.7507*tf.T913i + -0.424877*tf.T913
                  + 0.015987*tf.T9 + -0.000690875*tf.T953 + -0.207625*tf.lnT9)

    rate_eval.p_p__d__weak__electron_capture = rate

@numba.njit()
def p_d__he3(rate_eval, tf):
    # d + p --> he3
    rate = 0.0

    # de04 
    rate += np.exp(  8.93525 + -3.7208*tf.T913i + 0.198654*tf.T913
                  + 0.333333*tf.lnT9)
    # de04n
    rate += np.exp(  7.52898 + -3.7208*tf.T913i + 0.871782*tf.T913
                  + -0.666667*tf.lnT9)

    rate_eval.p_d__he3 = rate

@numba.njit()
def d_d__he4(rate_eval, tf):
    # d + d --> he4
    rate = 0.0

    # nacrn
    rate += np.exp(  3.78177 + -4.26166*tf.T913i + -0.119233*tf.T913
                  + 0.778829*tf.T9 + -0.0925203*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.d_d__he4 = rate

@numba.njit()
def p_he3__he4__weak__bet_pos_(rate_eval, tf):
    # he3 + p --> he4
    rate = 0.0

    # bet+w
    rate += np.exp(  -27.7611 + -4.30107e-12*tf.T9i + -6.141*tf.T913i + -1.93473e-09*tf.T913
                  + 2.04145e-10*tf.T9 + -1.80372e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.p_he3__he4__weak__bet_pos_ = rate

@numba.njit()
def he4_he3__be7(rate_eval, tf):
    # he3 + he4 --> be7
    rate = 0.0

    # cd08n
    rate += np.exp(  17.7075 + -12.8271*tf.T913i + -3.8126*tf.T913
                  + 0.0942285*tf.T9 + -0.00301018*tf.T953 + 1.33333*tf.lnT9)
    # cd08n
    rate += np.exp(  15.6099 + -12.8271*tf.T913i + -0.0308225*tf.T913
                  + -0.654685*tf.T9 + 0.0896331*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he4_he3__be7 = rate

@numba.njit()
def p_be7__b8(rate_eval, tf):
    # be7 + p --> b8
    rate = 0.0

    # nacrn
    rate += np.exp(  12.5315 + -10.264*tf.T913i + -0.203472*tf.T913
                  + 0.121083*tf.T9 + -0.00700063*tf.T953 + -0.666667*tf.lnT9)
    # nacrr
    rate += np.exp(  7.73399 + -7.345*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_be7__b8 = rate

@numba.njit()
def d_he3__p_he4(rate_eval, tf):
    # he3 + d --> p + he4
    rate = 0.0

    # de04 
    rate += np.exp(  24.6839 + -7.182*tf.T913i + 0.473288*tf.T913
                  + 1.46847*tf.T9 + -27.9603*tf.T953 + -0.666667*tf.lnT9)
    # de04 
    rate += np.exp(  41.2969 + -7.182*tf.T913i + -17.1349*tf.T913
                  + 1.36908*tf.T9 + -0.0814423*tf.T953 + 3.35395*tf.lnT9)

    rate_eval.d_he3__p_he4 = rate

@numba.njit()
def p_li7__he4_he4(rate_eval, tf):
    # li7 + p --> he4 + he4
    rate = 0.0

    # de04 
    rate += np.exp(  11.9576 + -8.4727*tf.T913i + 0.417943*tf.T913
                  + 5.34565*tf.T9 + -4.8684*tf.T953 + -0.666667*tf.lnT9)
    # de04r
    rate += np.exp(  21.8999 + -26.1527*tf.T9i
                  + -1.5*tf.lnT9)
    # de04 
    rate += np.exp(  20.4438 + -8.4727*tf.T913i + 0.297934*tf.T913
                  + 0.0582335*tf.T9 + -0.00413383*tf.T953 + -0.666667*tf.lnT9)
    # de04r
    rate += np.exp(  14.2538 + -4.478*tf.T9i
                  + -1.5*tf.lnT9)

    rate_eval.p_li7__he4_he4 = rate

@numba.njit()
def he3_he3__p_p_he4(rate_eval, tf):
    # he3 + he3 --> p + p + he4
    rate = 0.0

    # nacrn
    rate += np.exp(  24.7788 + -12.277*tf.T913i + -0.103699*tf.T913
                  + -0.0649967*tf.T9 + 0.0168191*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he3_he3__p_p_he4 = rate

@numba.njit()
def d_be7__p_he4_he4(rate_eval, tf):
    # be7 + d --> p + he4 + he4
    rate = 0.0

    # cf88n
    rate += np.exp(  27.6987 + -12.428*tf.T913i
                  + -0.666667*tf.lnT9)

    rate_eval.d_be7__p_he4_he4 = rate

@numba.njit()
def he3_be7__p_p_he4_he4(rate_eval, tf):
    # be7 + he3 --> p + p + he4 + he4
    rate = 0.0

    # mafon
    rate += np.exp(  31.7435 + -5.45213e-12*tf.T9i + -21.793*tf.T913i + -1.98126e-09*tf.T913
                  + 1.84204e-10*tf.T9 + -1.46403e-11*tf.T953 + -0.666667*tf.lnT9)

    rate_eval.he3_be7__p_p_he4_he4 = rate

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
    be7__li7__weak__electron_capture(rate_eval, tf)
    b8__be8__weak__wc17(rate_eval, tf)
    b8__he4_he4__weak__wc12(rate_eval, tf)
    p_p__d__weak__bet_pos_(rate_eval, tf)
    p_p__d__weak__electron_capture(rate_eval, tf)
    p_d__he3(rate_eval, tf)
    d_d__he4(rate_eval, tf)
    p_he3__he4__weak__bet_pos_(rate_eval, tf)
    he4_he3__be7(rate_eval, tf)
    p_be7__b8(rate_eval, tf)
    d_he3__p_he4(rate_eval, tf)
    p_li7__he4_he4(rate_eval, tf)
    he3_he3__p_p_he4(rate_eval, tf)
    d_be7__p_he4_he4(rate_eval, tf)
    he3_be7__p_p_he4_he4(rate_eval, tf)
    he4_he4_he4__c12(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p__d__weak__bet_pos_ *= scor
        rate_eval.p_p__d__weak__electron_capture *= scor

        scn_fac = ScreenFactors(1, 1, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_d__he3 *= scor

        scn_fac = ScreenFactors(1, 2, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_d__he4 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he3__he4__weak__bet_pos_ *= scor

        scn_fac = ScreenFactors(2, 4, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_he3__be7 *= scor

        scn_fac = ScreenFactors(1, 1, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_be7__b8 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he3__p_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_li7__he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_he3__p_p_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_be7__p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_be7__p_p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.he4_he4_he4__c12 *= scor * scor2

    dYdt = np.zeros((nnuc), dtype=np.float64)

    dYdt[jp] = (
       -2*5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jp]*Y[jbe7]*rate_eval.p_be7__b8
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       +rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       +2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +2*rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    dYdt[jd] = (
       -rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       -2*5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__he4
       -rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +5.00000000000000e-01*rho*Y[jp]**2*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*Y[jp]**2*rate_eval.p_p__d__weak__electron_capture
       )

    dYdt[jhe3] = (
       -rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__be7
       -rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       -2*5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       -rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +rho*Y[jp]*Y[jd]*rate_eval.p_d__he3
       )

    dYdt[jhe4] = (
       -rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__be7
       -3*1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__c12
       +2*Y[jb8]*rate_eval.b8__he4_he4__weak__wc12
       +5.00000000000000e-01*rho*Y[jd]**2*rate_eval.d_d__he4
       +rho*Y[jp]*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*Y[jhe3]*rate_eval.d_he3__p_he4
       +2*rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       +5.00000000000000e-01*rho*Y[jhe3]**2*rate_eval.he3_he3__p_p_he4
       +2*rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       +2*rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    dYdt[jli7] = (
       -rho*Y[jp]*Y[jli7]*rate_eval.p_li7__he4_he4
       +rho*ye(Y)*Y[jbe7]*rate_eval.be7__li7__weak__electron_capture
       )

    dYdt[jbe7] = (
       -rho*ye(Y)*Y[jbe7]*rate_eval.be7__li7__weak__electron_capture
       -rho*Y[jp]*Y[jbe7]*rate_eval.p_be7__b8
       -rho*Y[jd]*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       -rho*Y[jhe3]*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +rho*Y[jhe3]*Y[jhe4]*rate_eval.he4_he3__be7
       )

    dYdt[jbe8] = (
       +Y[jb8]*rate_eval.b8__be8__weak__wc17
       )

    dYdt[jb8] = (
       -Y[jb8]*rate_eval.b8__be8__weak__wc17
       -Y[jb8]*rate_eval.b8__he4_he4__weak__wc12
       +rho*Y[jp]*Y[jbe7]*rate_eval.p_be7__b8
       )

    dYdt[jc12] = (
       +1.66666666666667e-01*rho**2*Y[jhe4]**3*rate_eval.he4_he4_he4__c12
       )

    return dYdt

def jacobian(t, Y, rho, T, screen_func=None):
    return jacobian_eq(t, Y, rho, T, screen_func)

@numba.njit()
def jacobian_eq(t, Y, rho, T, screen_func):

    tf = Tfactors(T)
    rate_eval = RateEval()

    # reaclib rates
    be7__li7__weak__electron_capture(rate_eval, tf)
    b8__be8__weak__wc17(rate_eval, tf)
    b8__he4_he4__weak__wc12(rate_eval, tf)
    p_p__d__weak__bet_pos_(rate_eval, tf)
    p_p__d__weak__electron_capture(rate_eval, tf)
    p_d__he3(rate_eval, tf)
    d_d__he4(rate_eval, tf)
    p_he3__he4__weak__bet_pos_(rate_eval, tf)
    he4_he3__be7(rate_eval, tf)
    p_be7__b8(rate_eval, tf)
    d_he3__p_he4(rate_eval, tf)
    p_li7__he4_he4(rate_eval, tf)
    he3_he3__p_p_he4(rate_eval, tf)
    d_be7__p_he4_he4(rate_eval, tf)
    he3_be7__p_p_he4_he4(rate_eval, tf)
    he4_he4_he4__c12(rate_eval, tf)

    if screen_func is not None:
        plasma_state = PlasmaState(T, rho, Y, Z)

        scn_fac = ScreenFactors(1, 1, 1, 1)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_p__d__weak__bet_pos_ *= scor
        rate_eval.p_p__d__weak__electron_capture *= scor

        scn_fac = ScreenFactors(1, 1, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_d__he3 *= scor

        scn_fac = ScreenFactors(1, 2, 1, 2)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_d__he4 *= scor

        scn_fac = ScreenFactors(1, 1, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_he3__he4__weak__bet_pos_ *= scor

        scn_fac = ScreenFactors(2, 4, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he4_he3__be7 *= scor

        scn_fac = ScreenFactors(1, 1, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_be7__b8 *= scor

        scn_fac = ScreenFactors(1, 2, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_he3__p_he4 *= scor

        scn_fac = ScreenFactors(1, 1, 3, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.p_li7__he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 2, 3)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_he3__p_p_he4 *= scor

        scn_fac = ScreenFactors(1, 2, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.d_be7__p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 3, 4, 7)
        scor = screen_func(plasma_state, scn_fac)
        rate_eval.he3_be7__p_p_he4_he4 *= scor

        scn_fac = ScreenFactors(2, 4, 2, 4)
        scor = screen_func(plasma_state, scn_fac)
        scn_fac2 = ScreenFactors(2, 4, 4, 8)
        scor2 = screen_func(plasma_state, scn_fac2)
        rate_eval.he4_he4_he4__c12 *= scor * scor2

    jac = np.zeros((nnuc, nnuc), dtype=np.float64)

    jac[jp, jp] = (
       -2*5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       -2*5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       -rho*Y[jd]*rate_eval.p_d__he3
       -rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jbe7]*rate_eval.p_be7__b8
       -rho*Y[jli7]*rate_eval.p_li7__he4_he4
       )

    jac[jp, jd] = (
       -rho*Y[jp]*rate_eval.p_d__he3
       +rho*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       )

    jac[jp, jhe3] = (
       -rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*rate_eval.d_he3__p_he4
       +2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       +2*rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jp, jli7] = (
       -rho*Y[jp]*rate_eval.p_li7__he4_he4
       )

    jac[jp, jbe7] = (
       -rho*Y[jp]*rate_eval.p_be7__b8
       +rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       +2*rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jd, jp] = (
       -rho*Y[jd]*rate_eval.p_d__he3
       +5.00000000000000e-01*rho*2*Y[jp]*rate_eval.p_p__d__weak__bet_pos_
       +5.00000000000000e-01*rho**2*ye(Y)*2*Y[jp]*rate_eval.p_p__d__weak__electron_capture
       )

    jac[jd, jd] = (
       -rho*Y[jp]*rate_eval.p_d__he3
       -2*5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__he4
       -rho*Y[jhe3]*rate_eval.d_he3__p_he4
       -rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       )

    jac[jd, jhe3] = (
       -rho*Y[jd]*rate_eval.d_he3__p_he4
       )

    jac[jd, jbe7] = (
       -rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       )

    jac[jhe3, jp] = (
       -rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*rate_eval.p_d__he3
       )

    jac[jhe3, jd] = (
       -rho*Y[jhe3]*rate_eval.d_he3__p_he4
       +rho*Y[jp]*rate_eval.p_d__he3
       )

    jac[jhe3, jhe3] = (
       -rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       -rho*Y[jhe4]*rate_eval.he4_he3__be7
       -rho*Y[jd]*rate_eval.d_he3__p_he4
       -2*5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       -rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jhe3, jhe4] = (
       -rho*Y[jhe3]*rate_eval.he4_he3__be7
       )

    jac[jhe3, jbe7] = (
       -rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jhe4, jp] = (
       +rho*Y[jhe3]*rate_eval.p_he3__he4__weak__bet_pos_
       +2*rho*Y[jli7]*rate_eval.p_li7__he4_he4
       )

    jac[jhe4, jd] = (
       +5.00000000000000e-01*rho*2*Y[jd]*rate_eval.d_d__he4
       +rho*Y[jhe3]*rate_eval.d_he3__p_he4
       +2*rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       )

    jac[jhe4, jhe3] = (
       -rho*Y[jhe4]*rate_eval.he4_he3__be7
       +rho*Y[jp]*rate_eval.p_he3__he4__weak__bet_pos_
       +rho*Y[jd]*rate_eval.d_he3__p_he4
       +5.00000000000000e-01*rho*2*Y[jhe3]*rate_eval.he3_he3__p_p_he4
       +2*rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jhe4, jhe4] = (
       -rho*Y[jhe3]*rate_eval.he4_he3__be7
       -3*1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__c12
       )

    jac[jhe4, jli7] = (
       +2*rho*Y[jp]*rate_eval.p_li7__he4_he4
       )

    jac[jhe4, jbe7] = (
       +2*rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       +2*rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jhe4, jb8] = (
       +2*rate_eval.b8__he4_he4__weak__wc12
       )

    jac[jli7, jp] = (
       -rho*Y[jli7]*rate_eval.p_li7__he4_he4
       )

    jac[jli7, jli7] = (
       -rho*Y[jp]*rate_eval.p_li7__he4_he4
       )

    jac[jli7, jbe7] = (
       +rho*ye(Y)**rate_eval.be7__li7__weak__electron_capture
       )

    jac[jbe7, jp] = (
       -rho*Y[jbe7]*rate_eval.p_be7__b8
       )

    jac[jbe7, jd] = (
       -rho*Y[jbe7]*rate_eval.d_be7__p_he4_he4
       )

    jac[jbe7, jhe3] = (
       -rho*Y[jbe7]*rate_eval.he3_be7__p_p_he4_he4
       +rho*Y[jhe4]*rate_eval.he4_he3__be7
       )

    jac[jbe7, jhe4] = (
       +rho*Y[jhe3]*rate_eval.he4_he3__be7
       )

    jac[jbe7, jbe7] = (
       -rho*ye(Y)**rate_eval.be7__li7__weak__electron_capture
       -rho*Y[jp]*rate_eval.p_be7__b8
       -rho*Y[jd]*rate_eval.d_be7__p_he4_he4
       -rho*Y[jhe3]*rate_eval.he3_be7__p_p_he4_he4
       )

    jac[jbe8, jb8] = (
       +rate_eval.b8__be8__weak__wc17
       )

    jac[jb8, jp] = (
       +rho*Y[jbe7]*rate_eval.p_be7__b8
       )

    jac[jb8, jbe7] = (
       +rho*Y[jp]*rate_eval.p_be7__b8
       )

    jac[jb8, jb8] = (
       -rate_eval.b8__be8__weak__wc17
       -rate_eval.b8__he4_he4__weak__wc12
       )

    jac[jc12, jhe4] = (
       +1.66666666666667e-01*rho**2*3*Y[jhe4]**2*rate_eval.he4_he4_he4__c12
       )

    return jac
