# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python(comb_demo)
#     language: python
#     name: comb_demo
# ---

# %% [markdown]
# # Conformal Change Point Detection demo

# %%
from functools import partial
from CP import pValues
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, \
    interp1d

import param
from matplotlib.figure import Figure
import panel as pn
import numpy as np
import scipy.stats as ss
from sklearn.model_selection import train_test_split

# %%
from IPython.display import display, HTML

display(HTML("<style>.container {width: 90% !important;} </style>"))

# %%

# pn.extension('mathjax', comm='ipywidgets')
pn.extension('mathjax', comms='vscode')


notes = pn.pane.LaTeX(r"""<h1>Conformal Change-Point Detection</h1>
The framework generally accepted for Change-Point Detection posits a sequence of random variates $X_1, X_2, \dots$, which in a first phase are
each generated independently from a same distribution $p(x)$ up to an index $\tau$ which is a priori unknown. In the second phase, that is, $t < \tau$, the variates $X_t$ are 
generated from a different distrbution $q(x)$.<br>
The problem is to detect the change as quickly as possible.<br>
Historically, this problem was studied in the context of industrial quality control to monitor manufacturing processes. 
For this reason, the first phase is referred to as "in-control" and the second phase as "out-of-control".
A common class of methods computes a statistics from the sequence of variates and triggers an alarm (i.e. detects a change) 
when the statistic exceeds a preset threshold.
The statistic is designed to exhibit small fluctuations in the in-control phase and to diverge during the out-of-control phase.
The lower the threshold, the quickest the detection during the out-of-control phase, but also the higher the chance of false alarm
during the in-control phase.<br>
The established methods, namely CUSUM and Shiryaev-Roberts, require the knowledge of the in-control probability density $p(x)$ as
well as of the out-of-control probability density $q(x)$.<br>
The conformal CPD approach relies only on the availability of a sufficiently sample of in-control variates.

<h2>The demo</h2>
In this demo, the user is presented with a number of controls (sliders, selection boxes) on the left and with charts on the right.
Going from top to bottom, random variates are generated, p-values are calculated, then transformed into Bayes Factors, and finally 
a CPD statistic is computed.

<h4>Synthetic Data Set</h4>
The in-control (IC) and out-of-control (OOC) distributions are both Gaussian. The user can change mean and variance. 
It is also possible to vary the number of samples for each phase.
The IC variates are represented in green, the OOC variates in red.
A calibration set -- with the IC mean and variance -- is also created (but not shown). The user can control the size of the calibration set.<br>
P-values are computed for the IC and OOC samples and shown in the second chart from the top.
A very simple non-conformity measure is used: $\alpha_i = \left | x_i \right |$.

<h4>Betting Function</h4>
The p-value is transformed via a "calibrator" or "betting function" into a Bayes Factor or e-value.
The following choices of calibrator are offered:<br>
\[
\begin{array}{lc}
 \text{power calibrator} &  k p ^ {k-1}\\ 
 \text{simple mixture} &  \frac{1-p+p\log p }{p \log^2 p }\\  
 \text{linear} & 2(1-p) \\
 \text{negative logarithm} & -\log(p)    
\end{array}
\]
<br>
The computation of the betting function can also be bypassed by choosing "None".

<h4>CPD Statistic</h4>
The user can choose among various methods: <br>
\[
\begin{array}{lc}
 \text{CUSUM}            &  S_0=1, \, S_{i+1} = s_i \cdot \max(1,S_i) \\ 
 \text{Shiryaev-Roberts} &  S_0=0, \, S_{i+1} = s_i \cdot (1+S_i) \\  
 \text{First moment}^*   &  S_0=0, \, S_{i+1} = (s_1-\mu)+S_i \\
 \text{Product}^*        &  S_0=0, \, S_{i+1} = (\log(s_i)+1) + S_i   
\end{array}
\]
<br>
The methods marked with an asterisk apply when the betting function is bypassed.<br>
The user can choose the value of the threshold.
The chart shows the behaviour of the statistic and the counts of the alarms, both during the IC phase as well as the OOC phase.
""", name="Notes")

#%%
# srv = notes.show()

# %%
def plot_samples(ic_samples, ooc_samples):
    f = Figure(figsize=(12, 3.1))
    ax_a = f.add_subplot(1, 1, 1)
    x = np.arange(0, ic_samples.shape[0]+ooc_samples.shape[0])
    ax_a.plot(x[:ic_samples.shape[0]], ic_samples, "g.",
              label="in-control")
    ax_a.plot(x[ic_samples.shape[0]:], ooc_samples, "r.",
              label="out-of-control")
    ax_a.set_title("Samples")
    ax_a.legend()
    ax_a.set_xlabel('"Time"')
    f.tight_layout()

    return f


class Synthetic_Data_Set(param.Parameterized):
    N_in_control = param.Integer(default=2000, bounds=(100, 10000))
    N_out_of_control = param.Integer(default=2000, bounds=(100, 10000))
    N_calibration = param.Integer(default=5000, bounds=(100, 10000))
    in_control_mean = param.Number(default=0.0, bounds=(-1.0, 1.0))
    in_control_var = param.Number(default=1.0, bounds=(0.1, 10.0))
    out_of_control_mean = param.Number(default=1.0, bounds=(-2.0, 2.0))
    out_of_control_var = param.Number(default=1.0, bounds=(0.1, 10.0))
    seed = param.Integer(default=0, bounds=(0, 32767))

    # Outputs
    output = param.Dict(default=dict(),
                        precedence=-1)  # To have all updates in one go

    n = 2

    def __init__(self, **params):
        super(Synthetic_Data_Set, self).__init__(**params)
        self.update()

    def update(self):
        output = dict()

        np.random.seed(self.seed)

        try:
            in_control_samples = ss.norm(loc=self.in_control_mean, scale=np.sqrt(self.in_control_var)).rvs(
                size=(self.N_in_control,))
            out_of_control_samples = ss.norm(loc=self.out_of_control_mean, scale=np.sqrt(self.out_of_control_var)).rvs(
                size=(self.N_out_of_control,))
            calibration_samples = ss.norm(loc=self.in_control_mean, scale=np.sqrt(self.in_control_var)).rvs(
                size=(self.N_calibration,))
        except np.linalg.LinAlgError:
            placeholder = np.array([0.0, 1.0])
            output['in_control_samples'] = placeholder
            output['out_of_control_samples'] = placeholder
            output['calibration_samples'] = placeholder
            self.output = output
            return

        output['in_control_samples'] = in_control_samples
        output['out_of_control_samples'] = out_of_control_samples
        output['calibration_samples'] = calibration_samples

        self.output = output

    @pn.depends("N_in_control", "N_out_of_control", "N_calibration",
                "in_control_mean", "in_control_var",
                "out_of_control_mean", "out_of_control_var", "seed")
    def view(self):
        self.update()
        f = plot_samples(
            self.output['in_control_samples'], self.output['out_of_control_samples'])
        return f

    def view2(self):
        return "# %d" % self.N


sd = Synthetic_Data_Set()
# %%
# sd_panel = pn.Row(sd, sd.view)

# srv = sd_panel.show()
# %%
# srv.stop()


# %%


def ecdf(x):
    v, c = np.unique(x, return_counts='true')
    q = np.cumsum(c) / np.sum(c)
    return v, q


def ECDF_cal_p(p_test, p_cal):
    v, q = ecdf(p_cal)
    v = np.concatenate(([0], v))
    q = np.concatenate(([0], q))
    us = interp1d(v, q, bounds_error=False, fill_value=(0, 1))
    return us(p_test)


# %%
# MICP

# %%


# %%
def plot_pVals(ic_samples, ooc_samples):
    f = Figure(figsize=(12, 3.1))
    ax_a = f.add_subplot(1, 1, 1)
    x = np.arange(0, ic_samples.shape[0]+ooc_samples.shape[0])
    ax_a.plot(x[:ic_samples.shape[0]], ic_samples, "g.",
              label="in-control")
    ax_a.plot(x[ic_samples.shape[0]:], ooc_samples, "r.",
              label="out-of-control")
    ax_a.set_title("p-values")
    ax_a.legend()
    ax_a.set_xlabel('"Time"')
    f.tight_layout()

    return f


# %%

def ncm(scores):
    return np.abs(scores)


class MICP(param.Parameterized):
    sd = param.Parameter(precedence=-1)
    ic_pVals = param.Array(precedence=-1)
    ooc_pVals = param.Array(precedence=-1)

    def __init__(self, sd, **params):
        self.sd = sd
        super(MICP, self).__init__(**params)
        self.update()

    def aux_update_(self, in_control_samples, out_of_control_samples, calibration_samples):
        randomize = True

        with param.batch_watch(self):
            self.ic_pVals = pValues(
                calibrationAlphas=ncm(calibration_samples),
                testAlphas=ncm(in_control_samples),
                randomized=randomize)
            self.ooc_pVals = pValues(
                calibrationAlphas=ncm(calibration_samples),
                testAlphas=ncm(out_of_control_samples),
                randomized=randomize)

    @pn.depends("sd.output", watch=True)
    def update(self):
        self.aux_update_(**self.sd.output)

    @pn.depends("ic_pVals", "ooc_pVals")
    def view(self):
        return pn.Column(
            plot_pVals(self.ic_pVals, self.ooc_pVals)
        )


# %%
# Now we compute the p-values with Mondrian Inductive
# %%
micp = MICP(sd)


# %%
micp_panel = pn.Row(micp.sd.param, pn.Column(micp.sd.view,
                                             micp.view))
# %%
# micp_panel

# %%


def power_calibrator(p, k):
    return k*(p**(k-1))


def simple_mixture(p):
    return (1-p+p*np.log(p))/(p*(np.log(p)*np.log(p)))


def plot_bf(ic_samples, ooc_samples):
    f = Figure(figsize=(12, 3.1))
    ax_a = f.add_subplot(1, 1, 1)
    x = np.arange(0, ic_samples.shape[0]+ooc_samples.shape[0])
    ax_a.plot(x[:ic_samples.shape[0]], ic_samples, "g.",
              label="in-control")
    ax_a.plot(x[ic_samples.shape[0]:], ooc_samples, "r.",
              label="out-of-control")
    ax_a.set_title("Bayes Factors")
    ax_a.legend()
    ax_a.set_xlabel('"Time"')
    f.tight_layout()

    return f


class Betting_Function(param.Parameterized):
    micp = param.Parameter(precedence=-1)
    k = param.Number(default=0.8, bounds=(0.0, 1.0))
    ic_bf = param.Array(precedence=-1)
    ooc_bf = param.Array(precedence=-1)
    betting_function = param.ObjectSelector(default="Power calibrator",
                                            objects=["Power calibrator",
                                                     "Simple mixture",
                                                     "Linear",
                                                     "Negative logarithm",
                                                     "None"])

    def __init__(self, micp, **params):
        self.micp = micp
        super(Betting_Function, self).__init__(**params)
        self.update()

    @pn.depends("k", "betting_function", "micp.ic_pVals", "micp.ooc_pVals", watch=True)
    def update(self):
        bf_dict = {
            "Power calibrator": partial(power_calibrator, k=self.k),
            "Simple mixture": simple_mixture,
            "Linear": lambda p: 2*(1-p),
            "Negative logarithm": lambda p: -np.log(p),
            "None": lambda p: p,
        }
        with param.batch_watch(self):
            self.ic_bf = bf_dict[self.betting_function](self.micp.ic_pVals)
            self.ooc_bf = bf_dict[self.betting_function](self.micp.ooc_pVals)

    @pn.depends("ic_bf", "ic_bf")
    def view(self):
        return pn.Row(
            plot_bf(self.ic_bf, self.ooc_bf)
        )


# %%
bf = Betting_Function(micp)
# %%
bf_panel = pn.Column(pn.Row(micp.sd.param, pn.Column(micp.sd.view,
                                                     micp.view)),
                     pn.Row(bf.param, bf.view))

# %%
# srv = bf_panel.show()

# %%
# srv.stop()

# %%

def CPD_r(s, x, s_0, thr):
    """return statistic for change-point detection V_n = s(V_{n-1})*(x_n)
    The statistic is reset to s_0 when the threshold is exceeded.
    Appropriate choices for s and s_0 give rise to CUSUM and S-R.
    NOTE: differs from the usual definition by not having f() and g()."""

    S = np.empty_like(x)
    S[0] = s_0
    for i in range(1, len(x)):
        if S[i-1] > thr:
            prev = s_0
        else:
            prev = S[i-1]
        S[i] = s(prev)*x[i]
    return S


def Conformal_r(x, c, thr):
    """Compute the CPD statistic with conformal method"""
    S = np.empty_like(x)
    S[0] = c(x[0])
    for i in range(1, len(x)):
        if np.abs(S[i-1]) > thr:
            # if S[i-1] > thr:
            S[i] = c(x[i])
        else:
            S[i] = c(x[i])+S[i-1]
    return S


def CPD_Conf(x, r, r_0, thr):
    """Vovk's conformal procedure"""
    ratios = x[1:]/x[:-1]
    R = np.empty_like(x)
    R[0] = r_0
    for i in range(1, len(R)):
        if R[i-1] > thr:
            prev = r_0
        else:
            prev = R[i-1]
        R[i] = r(prev)*ratios[i-1]
    return R


# %%


def plot_martingale(mart, ic_size, thr):
    f = Figure(figsize=(12, 3.1))
    ax_a = f.add_subplot(1, 1, 1)
    x = np.arange(0, mart.shape[0])
    ax_a.plot(x[:ic_size], mart[:ic_size], "g.",
              label="in-control")
    ax_a.plot(x[ic_size:], mart[ic_size:], "r.",
              label="out-of-control")
    ic_alarms = np.sum(np.abs(mart[:ic_size]) >= thr)
    ooc_alarms = np.sum(np.abs(mart[ic_size:]) >= thr)
    ax_a.set_title(f"Martingale (CPD statistic)")
    ax_a.legend()
    ax_a.set_xlabel('"Time"')
    ax_a.annotate(f"Alarms\nin-control: {ic_alarms}, out-of-control:{ooc_alarms}",
                  xy=(0.5, 0.9), xycoords='axes fraction', va='top', ha='center',
                  bbox=dict(boxstyle="round", fc="yellow", alpha=0.5), fontsize=14)
    
    f.tight_layout()

    return f


class Martingale(param.Parameterized):
    bf = param.Parameter(precedence=-1)
    threshold = param.Number(default=100.0, bounds=(0.0, 1000.0))
    mart = param.Array(precedence=-1)
    method = param.ObjectSelector(default="CUSUM",
                                  objects=["CUSUM",
                                           "Shiryaev-Roberts",
                                           "Product (log)",
                                           "First moment", ])

    def __init__(self, micp, **params):
        self.bf = bf
        super(Martingale, self).__init__(**params)
        self.update()

    @pn.depends("threshold", "method", "bf.ic_bf", "bf.ooc_bf", watch=True)
    def update(self):
        method_dict = {
            "CUSUM": partial(CPD_r, s=lambda x: max(1, x), s_0=1),
            "Shiryaev-Roberts": partial(CPD_r, s=lambda x: 1+x, s_0=0),
            "First moment": partial(Conformal_r, c=lambda x: x-0.5),
            "Product (log)": partial(Conformal_r, c=lambda x: np.log(x)+1),
        }
        stat = method_dict[self.method]
        with param.batch_watch(self):
            self.mart = stat(x=np.concatenate((
                self.bf.ic_bf, self.bf.ooc_bf)), thr=self.threshold)

    @pn.depends("mart")
    def view(self):
        return pn.Row(
            plot_martingale(self.mart, self.bf.ic_bf.shape[0], self.threshold)
        )


# %%
cpd = Martingale(bf)
# %%
cpd_panel = pn.Column("<h1>Conformal Change Point Detection</h1>",
                      pn.Row(micp.sd.param, pn.Column(micp.sd.view,
                                                      micp.view)),
                      pn.Row(bf.param, bf.view),
                      pn.Row(cpd.param, cpd.view),
                      name="Conf CPD")

# %%

# cpd_panel
# %%
# srv = cpd_panel.show()

# %%
cpd_demo = pn.Tabs(cpd_panel, notes)
cpd_demo
# %%
# srv = cpd_demo.show()
# %%
# srv.stop()
# %%
