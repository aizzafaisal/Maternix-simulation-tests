"""
Aim 2 - Optical detection: spectral / optical-budget model  (verification by analysis)
-------------------------------------------------------------------------------------
Models the proposed light path:  470 nm LED ring -> SYBR Green I (DNA-bound) ->
515 nm long-pass barrier filter -> smartphone camera (green channel).

All spectra are ANALYTIC APPROXIMATIONS fitted to published peak positions/widths,
not measured data.  Every assumed value is in PARAMS and is varied in a sensitivity
analysis so the conclusions do not hinge on any single number.
"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)
lam = np.arange(380.0, 721.0, 0.5)           # nm
dl = lam[1] - lam[0]

def gauss(c, fwhm, a=1.0):
    s = fwhm / 2.3548
    return a * np.exp(-0.5 * ((lam - c) / s) ** 2)

def lorentz(c, fwhm, a=1.0):
    g = fwhm / 2
    return a * g ** 2 / ((lam - c) ** 2 + g ** 2)

def norm_peak(y):
    return y / y.max()

# ---------------- assumed parameters (documented in the report table) ----------------
PARAMS = dict(
    led_peak=470.0, led_fwhm=25.0, led_tail=0.02,     # royal-blue LED: Gaussian core + 2% Lorentzian tail
    ex_peak=497.0, ex_fwhm=36.0,                        # SYBR Green I bound: abs max 497 nm
    em_peak=520.0, em_fwhm=32.0,                        # SYBR Green I bound: em max 520 nm
    lp_edge=515.0,                                      # proposed barrier filter
    eps=7.3e4,            # M^-1 cm^-1 at abs max  (ASSUMED, order of magnitude for cyanine dyes)
    conc_uM=2.0,          # ~1x SYBR in PCR mix (ASSUMED)
    path_cm=0.3,          # optical path through PCR tube (ASSUMED, ~3 mm)
    qy_bound=0.8,         # ASSUMED (high QY when bound)
    enhancement=1000.0,   # bound/free fluorescence enhancement (ASSUMED, sensitivity 100-3000)
    rho_scatter=0.05,     # diffuse reflectance of matte black housing/tube seen by camera (ASSUMED)
)

def led(p, peak=None, fwhm=None):
    peak = p["led_peak"] if peak is None else peak
    fwhm = p["led_fwhm"] if fwhm is None else fwhm
    return norm_peak(gauss(peak, fwhm) + lorentz(peak, fwhm * 1.5, p["led_tail"]))

def sybr_ex(p):   # main band + short-wavelength shoulder typical of cyanines
    return norm_peak(gauss(p["ex_peak"], p["ex_fwhm"]) + gauss(p["ex_peak"] - 28, 34, 0.30))

def sybr_em(p):   # main band + red vibronic tail
    return norm_peak(gauss(p["em_peak"], p["em_fwhm"]) + gauss(p["em_peak"] + 30, 48, 0.32))

def longpass(edge, width, tmax, block):
    t = tmax / (1 + np.exp(-(lam - edge) / width))
    return np.maximum(t, block)

def shortpass(edge, width, tmax, block):
    t = tmax / (1 + np.exp((lam - edge) / width))
    return np.maximum(t, block)

def bandpass(c, fwhm, tmax, block):
    return np.maximum(tmax * (1 / (1 + np.exp(-(lam - (c - fwhm / 2)) / 2.0))) *
                      (1 / (1 + np.exp((lam - (c + fwhm / 2)) / 2.0))), block)

def cam_green():   # generic CMOS Bayer green channel with IR-cut (approx.)
    return norm_peak(gauss(535, 95))

def cam_blue():
    return norm_peak(gauss(465, 80))

def integ(y):
    return float(np.sum(y) * dl)

def budget(p, led_spec, exc_filter, em_filter, cam=None):
    """Relative photon budget per unit LED optical power.
    Returns fluorescence (bound, free), scatter background and figures of merit."""
    cam = cam_green() if cam is None else cam
    L = led_spec * exc_filter
    L_n = L / integ(led_spec)                                        # per unit emitted LED power
    ex = sybr_ex(p)
    # absorbed fraction (thin-sample approx): A(l) = 1 - 10^(-eps(l) c l)
    absorb = 1 - 10 ** (-(p["eps"] * ex) * p["conc_uM"] * 1e-6 * p["path_cm"])
    absorbed = integ(L_n * absorb)                                   # fraction of LED power absorbed
    em = sybr_em(p); em_n = em / integ(em)
    det_em = integ(em_n * em_filter * cam)                           # fraction of emission detected
    geom_f, geom_s = 1 / (4 * np.pi), 1 / np.pi                      # isotropic vs Lambertian into same lens
    F_bound = absorbed * p["qy_bound"] * det_em * geom_f
    F_free = F_bound / p["enhancement"]
    S_scatter = p["rho_scatter"] * integ(L_n * em_filter * cam) * geom_s
    ex_eff = integ(L_n * ex)                                         # excitation overlap (0..1 scale)
    return dict(absorbed=absorbed, det_em=det_em, ex_overlap=ex_eff,
                F_bound=F_bound, F_free=F_free, S_scatter=S_scatter,
                SBR=F_bound / S_scatter, pos_neg_contrast=(F_bound + S_scatter) / (F_free + S_scatter))

CONFIGS = {
    "A  Proposed: 470 LED + 515 LP (interference, OD4)":
        dict(led=dict(), exf=np.ones_like(lam), emf=longpass(515, 1.5, 0.93, 1e-4)),
    "B  Low-cost: 470 LED + 515 LP (coloured gel, soft edge, OD2)":
        dict(led=dict(), exf=np.ones_like(lam), emf=longpass(515, 7.0, 0.85, 1e-2)),
    "C  Improved: 470 LED + 490 SP excitation clean-up + 515 LP":
        dict(led=dict(), exf=shortpass(490, 1.5, 0.92, 1e-4), emf=longpass(515, 1.5, 0.93, 1e-4)),
    "D  Alternative: 470 LED + 490 SP + 530/40 band-pass":
        dict(led=dict(), exf=shortpass(490, 1.5, 0.92, 1e-4), emf=bandpass(530, 40, 0.92, 1e-4)),
    "E  No barrier filter (camera only)":
        dict(led=dict(), exf=np.ones_like(lam), emf=np.ones_like(lam)),
}

def run_configs(p):
    res = {}
    for name, c in CONFIGS.items():
        res[name] = budget(p, led(p, **c["led"]), c["exf"], c["emf"])
    return res

def led_peak_sweep(p):
    out = []
    for pk in np.arange(440, 506, 2.5):
        b = budget(p, led(p, peak=pk), np.ones_like(lam), longpass(515, 1.5, 0.93, 1e-4))
        out.append((pk, b["ex_overlap"], b["SBR"], b["F_bound"]))
    return np.array(out)

def sensitivity(p):
    base = run_configs(p)["A  Proposed: 470 LED + 515 LP (interference, OD4)"]
    rows = []
    for key, lo, hi in [("rho_scatter", 0.01, 0.30), ("enhancement", 100, 3000), ("qy_bound", 0.4, 0.9),
                        ("led_tail", 0.0, 0.05), ("led_fwhm", 18, 35), ("conc_uM", 0.5, 4.0)]:
        vals = []
        for v in (lo, hi):
            q = dict(p); q[key] = v
            vals.append(run_configs(q)["A  Proposed: 470 LED + 515 LP (interference, OD4)"])
        rows.append(dict(param=key, lo=lo, hi=hi,
                         SBR_lo=vals[0]["SBR"], SBR_hi=vals[1]["SBR"],
                         C_lo=vals[0]["pos_neg_contrast"], C_hi=vals[1]["pos_neg_contrast"]))
    return base, rows

if __name__ == "__main__":
    p = PARAMS
    res = run_configs(p)
    sweep = led_peak_sweep(p)
    base, sens = sensitivity(p)
    json.dump(dict(configs=res, sens=sens, params=p, sweep=sweep.tolist()), open(f"{OUT}/spectral_results.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k:62s} ex_overlap {v['ex_overlap']:.3f}  absorbed {v['absorbed']*100:.2f}%  det_em {v['det_em']*100:5.1f}%  SBR {v['SBR']:8.2f}  pos/neg {v['pos_neg_contrast']:7.2f}")
    print("\nSensitivity (config A):")
    for r in sens:
        print(f"  {r['param']:12s} {r['lo']:>7}..{r['hi']:<7} SBR {r['SBR_lo']:.2f}..{r['SBR_hi']:.2f}  contrast {r['C_lo']:.2f}..{r['C_hi']:.2f}")
    best = sweep[np.argmax(sweep[:, 2])]
    print("\nLED peak sweep: best SBR at", best[0], "nm; at 470:", sweep[np.argmin(abs(sweep[:,0]-470))])
