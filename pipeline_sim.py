"""
Aim 2 - Simulated smartphone fluorescence read-out pipeline + virtual validation
--------------------------------------------------------------------------------
Physics-based image simulation (shot noise, read noise, vignetting, quantisation,
saturation, ambient light leakage, LED drift, phone placement error) feeding the
proposed analysis pipeline:
   locked exposure -> registration on fiducial markers -> ROI medians ->
   background subtraction -> normalisation to fluorescent reference patch ->
   positive / negative / invalid call (with on-cartridge positive & negative controls).

Virtual versions of the proposal's validation approaches:
   V0  threshold dilution series  (detection threshold, LOD95, false-positive rate)
   V1  inter-device: 3 phones x 3 operators, coefficient of variation
   V2  ambient light: dim / office / overhead fluorescent, sealed vs leaky vs open
   V3  LED output drift (battery sag) robustness
All results are SIMULATED. Seeds are fixed so every number is reproducible.
"""
import numpy as np, json, os, csv
import spectral_model as sm

OUT = sm.OUT
rng_global = np.random.default_rng(2026)

# ---------------- optical terms (relative, per W/m^2 LED irradiance) from spectral model ----------------
P = sm.PARAMS
DESIGN = os.environ.get("DESIGN", "A  Proposed: 470 LED + 515 LP (interference, OD4)")
cfg = sm.CONFIGS[DESIGN]
B = sm.budget(P, sm.led(P), cfg["exf"], cfg["emf"])
F_BOUND, F_FREE, SCAT = B["F_bound"], B["F_free"], B["S_scatter"]

# ambient: tri-phosphor fluorescent spectrum, detected through barrier filter + green channel
amb_spec = sm.norm_peak(sm.gauss(435, 12, 0.6) + sm.gauss(545, 10, 1.0) + sm.gauss(611, 12, 0.8) + 0.08)
AMB_PER_W = sm.integ(amb_spec / sm.integ(amb_spec) * cfg["emf"] * sm.cam_green()) / np.pi   # per unit reflectance
RHO_AMB = dict(bg=0.05, well=0.25, ref=0.15, marker=0.10)   # ambient reflectance: matte housing / glossy tube / acrylic ref (ASSUMED)
LUX_PER_W = 300.0            # luminous efficacy of radiation for fluorescent white light (approx.)
E_LED = 100.0                # W/m^2 LED irradiance at sample plane (ASSUMED: ~6 x 20 mW LEDs over ~12 cm^2)
REF_LEVEL = 0.6              # fluorescent reference patch = 60% of fully-bound SYBR signal
K_SAT = 2e10                 # dsDNA copies giving 50% dye binding (ASSUMED)

LIGHT = {"dim (50 lx)": 50, "office (500 lx)": 500, "overhead fluorescent (1500 lx)": 1500}
SEAL = {"sealed + foam seams": 1e-4, "unsealed seams": 1e-2, "no enclosure": 1.0}

PHONES = {   # sensitivity scale, read noise e-, full well e-, vignetting, green-channel gain mismatch
    "Phone A (flagship)": dict(k=1.00, rn=2.0, fw=6000, vig=0.15, bits=10),
    "Phone B (mid-range)": dict(k=0.75, rn=3.0, fw=5000, vig=0.25, bits=10),
    "Phone C (budget)":    dict(k=0.50, rn=5.0, fw=4000, vig=0.35, bits=8),
}
OPERATORS = {"Operator 1": 0.3, "Operator 2": 0.8, "Operator 3": 1.5}   # placement error SD in mm (jig play)

PX_PER_MM = 8.0
H, W = 260, 420
# layout (mm from image centre) : reference patch, 2 registration markers, wells
REF_POS = (-20.0, 0.0); MARKERS = [(-23.0, -12.0), (23.0, 12.0)]
WELL_R_MM = 2.2

EXPO = None   # set so Phone A, nominal LED, fully-bound well ~ 55% full well


def fluor(fbound):
    return F_FREE + (F_BOUND - F_FREE) * fbound


def render(wells, phone, op_sd, rng, lux=500, tau=1e-4, led_scale=1.0, misreg=None):
    """wells: list of (x_mm, y_mm, fbound). Returns 2D image (DN) and true offsets."""
    ph = PHONES[phone]
    dx, dy = (rng.normal(0, op_sd, 2) if misreg is None else misreg)
    yy, xx = np.mgrid[0:H, 0:W]
    X = (xx - W / 2) / PX_PER_MM - dx
    Y = (yy - H / 2) / PX_PER_MM - dy
    sig = np.full((H, W), SCAT)                                    # matte background scatter
    rho = np.full((H, W), RHO_AMB["bg"])
    def disk(cx, cy, r): return ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2
    for (cx, cy, fb) in wells:
        m = disk(cx, cy, WELL_R_MM); sig = np.where(m, SCAT + fluor(fb), sig); rho = np.where(m, RHO_AMB["well"], rho)
    sq = (abs(X - REF_POS[0]) <= 2.5) & (abs(Y - REF_POS[1]) <= 2.5)
    sig = np.where(sq, SCAT + REF_LEVEL * F_BOUND, sig); rho = np.where(sq, RHO_AMB["ref"], rho)
    for (mx, my) in MARKERS:
        m = disk(mx, my, 1.0); sig = np.where(m, SCAT + 1.5 * F_BOUND, sig); rho = np.where(m, RHO_AMB["marker"], rho)
    led_term = sig * E_LED * led_scale
    # leaked room light: surface-dependent reflectance + left-right gradient (window / overhead tube)
    grad = 1 + 0.5 * (xx - W / 2) / (W / 2)
    amb_term = tau * (lux / LUX_PER_W) * AMB_PER_W * rho * grad
    vig = 1 - ph["vig"] * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    electrons = EXPO * ph["k"] * (led_term + amb_term) * vig
    e = rng.poisson(np.clip(electrons, 0, None)) + rng.normal(0, ph["rn"], electrons.shape) + 20
    dn_max = 2 ** ph["bits"] - 1
    dn = np.clip(np.round(e / ph["fw"] * dn_max), 0, dn_max)
    return dn, (dx, dy), dn_max


def analyse(img, dn_max, wells_xy, register=True):
    """Proposed pipeline. Returns dict with raw means, ratio and flags."""
    # 1. registration: brightest compact blobs near expected marker positions
    off = np.array([0.0, 0.0]); found = True
    if register:
        ests = []
        for (mx, my) in MARKERS:
            cx, cy = W / 2 + mx * PX_PER_MM, H / 2 + my * PX_PER_MM
            r = int(3 * PX_PER_MM)
            y0, y1, x0, x1 = int(cy - r), int(cy + r), int(cx - r), int(cx + r)
            sub = img[max(y0, 0):y1, max(x0, 0):x1].astype(float)
            if sub.size == 0: found = False; continue
            thr = sub.min() + 0.75 * (sub.max() - sub.min())
            m = sub >= thr
            if m.sum() < 5 or (sub.max() - np.median(sub)) < 10: found = False; continue
            yy, xx = np.nonzero(m)
            ests.append(((xx.mean() + max(x0, 0) - cx) / PX_PER_MM, (yy.mean() + max(y0, 0) - cy) / PX_PER_MM))
        if ests: off = np.mean(ests, axis=0)
    yy, xx = np.mgrid[0:H, 0:W]
    X = (xx - W / 2) / PX_PER_MM - off[0]; Y = (yy - H / 2) / PX_PER_MM - off[1]
    def roi_disk(cx, cy, r): return ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2
    bg_mask = (abs(Y) > 16) & (abs(Y) < 20) & (abs(X) < 30)
    bg = np.median(img[bg_mask])
    ref_mask = (abs(X - REF_POS[0]) <= 1.8) & (abs(Y - REF_POS[1]) <= 1.8)
    ref = np.median(img[ref_mask])
    vals, sat = [], 0.0
    for (cx, cy) in wells_xy:
        m = roi_disk(cx, cy, WELL_R_MM * 0.6)
        vals.append(np.median(img[m])); sat = max(sat, np.mean(img[m] >= dn_max))
    sat = max(sat, np.mean(img[ref_mask] >= dn_max))
    ratio = [(v - bg) / (ref - bg) if (ref - bg) > 1e-9 else np.nan for v in vals]
    # sanity check: registration must land the reference ROI on the reference patch
    if (ref - bg) < 0.3 * max(np.max(np.asarray(vals) - bg), 1e-9) or (ref - bg) <= 1e-9:
        found = False
    return dict(raw=[v - bg for v in vals], ratio=ratio, ref=ref - bg, bg=bg, sat=sat, found=found)


def call(res, idx_sample, idx_pc, idx_nc, T, min_ref=15):
    r = res["ratio"]
    if (not res["found"]) or res["sat"] > 0.01 or res["ref"] < min_ref or np.isnan(r[idx_sample]):
        return "invalid"
    if r[idx_pc] < T or r[idx_nc] >= T:
        return "invalid"
    return "positive" if r[idx_sample] >= T else "negative"


# ---------------- PCR end-point model ----------------
def pcr_endpoint(n0_mean, rng, cycles=40, nmax=1e12):
    n0 = rng.poisson(n0_mean)
    E = np.clip(rng.normal(0.92, 0.03), 0.75, 1.0)
    pd = 10 ** rng.normal(8.7, 0.45)                      # primer-dimer / non-specific dsDNA (ASSUMED)
    N = float(n0)
    for _ in range(cycles):
        N = N + E * N * (1 - N / nmax)
    D = N + pd * (1 - N / nmax)
    return D / (D + K_SAT), n0


CART = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]     # sample, positive control, negative control


def cart_wells(fb_sample, rng):
    fb_pc, _ = pcr_endpoint(1e4, rng); fb_nc, _ = pcr_endpoint(0, rng)
    return [(CART[0][0], CART[0][1], fb_sample), (CART[1][0], CART[1][1], fb_pc), (CART[2][0], CART[2][1], fb_nc)]


PANEL_FB = [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]   # calibration panel (fraction-bound equivalents)
PANEL = [(-8 + 5.0 * i, -8.0 if i % 2 == 0 else 8.0, fb) for i, fb in enumerate(PANEL_FB)]


def main():
    global EXPO
    EXPO = 1.0
    EXPO = 0.55 * PHONES["Phone A (flagship)"]["fw"] / (PHONES["Phone A (flagship)"]["k"] * (SCAT + F_BOUND) * E_LED)
    rng = np.random.default_rng(42)
    results = {}

    # ---- V0a: calibration set to set threshold (independent of test set) ----
    cal = []
    for _ in range(60):
        fb, _ = pcr_endpoint(0, rng)
        img, _, mx = render(cart_wells(fb, rng), "Phone A (flagship)", 0.5, rng)
        cal.append(analyse(img, mx, CART)["ratio"][0])
    cal = np.array(cal)
    T = float(cal.mean() + 5 * cal.std())                    # threshold = NTC mean + 5 SD
    results["threshold"] = dict(T=T, ntc_mean=float(cal.mean()), ntc_sd=float(cal.std()), n=len(cal))

    # ---- V0b: dilution series test set ----
    levels = [0, 1, 2, 3, 5, 10, 30, 100, 1e3, 1e4, 1e5, 1e6]
    reps = 40
    dil = []
    for lv in levels:
        n = 200 if lv == 0 else reps
        for _ in range(n):
            fb, n0 = pcr_endpoint(lv, rng)
            img, _, mx = render(cart_wells(fb, rng), "Phone A (flagship)", 0.5, rng)
            r = analyse(img, mx, CART); c = call(r, 0, 1, 2, T)
            dil.append(dict(level=lv, n0=int(n0), ratio=float(r["ratio"][0]), call=c))
    results["dilution"] = dil
    # detection summary
    summ = []
    for lv in levels:
        rows = [d for d in dil if d["level"] == lv]
        valid = [d for d in rows if d["call"] != "invalid"]
        pos = sum(d["call"] == "positive" for d in valid)
        summ.append(dict(level=lv, n=len(rows), valid=len(valid), positive=pos,
                         rate=pos / max(len(valid), 1), theory_any_copy=1 - np.exp(-lv)))
    results["dilution_summary"] = summ
    lod = next((s["level"] for s in summ if s["level"] > 0 and s["rate"] >= 0.95 and
                all(t["rate"] >= 0.95 for t in summ if t["level"] >= s["level"])), None)
    # templated (>=1 copy) reactions only: optical sensitivity
    t1 = [d for d in dil if d["n0"] >= 1 and d["call"] != "invalid"]
    t0 = [d for d in dil if d["n0"] == 0 and d["call"] != "invalid"]
    results["lod95_copies"] = lod
    results["sens_given_template"] = sum(d["call"] == "positive" for d in t1) / len(t1)
    results["specificity_ntc"] = sum(d["call"] == "negative" for d in t0) / len(t0)

    # ---- V1: inter-device x operator on calibration panel ----
    inter = []
    for ph in PHONES:
        for op, sd in OPERATORS.items():
            for rep in range(5):
                img, (dx, dy), mx = render(PANEL, ph, sd, rng)
                xy = [(x, y) for x, y, _ in PANEL]
                r_reg = analyse(img, mx, xy, register=True); r_no = analyse(img, mx, xy, register=False)
                inter.append(dict(phone=ph, op=op, rep=rep, raw=r_reg["raw"], ratio=r_reg["ratio"], ok=bool(r_reg["found"]),
                                  ratio_noreg=r_no["ratio"], off=(float(dx), float(dy))))
    cv = []
    okc = [d for d in inter if d["ok"]]
    results["interdevice_rejected"] = len(inter) - len(okc)
    for i, fb in enumerate(PANEL_FB):
        if fb == 0: continue
        raw = np.array([d["raw"][i] for d in okc]); rat = np.array([d["ratio"][i] for d in okc])
        nor = np.array([d["ratio_noreg"][i] for d in okc])
        cv.append(dict(level=fb, cv_raw=float(raw.std() / raw.mean() * 100), cv_norm=float(rat.std() / rat.mean() * 100),
                       cv_norm_noreg=float(nor.std() / nor.mean() * 100), mean_ratio=float(rat.mean())))
    results["interdevice_cv"] = cv
    results["interdevice"] = inter

    # ---- V2: ambient light x enclosure seal (classification agreement) ----
    amb = []
    # fixed reference set of reactions: 30 positive (100 copies), 30 negative (NTC)
    rxn = [(cart_wells(pcr_endpoint(100, rng)[0], rng), "positive") for _ in range(30)] + \
          [(cart_wells(pcr_endpoint(0, rng)[0], rng), "negative") for _ in range(30)]
    def run_cond(tau, lux):
        calls, ratios = [], []
        for wl, truth in rxn:
            img, _, mx = render(wl, "Phone B (mid-range)", 0.8, rng, lux=lux, tau=tau)
            r = analyse(img, mx, CART); calls.append(call(r, 0, 1, 2, T)); ratios.append(r["ratio"][0])
        return calls, ratios
    ref_calls, ref_r = run_cond(1e-4, 50)                      # reference: sealed, dim
    for seal, tau in SEAL.items():
        for light, lux in LIGHT.items():
            calls, ratios = run_cond(tau, lux)
            agree = np.mean([a == b for a, b in zip(calls, ref_calls)])
            acc = np.mean([c == t for c, (_, t) in zip(calls, rxn)])
            amb.append(dict(seal=seal, light=light, lux=lux, agreement=float(agree), accuracy=float(acc),
                            invalid=float(np.mean([c == "invalid" for c in calls])),
                            false_pos=float(np.mean([c == "positive" for c, (_, t) in zip(calls, rxn) if t == "negative"])),
                            neg_ratio_mean=float(np.nanmean(ratios[30:])), pos_ratio_mean=float(np.nanmean(ratios[:30]))))
    results["ambient"] = amb

    # ---- V3: LED drift (battery sag) ----
    drift = []
    for s in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        raws, rats = [], []
        for _ in range(15):
            img, _, mx = render(PANEL, "Phone A (flagship)", 0.5, rng, led_scale=s)
            r = analyse(img, mx, [(x, y) for x, y, _ in PANEL])
            raws.append(r["raw"][5]); rats.append(r["ratio"][5])
        drift.append(dict(led=s, raw=float(np.mean(raws)), ratio=float(np.mean(rats))))
    results["led_drift"] = drift
    results["design"] = DESIGN
    results["optics"] = dict(F_bound=F_BOUND, F_free=F_FREE, scatter=SCAT, amb_per_W=AMB_PER_W, E_led=E_LED)

    json.dump(results, open(f"{OUT}/pipeline_results.json", "w"), indent=1, default=float)
    # example images for figures
    ex = {}
    rng2 = np.random.default_rng(7)
    fbp, _ = pcr_endpoint(1e3, rng2); fbn, _ = pcr_endpoint(0, rng2)
    for tag, fb, ph, lux, tau in [("pos_A", fbp, "Phone A (flagship)", 500, 1e-4), ("neg_A", fbn, "Phone A (flagship)", 500, 1e-4),
                                  ("pos_C", fbp, "Phone C (budget)", 500, 1e-4), ("pos_open", fbp, "Phone B (mid-range)", 1500, 1.0)]:
        img, _, mx = render(cart_wells(fb, rng2), ph, 0.5, rng2, lux=lux, tau=tau)
        ex[tag] = (img / mx).tolist()
    img, _, mx = render(PANEL, "Phone A (flagship)", 0.5, rng2); ex["panel"] = (img / mx).tolist()
    json.dump(ex, open(f"{OUT}/example_images.json", "w"))

    print("Threshold T =", round(T, 4), " NTC mean/sd", round(cal.mean(), 4), round(cal.std(), 4))
    for s in summ: print(f"  {s['level']:>8}: {s['positive']}/{s['valid']} positive (rate {s['rate']:.2f}; theory P(>=1 copy) {s['theory_any_copy']:.2f})")
    print("LOD95 =", lod, "copies/rxn; sensitivity given >=1 copy:", round(results['sens_given_template'], 3), " specificity NTC:", round(results['specificity_ntc'], 3))
    print("inter-device captures rejected (re-seat prompt):", results["interdevice_rejected"], "of", len(inter))
    for c in cv: print(f"  panel {c['level']:.2f}: CV raw {c['cv_raw']:.1f}%  norm {c['cv_norm']:.1f}%  norm(no reg) {c['cv_norm_noreg']:.1f}%")
    for a in amb: print(f"  {a['seal']:22s} {a['light']:30s} agree-vs-ref {a['agreement']*100:5.1f}%  acc {a['accuracy']*100:5.1f}%  invalid {a['invalid']*100:4.1f}%  FP {a['false_pos']*100:4.1f}%  posR {a['pos_ratio_mean']:.3f} negR {a['neg_ratio_mean']:.3f}")
    for d in drift: print(f"  LED x{d['led']:.1f}: raw {d['raw']:.1f}  ratio {d['ratio']:.3f}")


if __name__ == "__main__":
    main()
