"""MaterNix virtual test dashboard (Streamlit). Run locally with:  streamlit run app.py"""
import os, io, csv, json, datetime, importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import cartridge_model as cm

st.set_page_config(page_title="MaterNix virtual tests", layout="wide")
DESIGNS = {"A proposed (470 LED + 515 LP)": "A  Proposed: 470 LED + 515 LP (interference, OD4)",
           "B low-cost gel filter": "B  Low-cost: 470 LED + 515 LP (coloured gel, soft edge, OD2)",
           "C improved (+ 490 SP)": "C  Improved: 470 LED + 490 SP excitation clean-up + 515 LP",
           "D 490 SP + 530/40 BP": "D  Alternative: 470 LED + 490 SP + 530/40 band-pass",
           "E no filter": "E  No barrier filter (camera only)"}

def get_ps(design_label):
    import pipeline_sim as ps
    if ps.DESIGN != DESIGNS[design_label]:
        os.environ["DESIGN"] = DESIGNS[design_label]; ps = importlib.reload(ps)
    ph = ps.PHONES["Phone A (flagship)"]; ps.EXPO = 0.55 * ph["fw"] / (ph["k"] * (ps.SCAT + ps.F_BOUND) * ps.E_LED)
    return ps

@st.cache_data
def threshold(design_label):
    ps = get_ps(design_label); rng = np.random.default_rng(42); cal = []
    for _ in range(60):
        fb, _ = ps.pcr_endpoint(0, rng); img, _, mx = ps.render(ps.cart_wells(fb, rng), "Phone A (flagship)", 0.5, rng)
        cal.append(ps.analyse(img, mx, ps.CART)["ratio"][0])
    cal = np.array(cal); return float(cal.mean() + 5 * cal.std())

def load_pipeline(design_label):
    T = threshold(design_label); return get_ps(design_label), T

if "log" not in st.session_state: st.session_state.log = []
def log(test, settings, result, criterion, ok):
    st.session_state.log.append(dict(timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tester=st.session_state.get("tester", ""),
                                     test=test, settings=json.dumps(settings), result=result, criterion=criterion, outcome="PASS" if ok else "FAIL"))
    (st.success if ok else st.error)(f"**{test}**: {result} (criterion {criterion}) → {'PASS' if ok else 'FAIL'}")

st.title("MaterNix virtual test dashboard")
st.caption("Simulated verification-by-analysis tests for the sample-preprocessing cartridge and the smartphone optical read-out. All results are simulated.")
st.text_input("Tester", value="Aizza Faisal", key="tester")
tab_c, tab_o, tab_l = st.tabs(["Cartridge", "Optical", "Log and download"])

# ---------------- cartridge ----------------
with tab_c:
    a, b = st.columns(2)
    with a:
        floor = st.slider("Chamber floor (mm)", 0.5, 4.0, 3.0, 0.25)
        gap = st.slider("Magnet air gap (mm)", 0.1, 2.0, 0.5, 0.1)
        bead = st.selectbox("Bead size (µm)", [1.0, 2.8])
        Br = st.slider("Magnet strength Br (T)", 1.10, 1.48, 1.45, 0.01)
        compressed = st.checkbox("Compressed protocol")
    with b:
        heater = st.slider("Heater temperature (°C)", 70, 100, 95)
        wall = st.slider("Valve wall thickness (mm)", 0.3, 2.5, 1.0, 0.1)
        melt = st.slider("Wax melting point (°C)", 50, 75, 60)
        lysis_T = st.slider("Lysis temperature (°C)", 25, 65, 56)
        dP = st.slider("Channel driving pressure (Pa)", 20, 3000, 500, 10)
        runs = st.slider("Monte Carlo runs", 50, 2000, 300, 50)
        seed = st.number_input("Seed", value=3, step=1)
    if st.button("Run cartridge tests", type="primary"):
        s = dict(floor=floor, gap=gap, bead=bead, Br=Br, compressed=compressed, heater=heater, wall=wall, melt=melt, lysis=lysis_T, dP=dP)
        cap = cm.chamber_volumes(); fill = dict(sample=250, lysis=300, wash1=350, wash2=350, elution=50)
        worst = max(100 * fill[k] / cap[k] for k in cap)
        log("CT1 chamber fill", s, f"max fill {worst:.1f}%", "≤ 70%", worst <= 70)
        t_aq, _ = cm.transfer_time(300, dP, visc=1e-3); t_et, _ = cm.transfer_time(300, dP, visc=2.2e-3)
        log("CT2 channel transfer", s, f"{t_aq:.1f} s buffer / {t_et:.1f} s ethanol", "< 30 s", max(t_aq, t_et) < 30)
        aq = cm.bead_capture(floor, gap, bead, 1e-3, Br=Br); et = cm.bead_capture(floor, gap, bead, 2.2e-3, Br=Br)
        log("CT3 bead capture", s, f"{aq['capture_s']:.0f} s buffer / {et['capture_s']:.0f} s ethanol", "≤ 30 s", et["capture_s"] <= 30)
        with st.spinner("Wax valve heat model…"):
            t_open, hist = cm.wax_open(heater, 25.0, wall, Tm=melt)
        valve_ok = np.isfinite(t_open)
        if valve_ok:
            log("CT4 valve opening", s, f"opens at {t_open:.0f} s", "≤ 120 s", t_open <= 120)
        else:
            log("CT4 valve opening", s, "wax never fully melts within 600 s (heater too cool or wall too thick)", "≤ 120 s", False)
            st.warning("The valve never opens with these settings, so the total time cannot be calculated. Raise the heater temperature, lower the wax melting point or use a thinner wall.")
        margin = melt - max(40, lysis_T); log("CT4 temperature margin", s, f"{margin:.0f} °C", "≥ 5 °C", margin >= 5)
        if valve_ok:
            w = cm.workflow(floor, bead, gap, compressed=compressed, valve_s=t_open)
            log("CT5 sample-to-answer", s, f"{w['total_min']:.1f} min", "≤ 45 min", w["total_min"] <= 45)
            rng = np.random.default_rng(int(seed)); orig = cm.bead_capture; tot = []
            for _ in range(runs):
                Ms = rng.uniform(1.5e4, 4.5e4); vf = rng.uniform(0.8, 1.3); g = rng.uniform(max(0.1, gap - 0.2), gap + 0.5); bb = rng.uniform(Br - 0.15, Br + 0.03)
                cm.bead_capture = lambda f, gg, be, mu, Ms=Ms, vf=vf, g=g, bb=bb: orig(f, g, be, mu * vf, Ms=Ms, Br=bb)
                tot.append(cm.workflow(floor, bead, gap, compressed=compressed, valve_s=t_open)["total_min"])
            cm.bead_capture = orig; tot = np.array(tot); tot = tot[np.isfinite(tot)]; pas = (tot <= 45).mean() * 100 if len(tot) else 0.0
            log("CT6 Monte Carlo", dict(s, runs=runs, seed=int(seed)), f"{pas:.1f}% of runs ≤ 45 min (median {np.median(tot):.1f})", "≥ 95%", pas >= 95)
            fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
            ax[0].barh([n for n, _ in w["steps"]], [v for _, v in w["steps"]], color="#2a78d6"); ax[0].invert_yaxis(); ax[0].set_xlabel("min"); ax[0].set_title(f"Total {w['total_min']:.1f} min", loc="left")
            h = np.array(hist); ax[1].plot(h[:, 0], h[:, 1], color="#eb6834"); ax[1].axhline(melt, ls="--", color="k"); ax[1].axhline(lysis_T, ls=":", color="#6250d6"); ax[1].axvline(t_open, color="#2a78d6")
            ax[1].set_xlabel("s"); ax[1].set_ylabel("°C"); ax[1].set_title(f"Wax opens at {t_open:.0f} s", loc="left")
            ax[2].hist(tot, bins=30, color="#1baf7a"); ax[2].axvline(45, ls="--", color="k"); ax[2].set_xlabel("total min"); ax[2].set_title(f"{pas:.0f}% within 45 min", loc="left")
            fig.tight_layout(); st.pyplot(fig)
        else:
            h = np.array(hist)
            if len(h):
                fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(h[:, 0], h[:, 1], color="#eb6834"); ax.axhline(melt, ls="--", color="k")
                ax.set_xlabel("s"); ax.set_ylabel("°C"); ax.set_title("Wax temperature: never reaches full melt", loc="left"); st.pyplot(fig)

# ---------------- optical ----------------
with tab_o:
    a, b = st.columns(2)
    with a:
        design = st.selectbox("Optical design", list(DESIGNS))
        phone = st.selectbox("Phone", ["Phone A (flagship)", "Phone B (mid-range)", "Phone C (budget)"], index=1)
        lux = st.slider("Room light (lx)", 0, 2000, 500, 50)
        enclosure = st.selectbox("Enclosure", ["sealed + foam seams", "unsealed seams", "no enclosure"])
        led = st.slider("LED output (%)", 40, 130, 100, 5)
    with b:
        copies = st.number_input("DNA copies per reaction", value=100.0, min_value=0.0, step=1.0)
        misplace = st.slider("Phone placement error (mm)", 0.0, 3.0, 0.5, 0.1)
        oseed = st.number_input("Seed ", value=1, step=1)
        reps = st.slider("Replicates per level (dilution)", 5, 60, 20, 5)
        n_each = st.slider("Reactions per condition (ambient)", 10, 60, 20, 10)
    c1, c2, c3 = st.columns(3)
    ps, T = load_pipeline(design)
    s = dict(design=design, phone=phone, lux=lux, enclosure=enclosure, led=led, copies=copies, misplace=misplace, seed=int(oseed))
    if c1.button("Single capture", type="primary"):
        rng = np.random.default_rng(int(oseed)); fb, n0 = ps.pcr_endpoint(copies, rng)
        img, off, mx = ps.render(ps.cart_wells(fb, rng), phone, misplace, rng, lux=lux, tau=ps.SEAL[enclosure], led_scale=led / 100)
        r = ps.analyse(img, mx, ps.CART); c = ps.call(r, 0, 1, 2, T)
        st.write(f"Template copies in tube: **{n0}** · threshold T = **{T:.3f}** · background {r['bg']:.1f} DN · reference {r['ref']:.1f} DN · saturation {r['sat']*100:.2f}%")
        st.table({"Region": ["Sample", "Positive control", "Negative control"], "Signal (DN)": [round(float(x), 1) for x in r["raw"]], "R": [round(float(x), 3) for x in r["ratio"]]})
        log("Single capture", s, f"call {c.upper()}", "record only", True)
        fig, ax = plt.subplots(figsize=(7, 4.2)); ax.imshow(img / mx, cmap="gray", vmin=0, vmax=1); ax.axis("off")
        for (x, y), n in zip(ps.CART, ["S", "PC", "NC"]):
            cx, cy = ps.W / 2 + x * ps.PX_PER_MM, ps.H / 2 + y * ps.PX_PER_MM
            ax.add_patch(plt.Circle((cx, cy), ps.WELL_R_MM * ps.PX_PER_MM * 1.3, fill=False, color="orange")); ax.text(cx, cy - 28, n, color="orange", ha="center")
        ax.set_title(f"{c.upper()} · {copies:g} copies · {phone} · {lux} lx · {enclosure}", loc="left", fontsize=9); st.pyplot(fig)
    if c2.button("Dilution series"):
        with st.spinner("Running dilution series…"):
            rng = np.random.default_rng(int(oseed)); levels = [0, 1, 2, 3, 5, 10, 100, 1e4]; rows = []
            for lv in levels:
                for _ in range(reps * (3 if lv == 0 else 1)):
                    fb, n0 = ps.pcr_endpoint(lv, rng); img, _, mx = ps.render(ps.cart_wells(fb, rng), phone, 0.5, rng)
                    rows.append((lv, n0, ps.call(ps.analyse(img, mx, ps.CART), 0, 1, 2, T)))
        rate = []
        for lv in levels:
            v = [c for l, n, c in rows if l == lv and c != "invalid"]; rate.append(sum(c == "positive" for c in v) / max(len(v), 1))
        lod = next((lv for lv, r in zip(levels, rate) if lv > 0 and r >= .95 and all(r2 >= .95 for l2, r2 in zip(levels, rate) if l2 >= lv)), None)
        ntc = [c for l, n, c in rows if n == 0 and c != "invalid"]; spec = sum(c == "negative" for c in ntc) / len(ntc) * 100
        log("OT4 limit of detection", dict(s, reps=reps), f"LOD95 {lod} copies", "≤ 10 copies", lod is not None and lod <= 10)
        log("OT4 specificity", dict(s, reps=reps), f"{spec:.1f}% (n = {len(ntc)})", "≥ 95%", spec >= 95)
        fig, ax = plt.subplots(figsize=(7, 3)); ax.semilogx([max(l, .3) for l in levels], [r * 100 for r in rate], "o-"); ax.axhline(95, ls="--", color="k")
        ax.set_xlabel("copies (0 shown at 0.3)"); ax.set_ylabel("% positive"); st.pyplot(fig)
    if c3.button("Ambient light"):
        with st.spinner("Running ambient-light test…"):
            rng = np.random.default_rng(int(oseed))
            rxn = [(ps.cart_wells(ps.pcr_endpoint(100, rng)[0], rng), "positive") for _ in range(n_each)] + [(ps.cart_wells(ps.pcr_endpoint(0, rng)[0], rng), "negative") for _ in range(n_each)]
            labels, fps, inv = [], [], []
            for seal, tau in ps.SEAL.items():
                for light, lx in ps.LIGHT.items():
                    calls = []
                    for wl, _ in rxn:
                        im, _, mx = ps.render(wl, phone, 0.8, rng, lux=lx, tau=tau); calls.append(ps.call(ps.analyse(im, mx, ps.CART), 0, 1, 2, T))
                    fps.append(np.mean([c == "positive" for c, (_, t) in zip(calls, rxn) if t == "negative"]) * 100)
                    inv.append(np.mean([c == "invalid" for c in calls]) * 100); labels.append(f"{seal.split(' ')[0]}\n{lx} lx")
        log("OT7 ambient false positives", dict(s, n=n_each), f"max {max(fps):.1f}%", "≤ 5%", max(fps) <= 5)
        fig, ax = plt.subplots(figsize=(9, 3)); x = np.arange(len(labels))
        ax.bar(x - .2, fps, .4, label="false positive %", color="#e34948"); ax.bar(x + .2, inv, .4, label="invalid %", color="#8a99ad")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.legend(fontsize=8); st.pyplot(fig)

# ---------------- log ----------------
with tab_l:
    st.write("Every test you run in this session is listed here. Download the log before closing the page.")
    if st.session_state.log:
        st.dataframe(st.session_state.log, width="stretch")
        buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(st.session_state.log[0])); w.writeheader(); w.writerows(st.session_state.log)
        st.download_button("Download test log (CSV)", buf.getvalue(), file_name=f"MaterNix_test_log_{datetime.date.today()}.csv", mime="text/csv")
    else:
        st.info("No tests run yet.")
