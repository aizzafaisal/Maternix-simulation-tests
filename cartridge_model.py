"""
MaterNix Aim 3 - cartridge model (verification by analysis)
Geometry from cartridge_prototype_FINAL.SLDPRT: chambers 7 x 17 mm, 5 mm deep, 10 mm pitch,
sample pocket 6 x 16 mm, microchannel along the chamber tips, wax valve to the PCR tube.
Every input marked ASSUMED is a literature-typical value; change it in the test settings.
"""
import numpy as np

MU0 = 4e-7 * np.pi
DEPTH_MM, PITCH_MM = 5.0, 10.0

# ---------------- A. chamber volumes ----------------
def chamber_volumes(w=7.0, l=17.0, depth=5.0, tip=3.5, ws=6.0, ls=16.0):
    v_ch = w * (l - tip) * depth + 0.5 * w * tip * depth
    v_s = ws * (ls - ws) * depth + np.pi * (ws / 2) ** 2 * depth
    return dict(sample=v_s, lysis=v_ch, wash1=v_ch, wash2=v_ch, elution=v_ch)

# ---------------- B. microchannel flow ----------------
def channel_resistance(w_mm, h_mm, L_mm, visc):
    """Hydraulic resistance of a rectangular channel (Pa.s/m^3), h <= w."""
    w, h, L = w_mm * 1e-3, h_mm * 1e-3, L_mm * 1e-3
    if h > w: w, h = h, w
    return 12 * visc * L / (w * h ** 3 * (1 - 0.63 * h / w))

def transfer_time(vol_uL, dP_Pa, w_mm=0.5, h_mm=0.5, L_mm=10.0, visc=1e-3):
    Q = dP_Pa / channel_resistance(w_mm, h_mm, L_mm, visc)      # m^3/s
    return vol_uL * 1e-9 / Q, Q * 1e9                            # s, uL/s

# ---------------- C. magnetic beads ----------------
def Bz(z, Br=1.45, a=10e-3, b=10e-3, c=5e-3):
    f = lambda zz: np.arctan(a * b / (2 * zz * np.sqrt(4 * zz ** 2 + a ** 2 + b ** 2)))
    return Br / np.pi * (f(z) - f(z + c))

def bead_capture(floor_mm=3.0, gap_mm=0.5, bead_um=1.0, visc=1e-3, Ms=3.0e4, Br=1.45, depth_mm=DEPTH_MM):
    z = np.linspace(0.3e-3, 15e-3, 600)
    B = Bz(z, Br); dB = np.abs(np.gradient(B, z))
    r = bead_um * 0.5e-6; V = 4 / 3 * np.pi * r ** 3
    M = np.minimum(B / MU0, Ms)
    v = V * M * dB / (6 * np.pi * visc * r)
    s = (floor_mm + gap_mm) * 1e-3
    zz = np.linspace(s, s + depth_mm * 1e-3, 300)
    cap = float(np.trapezoid(1 / np.interp(zz, z, v), zz))
    v_floor = float(np.interp(s, z, v))
    trans = PITCH_MM * 1e-3 / min(0.5 * v_floor, 2e-3)
    return dict(capture_s=cap, v_floor_um_s=v_floor * 1e6, transfer_s=trans, B_mT=float(np.interp(s, z, B)) * 1e3)

# ---------------- D. wax valve ----------------
def wax_open(T_hot=95.0, T0=25.0, wall_mm=1.0, plug_mm=1.0, Tm=60.0, k_wall=0.13, Lf=2.0e5, dt=2e-3, tmax=600):
    k_w, rho_w, cp_w = 0.25, 900.0, 2100.0
    rho_p, cp_p = 1240.0, 1800.0
    L = (wall_mm + plug_mm) * 1e-3; nx = 60; x = np.linspace(0, L, nx); dx = x[1] - x[0]
    isw = x > wall_mm * 1e-3
    k = np.where(isw, k_w, k_wall); rc = np.where(isw, rho_w * cp_w, rho_p * cp_p)
    H = rc * (T0 - Tm); kf = 0.5 * (k[1:] + k[:-1])
    def T_of(H):
        T = np.where(H < 0, Tm + H / rc, Tm)
        T = np.where(isw & (H > rho_w * Lf), Tm + (H - rho_w * Lf) / rc, T)
        return np.where(~isw, Tm + H / rc, T)
    t = 0.0; hist = []; n = 0
    while t < tmax:
        T = T_of(H); T[0] = T_hot
        q = kf * (T[1:] - T[:-1]) / dx
        dH = np.zeros_like(H); dH[1:-1] = (q[1:] - q[:-1]) / dx; dH[-1] = -q[-1] / dx
        H[1:] += dH[1:] * dt; H[0] = rc[0] * (T_hot - Tm); t += dt; n += 1
        if n % 250 == 0: hist.append((t, float(T_of(H)[isw].mean())))
        if np.all(H[isw] >= rho_w * Lf): return t, hist
    return float("nan"), hist

# ---------------- E. workflow ----------------
def workflow(floor_mm=3.0, bead_um=1.0, gap_mm=0.5, compressed=False, valve_s=74.0, pcr_min=30.0, read_min=1.0):
    aq = bead_capture(floor_mm, gap_mm, bead_um, 1.0e-3); et = bead_capture(floor_mm, gap_mm, bead_um, 2.2e-3)
    if compressed:
        chem = [("Insert swab, close lid", 0.5), ("Swab elution + lysis", 5.0), ("Binding", 2.0), ("Wash 1", 0.5), ("Wash 2", 0.5), ("Dry", 0.5), ("Elution", 1.5)]
        valve = 0.0   # opens inside the first PCR heating step
    else:
        chem = [("Insert swab, close lid", 0.5), ("Swab elution", 2.0), ("Lysis", 8.0), ("Binding", 3.0), ("Wash 1", 1.0), ("Wash 2", 1.0), ("Dry", 1.0), ("Elution", 2.0)]
        valve = valve_s / 60
    mag = (aq["capture_s"] + aq["transfer_s"] + aq["capture_s"] + et["transfer_s"] + et["capture_s"] + aq["transfer_s"] + aq["capture_s"]) / 60
    pre = sum(d for _, d in chem) + mag + valve
    return dict(steps=chem + [("Magnetic capture + transfers", mag), ("Wax valve", valve), ("PCR", pcr_min), ("Image + result", read_min)],
                magnetic_min=mag, prep_min=pre, total_min=pre + pcr_min + read_min)
