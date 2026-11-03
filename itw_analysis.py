#!/usr/bin/env python3
"""
itw_analysis.py - Ankle rocker detection, quasi-stiffness and Alvarez ITW severity
classification for children with idiopathic toe walking (ITW).

README
------
What it does (per limb, per condition, for every subject found in the data folder):
  1. Detects the ankle rockers from each limb's own data (no normal-gait timing assumed),
     especially the SECOND ankle rocker (tibial advancement, end of 1st rocker -> peak
     dorsiflexion).
  2. Computes ankle quasi-stiffness (slope of the moment-angle curve, dM/dtheta).
  3. Classifies every limb into ITW Type 1 / 2 / 3 using
        Alvarez C, De Vera M, Beauchamp R, Ward V, Black A.
        Gait & Posture 26 (2007) 428-435.
  4. Opens an interactive matplotlib viewer and exports CSV tables (+ a PDF of all figures).

Input files (tab-delimited, one ensemble-averaged gait cycle of 101 samples, IC -> IC):
    S01_Precast.txt, S01_Postcast.txt, S02_Precast.txt, ...   (any number of subjects)
Columns: Left angle, Left moment, Left power, Right angle, Right moment, Right power.
Conventions (see Config): angle in deg with dorsiflexion POSITIVE, plantarflexor moment
positive (Nm/kg), power in W/kg.  Flip them in Config if your system differs.

Outputs (in --out, default ./results next to this script):
    ankle_quasi_stiffness.csv, itw_classification.csv, limb_summary.csv, run_log.txt
    all_figures.pdf (with --no-gui, or press 'E' in the viewer)

Viewer keys:  Left/Right = previous/next record (or page in paginated group views)
              Up/Down = plot type      PageUp/PageDown = previous/next subject
              R = random record        S = save current figure      E = export all to PDF

Install:   pip install numpy pandas scipy matplotlib
Usage:     python itw_analysis.py --data-dir ./data --out ./results [--body-mass masses.csv]
                                  [--no-gui] [--selftest]
Run button in VS Code (no arguments) also works: the data folder defaults to the folder
that contains this script (falls back to ./data).

Ready-to-paste .vscode/launch.json
----------------------------------
{
  "version": "0.2.0",
  "configurations": [
    {"name": "ITW viewer (GUI)", "type": "debugpy", "request": "launch",
     "program": "${file}", "console": "integratedTerminal",
     "args": ["--data-dir", "${workspaceFolder}/data", "--out", "${workspaceFolder}/results"]},
    {"name": "ITW selftest", "type": "debugpy", "request": "launch",
     "program": "${file}", "console": "integratedTerminal",
     "args": ["--selftest", "--data-dir", "${workspaceFolder}/data"]},
    {"name": "ITW export only (no GUI)", "type": "debugpy", "request": "launch",
     "program": "${file}", "console": "integratedTerminal",
     "args": ["--no-gui", "--data-dir", "${workspaceFolder}/data",
              "--out", "${workspaceFolder}/results"]}
  ]
}
(Older VS Code Python extensions use "type": "python" instead of "debugpy".)

Notes on interpretation of the paper (documented because the PDF is ambiguous)
------------------------------------------------------------------------------
* First ankle rocker is PRESENT when the IC angle has LESS than 5 deg of plantarflexion
  (IC angle > -5 deg with dorsiflexion positive) AND the ankle goes further into
  plantarflexion during loading response (downgoing excursion).  Alvarez's Table 3
  supports this reading (Type 1 IC ~ -0.7 deg, Type 3 IC ~ -21.7 deg with no first
  rocker).  Set Config.first_rocker_requires_small_ic_plantarflexion = False for the
  literal reading ("more than 5 deg plantarflexed").
* Combinations with AM1 > AM2 that are not the Type 3 pattern are "Unclassified" (the
  paper's 7 unclassified feet were Yes/Yes/Yes).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import random
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

NAN = float("nan")


# =============================================================================
# CONFIG  (every threshold lives here)
# =============================================================================
@dataclass
class Config:
    # --- data conventions (flip if your system differs) -----------------------
    dorsiflexion_positive: bool = True      # False -> angle sign is inverted on load
    moment_sign: float = 1.0                # multiply moments (use -1 to flip)
    n_samples: int = 101                    # samples per gait cycle
    file_regex: str = r"^(S\d+)_(Pre|Post)cast\.txt$"
    # --- end-of-cycle artefact handling ---------------------------------------
    exclude_last_frames: int = 1            # frames dropped from detection/regression
    last_frame_jump_moment: float = 0.01    # Nm/kg jump that flags LAST_FRAME_ARTEFACT
    last_frame_jump_power: float = 0.02     # W/kg jump that flags LAST_FRAME_ARTEFACT
    # --- Alvarez criteria -------------------------------------------------------
    first_rocker_end_pct: float = 12.0      # first-rocker window 0-12 % GC
    ic_plantarflexion_threshold_deg: float = 5.0
    first_rocker_requires_small_ic_plantarflexion: bool = True   # see module docstring
    downgoing_tol_deg: float = 0.5
    early_third_rocker_pct: float = 30.0
    am1_end_pct: float = 35.0               # AM1 window: 0 - 35 % GC
    am2_start_pct: float = 35.0             # AM2 window: 35 % GC - toe-off
    am2_search_end_pct: float = 75.0        # upper bound when searching for the AM2 peak
    toe_off_moment_frac: float = 0.05       # toe-off: moment < 5 % of AM2 after AM2 peak
    toe_off_fallback_pct: float = 60.0
    min_valid_moment: float = 0.1           # Nm/kg; below this kinetics are invalid
    # --- push-off power -----------------------------------------------------------
    pushoff_start_pct: float = 35.0
    # --- event refinement ----------------------------------------------------------
    spline_points: int = 1000
    plateau_within_deg: float = 1.0
    plateau_min_pct: float = 10.0
    # --- stiffness quality gates -----------------------------------------------------
    min_rom_deg: float = 3.0
    min_r2: float = 0.5
    min_points: int = 5
    subwindow_lo: float = 0.2
    subwindow_hi: float = 0.8
    # --- borderline margins ------------------------------------------------------------
    borderline_ic_deg: float = 1.0
    borderline_third_pct: float = 2.0
    borderline_ratio_lo: float = 0.9
    borderline_ratio_hi: float = 1.1
    # --- viewer ----------------------------------------------------------------------------
    rows_per_page: int = 30


CONFIG = Config()

LIMBS = ("L", "R")
LIMB_COLS = {"L": (0, 1, 2), "R": (3, 4, 5)}
LIMB_COL = {"L": "tab:blue", "R": "tab:red"}
COND_COL = {"Pre": "tab:orange", "Post": "tab:blue"}
SEV_NOKIN = "Unclassifiable - no valid kinetics"
TYPE_ORDER = ["Type 1", "Type 2", "Type 3", "Unclassified", SEV_NOKIN]
TYPE_COL = {"Type 1": "tab:green", "Type 2": "tab:orange", "Type 3": "tab:red",
            "Unclassified": "tab:purple", SEV_NOKIN: "tab:gray", "Error": "k"}


# =============================================================================
# SMALL HELPERS
# =============================================================================
def natural_key(subject: str) -> Tuple[int, str]:
    """Sort key so S2 < S10 < S100."""
    m = re.search(r"\d+", subject)
    return (int(m.group()) if m else 0, subject)


def yn(flag: Optional[bool]) -> str:
    return "NA" if flag is None else ("Yes" if flag else "No")


def fmt(x, f: str = "{:.3f}", na: str = "n/a") -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return na
        return f.format(x)
    except Exception:
        return str(x)


def short_sev(sev: str) -> str:
    return {"Type 1": "T1", "Type 2": "T2", "Type 3": "T3", "Unclassified": "U",
            SEV_NOKIN: "NK"}.get(sev, "?")


def sev_key(sev: str) -> str:
    return sev if sev in ("Type 1", "Type 2", "Type 3") else "Other"


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _ols(x, y) -> dict:
    """Ordinary least squares of y on x. Returns slope, intercept, r2, n, xmin, xmax."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = dict(slope=NAN, intercept=NAN, r2=NAN, n=int(x.size), xmin=NAN, xmax=NAN)
    if x.size < 2:
        return out
    out["xmin"], out["xmax"] = float(x.min()), float(x.max())
    if np.ptp(x) < 1e-9:
        return out
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.update(slope=float(slope), intercept=float(intercept),
               r2=(1 - ss_res / ss_tot) if ss_tot > 1e-12 else NAN)
    return out


def _refine_peak(x: np.ndarray, y: np.ndarray, npts: int) -> Tuple[float, float]:
    """Cubic-spline refined (time, value) of the maximum of y(x)."""
    if len(x) < 3:
        i = int(np.argmax(y))
        return float(x[i]), float(y[i])
    try:
        g = np.linspace(x[0], x[-1], npts)
        v = CubicSpline(x, y)(g)
        i = int(np.argmax(v))
        return float(g[i]), float(v[i])
    except Exception:
        i = int(np.argmax(y))
        return float(x[i]), float(y[i])


# =============================================================================
# LOAD
# =============================================================================
def load_file(path: Path, cfg: Config) -> np.ndarray:
    """Parse one tab-delimited file; keep only lines whose first field is an integer.
    Returns an (n_samples, 6) array or raises ValueError."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 7 or not re.fullmatch(r"\d+", parts[0].strip()):
                continue
            try:
                rows.append([float(v) for v in parts[1:7]])
            except ValueError:
                continue
    if len(rows) != cfg.n_samples:
        raise ValueError(f"expected {cfg.n_samples} data rows, found {len(rows)}")
    return np.asarray(rows, float)


def discover_files(data_dir: Path, cfg: Config):
    """Find every file matching cfg.file_regex. Returns ({(subject, cond): path}, ignored names)."""
    rx = re.compile(cfg.file_regex, re.I)
    found: Dict[Tuple[str, str], Path] = {}
    ignored: List[str] = []
    for p in sorted(Path(data_dir).glob("*.txt")):
        m = rx.match(p.name)
        if m:
            found[(m.group(1).upper(), m.group(2).capitalize())] = p
        else:
            ignored.append(p.name)
    return found, ignored


# =============================================================================
# EVENTS + CLASSIFICATION + STIFFNESS + PUSH-OFF  (one limb)
# =============================================================================
def classify_alvarez(r1: bool, r3: bool, am: Optional[bool], kin_valid: bool) -> str:
    """Alvarez 2007 Table 1. Type 1: R1 yes, R3 no, AM1>AM2 no. Type 2: AM1>AM2 no (any
    other R1/R3 combination). Type 3: R1 no, R3 yes, AM1>AM2 yes. Else Unclassified."""
    if not kin_valid:
        return SEV_NOKIN
    if not am:
        return "Type 1" if (r1 and not r3) else "Type 2"
    if (not r1) and r3:
        return "Type 3"
    return "Unclassified"


def blank_result(flags: List[str]) -> dict:
    return {"severity": "Error", "kinetics_valid": False, "flags": list(flags), "_ok": False,
            "_idx": None, "_fits": {}, "_crit": (None, None, None)}


def analyze_limb(angle: np.ndarray, moment: np.ndarray, power: np.ndarray, cfg: Config,
                 mass_kg: Optional[float] = None) -> dict:
    """ Analyze one ensemble-averaged ankle gait cycle.

    This is the main biomechanical analysis function in the script.

    INPUTS
    ------
    angle
        Ankle angle for one limb across the gait cycle.

        By default, the script assumes:
            dorsiflexion   = positive
            plantarflexion = negative

        The sign convention can be changed in Config before this function
        is called.

    moment
        Ankle joint moment across the gait cycle in Nm/kg.

        The script assumes the moment has already been transformed to the
        desired sign convention when the dataset was created.

    power
        Ankle joint power across the gait cycle in W/kg.

    cfg
        Config object containing all thresholds and analysis settings.

    mass_kg
        Optional body mass. If supplied, normalized quasi-stiffness
        (Nm/kg/deg) can also be converted to an approximate absolute
        quasi-stiffness in Nm/deg.

    RETURNS
    -------
    dict
        Dictionary containing:
            * detected ankle-rocker events
            * Alvarez criteria
            * Alvarez severity classification
            * AM1 and AM2
            * second-rocker quasi-stiffness
            * first-rocker and powered-PF slopes
            * mechanical work estimates
            * push-off power
            * quality-control flags
            * internal indices used by plotting functions

    NOTES
    -----
    The function is designed so unusual gait signals generally produce
    warning flags and/or NaN results rather than causing the entire
    analysis pipeline to fail.
    """
     # -------------------------------------------------------------------------
    # INITIALIZE QUALITY-CONTROL FLAGS
    # -------------------------------------------------------------------------
    # Rather than immediately rejecting unusual data, the analysis records
    # QC flags. These flags are exported later and allow the investigator to
    # decide whether individual estimates should be included.
    flags: List[str] = []
     # Total number of samples in the original gait cycle.
    # Normally this will be 101 samples representing 0-100% gait cycle.
    n_full = len(angle)
     # Some motion-analysis exports contain an artificial discontinuity in the
    # final frame because the last sample represents the next initial contact.
    #
    # exclude_last_frames specifies how many terminal samples should be
    # excluded from event detection and regression.
    n = n_full - cfg.exclude_last_frames
     # Construct a percentage-of-gait-cycle axis for the complete signal.
    #
    # Example for a 101-sample gait cycle:
    #     sample 0   -> 0%
    #     sample 50  -> 50%
    #     sample 100 -> 100%
    
    pct_full = np.linspace(0, 100, n_full)
    # -------------------------------------------------------------------------
    # CHECK FOR A POSSIBLE FINAL-FRAME ARTIFACT
    # -------------------------------------------------------------------------
    # Compare the last two samples of moment and power.
    #
    # If either changes by more than its configured threshold, mark the
    # record. The record is NOT automatically rejected.
    if n_full >= 3 and (abs(moment[-1] - moment[-2]) > cfg.last_frame_jump_moment
                        or abs(power[-1] - power[-2]) > cfg.last_frame_jump_power):
        flags.append("LAST_FRAME_ARTEFACT")
    # Create shortened working arrays that exclude the configured terminal
    # frames. The original arrays remain available to the Record object.
    a, m, p, pct = angle[:n], moment[:n], power[:n], pct_full[:n]
    
     # -------------------------------------------------------------------------
    # DETERMINE WHETHER ANKLE KINETICS ARE USABLE
    # -------------------------------------------------------------------------
    # If the largest absolute ankle moment never reaches min_valid_moment,
    # assume that the kinetic signal is absent or invalid.
    #
    # This is particularly useful for trials where kinematic data exist but
    # force/moment data are missing or essentially zero.
    
    kin_valid = bool(np.nanmax(np.abs(m)) >= cfg.min_valid_moment)
    if not kin_valid:
        flags.append("NO_VALID_KINETICS")

    # =========================================================================
    # AM1, AM2, AND TOE-OFF
    # =========================================================================
    # Alvarez criterion 3 depends on whether AM1 is greater than AM2.
    #
    # AM1 = early ankle-moment maximum
    # AM2 = later ankle-moment maximum
    #
    # Rather than assuming a particular sample number, search windows are
    # defined as percentages of the gait cycle in Config.

    # Samples belonging to the AM1 search window.
    i1 = np.where(pct < cfg.am1_end_pct)[0]
    # Samples belonging to the AM2 search window.
    i2 = np.where((pct >= cfg.am2_start_pct) & (pct <= cfg.am2_search_end_pct))[0]
     # np.argmax() reports the position of the largest value WITHIN i1/i2.
    #
    # Indexing i1/i2 with that value converts it back to the corresponding
    # index in the full working gait-cycle array.
    am1_idx = int(i1[np.argmax(m[i1])])
    am2_idx = int(i2[np.argmax(m[i2])])
    
    # Actual ankle-moment magnitudes at those samples.
    am1, am2 = float(m[am1_idx]), float(m[am2_idx])
     # -------------------------------------------------------------------------
    # DEFAULT TOE-OFF LOCATION
    # -------------------------------------------------------------------------
    # Build a fallback toe-off index at the configured gait-cycle percentage.
    #
    # This is used if toe-off cannot be identified from the moment signal.
    fb_idx = int(np.argmin(np.abs(pct - cfg.toe_off_fallback_pct)))
    toe_idx = None
    # -------------------------------------------------------------------------
    # ESTIMATE TOE-OFF FROM THE ANKLE MOMENT
    # -------------------------------------------------------------------------
    # If kinetics are valid and AM2 is positive, search AFTER the AM2 peak for
    # the first sample at which ankle moment falls below a small fraction of
    # AM2.
    #
    # Default:
    #     threshold = 5% of AM2
    #
    # This provides a signal-dependent estimate of the end of stance.
    if kin_valid and am2 > 0:
        after = np.where((np.arange(n) > am2_idx) & (m < cfg.toe_off_moment_frac * am2))[0]
        if after.size:
            toe_idx = int(after[0])
    if toe_idx is None:
        toe_idx = fb_idx
        flags.append("TOE_OFF_FALLBACK")
      # Ensure that toe-off cannot occur so early that subsequent operations
    # have essentially no usable data.
    toe_idx = max(toe_idx, 3)
    
     # =========================================================================
        # FIRST ANKLE ROCKER
        # Alvarez criterion 1
        # =========================================================================
        # Determine whether loading response contains an identifiable first
        # ankle rocker.
        #
        # Under the default sign convention:
        #
        #      dorsiflexion   = positive
        #      plantarflexion = negative
        #
        # The first rocker requires:
        #
        #   1. An acceptable ankle angle at initial contact.
        #
        #   2. A subsequent plantarflexion excursion during the early portion
        #      of the gait cycle.
    
        # Ankle angle at initial contact.

    # ---- first ankle rocker (Alvarez criterion 1) ----------------------------------------
    ic = float(a[0])
    # Identify all samples inside the first-rocker search window.
        # Default = 0-12% gait cycle.
    fr_win = np.where(pct <= cfg.first_rocker_end_pct)[0]
    fr_idx = int(fr_win[np.argmin(a[fr_win])])
    fr_min = float(a[fr_idx])
     # -------------------------------------------------------------------------
        # DID THE ANKLE ACTUALLY PLANTARFLEX AFTER INITIAL CONTACT?
        # -------------------------------------------------------------------------
        # Requiring a minimum change prevents very small fluctuations or numerical
        # noise from being interpreted as a real first-rocker excursion.
    downgoing = fr_min < ic - cfg.downgoing_tol_deg
    thr = cfg.ic_plantarflexion_threshold_deg
    ic_ok = (ic > -thr) if cfg.first_rocker_requires_small_ic_plantarflexion else (ic < -thr)
    r1 = bool(ic_ok and downgoing)
    if abs(ic - (-thr)) < cfg.borderline_ic_deg:
        flags.append("BORDERLINE_IC_ANGLE")

    # ---- second rocker: end of 1st rocker (or IC) -> peak dorsiflexion ------------------
    sr_start = fr_idx if r1 else 0
    seg = a[sr_start:toe_idx + 1]
    peak_idx = sr_start + int(np.argmax(seg))
    if toe_idx - sr_start < 3 or peak_idx == sr_start:
        flags.append("SECOND_ROCKER_WEAK")
    peak_time, peak_angle = _refine_peak(pct[sr_start:toe_idx + 1], seg, cfg.spline_points)

    # ---- third rocker (Alvarez criterion 2) ---------------------------------------------------
    r3 = bool(peak_time <= cfg.early_third_rocker_pct)
    if abs(peak_time - cfg.early_third_rocker_pct) < cfg.borderline_third_pct:
        flags.append("BORDERLINE_THIRD_ROCKER")
    thr_pl = a[peak_idx] - cfg.plateau_within_deg
    lft = peak_idx
    while lft - 1 >= sr_start and a[lft - 1] >= thr_pl:
        lft -= 1
    rgt = peak_idx
    while rgt + 1 <= toe_idx and a[rgt + 1] >= thr_pl:
        rgt += 1
    plateau_run = float(pct[rgt] - pct[lft])
    if plateau_run > cfg.plateau_min_pct:
        flags.append("PLATEAU")

    am = bool(am1 > am2) if kin_valid else None
    ratio = am1 / am2 if (kin_valid and am2 > 1e-9) else NAN
    if np.isfinite(ratio) and cfg.borderline_ratio_lo <= ratio <= cfg.borderline_ratio_hi:
        flags.append("BORDERLINE_AM_RATIO")
    severity = classify_alvarez(r1, r3, am, kin_valid)

    res: dict = {
        "severity": severity,
        "first_rocker_present": yn(r1), "early_third_rocker": yn(r3), "am1_gt_am2": yn(am),
        "criteria_combo": f"1st rocker={yn(r1)}, 3rd rocker={yn(r3)}, AM1>AM2={yn(am)}",
        "kinetics_valid": kin_valid,
        "ic_angle_deg": ic, "min_angle_loading_response_deg": fr_min,
        "first_rocker_excursion_deg": ic - fr_min,
        "first_rocker_end_pctGC": float(pct[fr_idx]) if r1 else NAN,
        "second_rocker_start_pctGC": float(pct[sr_start]),
        "second_rocker_end_pctGC": peak_time,
        "third_rocker_onset_pctGC": peak_time,
        "peak_dorsiflexion_angle_deg": peak_angle,
        "second_rocker_plateau_run_pctGC": plateau_run,
        "toe_off_pctGC": float(pct[toe_idx]),
        "am1_Nm_per_kg": am1 if kin_valid else NAN,
        "am1_time_pctGC": float(pct[am1_idx]) if kin_valid else NAN,
        "am2_Nm_per_kg": am2 if kin_valid else NAN,
        "am2_time_pctGC": float(pct[am2_idx]) if kin_valid else NAN,
        "am1_am2_ratio": ratio,
        "_ok": True, "_crit": (r1, r3, am),
        "_idx": dict(fr=fr_idx, sr_start=sr_start, peak=peak_idx, toe=toe_idx,
                     am1=am1_idx, am2=am2_idx),
        "_fits": {},
    }

    # ---- quasi-stiffness ------------------------------------------------------------------------
    stiff_keys = ["second_rocker_k", "second_rocker_R2", "second_rocker_n_points",
                  "second_rocker_ROM_deg", "second_rocker_moment_change_Nm_per_kg",
                  "second_rocker_secant_k_Nm_per_kg_per_deg",
                  "second_rocker_subwindow_k_Nm_per_kg_per_deg", "second_rocker_subwindow_R2",
                  "second_rocker_subwindow_n", "first_rocker_k_Nm_per_kg_per_deg",
                  "first_rocker_R2", "first_rocker_n", "powered_pf_k_Nm_per_kg_per_deg",
                  "powered_pf_R2", "powered_pf_n", "work_absorbed_second_rocker_J_per_kg",
                  "work_released_powered_pf_J_per_kg", "work_release_to_absorb_ratio",
                  "second_rocker_k_Nm_per_kg_per_deg", "second_rocker_k_Nm_per_kg_per_rad",
                  "second_rocker_k_Nm_per_deg"]
    for k in stiff_keys:
        res[k] = NAN
    res.pop("second_rocker_k")
    if kin_valid:
        w = np.arange(sr_start, peak_idx + 1)
        fit = _ols(a[w], m[w])
        rom = float(a[peak_idx] - a[sr_start])
        res.update({"second_rocker_k_Nm_per_kg_per_deg": fit["slope"],
                    "second_rocker_k_Nm_per_kg_per_rad": fit["slope"] * 180 / math.pi,
                    "second_rocker_k_Nm_per_deg": fit["slope"] * mass_kg if mass_kg else NAN,
                    "second_rocker_R2": fit["r2"], "second_rocker_n_points": fit["n"],
                    "second_rocker_ROM_deg": rom,
                    "second_rocker_moment_change_Nm_per_kg": float(m[peak_idx] - m[sr_start])})
        res["_fits"]["second"] = fit
        if rom < cfg.min_rom_deg:
            flags.append("LOW_ROM")
        if not (fit["r2"] >= cfg.min_r2):
            flags.append("LOW_R2")
        if fit["n"] < cfg.min_points:
            flags.append("FEW_POINTS")
        if rom > 1e-6:
            res["second_rocker_secant_k_Nm_per_kg_per_deg"] = float(
                (m[peak_idx] - m[sr_start]) / (a[peak_idx] - a[sr_start]))
            lo = a[w].min() + cfg.subwindow_lo * rom
            hi = a[w].min() + cfg.subwindow_hi * rom
            sel = w[(a[w] >= lo) & (a[w] <= hi)]
            if sel.size >= 3:
                sf = _ols(a[sel], m[sel])
                res.update({"second_rocker_subwindow_k_Nm_per_kg_per_deg": sf["slope"],
                            "second_rocker_subwindow_R2": sf["r2"],
                            "second_rocker_subwindow_n": sf["n"]})
                res["_fits"]["sub"] = sf
            else:
                flags.append("SUBWINDOW_TOO_FEW_POINTS")
        if r1:
            w1 = np.arange(0, fr_idx + 1)
            if w1.size >= 3:
                f1 = _ols(a[w1], m[w1])
                res.update({"first_rocker_k_Nm_per_kg_per_deg": f1["slope"],
                            "first_rocker_R2": f1["r2"], "first_rocker_n": f1["n"]})
                res["_fits"]["first"] = f1
        wp = np.arange(peak_idx, toe_idx + 1)
        if wp.size >= 3:
            fp = _ols(a[wp], m[wp])
            res.update({"powered_pf_k_Nm_per_kg_per_deg": fp["slope"],
                        "powered_pf_R2": fp["r2"], "powered_pf_n": fp["n"]})
            res["_fits"]["powered"] = fp
        # work: integral M d(theta), theta in rad -> J/kg
        w_abs = _trapz(m[w], np.deg2rad(a[w]))
        w_rel = -_trapz(m[wp], np.deg2rad(a[wp])) if wp.size >= 2 else NAN
        res["work_absorbed_second_rocker_J_per_kg"] = w_abs
        res["work_released_powered_pf_J_per_kg"] = w_rel
        res["work_release_to_absorb_ratio"] = (w_rel / w_abs) if abs(w_abs) > 1e-6 else NAN

    # ---- push-off power (A2 burst) ---------------------------------------------------------------
    res.update({"peak_pushoff_power_W_per_kg": NAN, "pushoff_power_time_pctGC": NAN,
                "pushoff_positive_work_W_per_kg_x_pctGC": NAN})
    if kin_valid:
        start_pct = max(peak_time, cfg.pushoff_start_pct)
        win = np.where((pct >= start_pct) & (np.arange(n) <= toe_idx))[0]
        if win.size >= 1 and float(np.max(p[win])) > 0:
            pk_t, _ = _refine_peak(pct[win], p[win], cfg.spline_points)
            res["peak_pushoff_power_W_per_kg"] = float(np.max(p[win]))
            res["pushoff_power_time_pctGC"] = pk_t
            res["pushoff_positive_work_W_per_kg_x_pctGC"] = _trapz(np.clip(p[win], 0, None),
                                                                    pct[win])
        else:
            flags.append("NO_PUSHOFF_PEAK")
    res["flags"] = flags
    return res


# =============================================================================
# DATASET
# =============================================================================
@dataclass
class Record:
    subject: str
    condition: str
    limb: str
    angle: np.ndarray
    moment: np.ndarray
    power: np.ndarray
    pct: np.ndarray
    n_use: int
    res: dict

    @property
    def label(self) -> str:
        return f"{self.subject} {self.condition} {self.limb}"


class Dataset:
    def __init__(self) -> None:
        self.records: List[Record] = []
        self._idx: Dict[Tuple[str, str, str], Record] = {}
        self.subjects: List[str] = []
        self.missing: List[str] = []
        self.ignored: List[str] = []
        self.failed: List[Tuple[str, str]] = []
        self.found_files: List[str] = []

    def add(self, r: Record) -> None:
        self.records.append(r)
        self._idx[(r.subject, r.condition, r.limb)] = r
        if r.subject not in self.subjects:
            self.subjects.append(r.subject)

    def get(self, subject: str, cond: str, limb: str) -> Optional[Record]:
        return self._idx.get((subject, cond, limb))

    def pairs(self, limb: str = "Both") -> List[Tuple[Record, Record]]:
        out = []
        for s in self.subjects:
            for lb in LIMBS:
                if limb != "Both" and lb != limb:
                    continue
                a, b = self.get(s, "Pre", lb), self.get(s, "Post", lb)
                if a is not None and b is not None:
                    out.append((a, b))
        return out


def read_body_mass(path: Optional[Path]) -> Dict[str, float]:
    if not path:
        return {}
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return {str(s).strip().upper(): float(v) for s, v in zip(df["subject"], df["mass_kg"])}


def build_dataset(data_dir: Path, cfg: Config, mass: Optional[Dict[str, float]] = None) -> Dataset:
    mass = mass or {}
    ds = Dataset()
    found, ds.ignored = discover_files(data_dir, cfg)
    ds.found_files = [p.name for p in found.values()]
    subjects = sorted({s for s, _ in found}, key=natural_key)
    for s in subjects:
        for c in ("Pre", "Post"):
            path = found.get((s, c))
            if path is None:
                ds.missing.append(f"{s}_{c}cast (missing -> subject UNPAIRED)")
                continue
            try:
                arr = load_file(path, cfg)
            except Exception as e:  # noqa: BLE001
                ds.failed.append((path.name, str(e)))
                continue
            for limb in LIMBS:
                c0, c1, c2 = LIMB_COLS[limb]
                ang = arr[:, c0] * (1.0 if cfg.dorsiflexion_positive else -1.0)
                mom = arr[:, c1] * cfg.moment_sign
                pw = arr[:, c2]
                try:
                    res = analyze_limb(ang, mom, pw, cfg, mass.get(s))
                except Exception as e:  # noqa: BLE001
                    res = blank_result([f"ANALYSIS_ERROR:{type(e).__name__}"])
                n_use = len(ang) - cfg.exclude_last_frames
                ds.add(Record(s, c, limb, ang, mom, pw, np.linspace(0, 100, len(ang)), n_use, res))
    ds.subjects.sort(key=natural_key)
    return ds


# =============================================================================
# TABLES + EXPORT
# =============================================================================
CLASS_COLS = ["subject", "condition", "limb", "severity", "first_rocker_present",
              "early_third_rocker", "am1_gt_am2", "criteria_combo", "ic_angle_deg",
              "min_angle_loading_response_deg", "first_rocker_excursion_deg",
              "first_rocker_end_pctGC", "second_rocker_start_pctGC", "second_rocker_end_pctGC",
              "third_rocker_onset_pctGC", "peak_dorsiflexion_angle_deg",
              "second_rocker_plateau_run_pctGC", "toe_off_pctGC", "am1_Nm_per_kg",
              "am1_time_pctGC", "am2_Nm_per_kg", "am2_time_pctGC", "am1_am2_ratio",
              "peak_pushoff_power_W_per_kg", "pushoff_power_time_pctGC",
              "pushoff_positive_work_W_per_kg_x_pctGC", "kinetics_valid", "flags"]
STIFF_COLS = ["subject", "condition", "limb", "kinetics_valid",
              "second_rocker_k_Nm_per_kg_per_deg", "second_rocker_k_Nm_per_kg_per_rad",
              "second_rocker_k_Nm_per_deg", "second_rocker_R2", "second_rocker_n_points",
              "second_rocker_ROM_deg", "second_rocker_moment_change_Nm_per_kg",
              "second_rocker_secant_k_Nm_per_kg_per_deg",
              "second_rocker_subwindow_k_Nm_per_kg_per_deg", "second_rocker_subwindow_R2",
              "second_rocker_subwindow_n", "first_rocker_k_Nm_per_kg_per_deg", "first_rocker_R2",
              "first_rocker_n", "powered_pf_k_Nm_per_kg_per_deg", "powered_pf_R2", "powered_pf_n",
              "work_absorbed_second_rocker_J_per_kg", "work_released_powered_pf_J_per_kg",
              "work_release_to_absorb_ratio", "flags"]
WIDE_NUM = ["second_rocker_k_Nm_per_kg_per_deg", "second_rocker_R2", "second_rocker_ROM_deg",
            "second_rocker_secant_k_Nm_per_kg_per_deg",
            "second_rocker_subwindow_k_Nm_per_kg_per_deg", "first_rocker_k_Nm_per_kg_per_deg",
            "powered_pf_k_Nm_per_kg_per_deg", "work_absorbed_second_rocker_J_per_kg",
            "work_released_powered_pf_J_per_kg", "am1_Nm_per_kg", "am2_Nm_per_kg",
            "am1_am2_ratio", "third_rocker_onset_pctGC", "ic_angle_deg",
            "peak_dorsiflexion_angle_deg", "peak_pushoff_power_W_per_kg",
            "pushoff_power_time_pctGC", "pushoff_positive_work_W_per_kg_x_pctGC"]
WIDE_CAT = ["severity", "first_rocker_present", "early_third_rocker", "am1_gt_am2", "flags"]


def long_table(ds: Dataset) -> pd.DataFrame:
    rows = []
    for r in ds.records:
        d = {"subject": r.subject, "condition": r.condition, "limb": r.limb}
        d.update({k: v for k, v in r.res.items() if not k.startswith("_") and k != "flags"})
        d["flags"] = ";".join(r.res["flags"])
        rows.append(d)
    return pd.DataFrame(rows)


def wide_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subj, limb), g in df.groupby(["subject", "limb"], sort=False):
        pre = g[g.condition == "Pre"]
        post = g[g.condition == "Post"]
        pr = pre.iloc[0] if len(pre) else None
        po = post.iloc[0] if len(post) else None
        d: dict = {"subject": subj, "limb": limb}
        for c in WIDE_CAT:
            d[f"{c}_pre"] = pr[c] if pr is not None and c in pr else ""
            d[f"{c}_post"] = po[c] if po is not None and c in po else ""
        d["severity_change"] = (f"{d['severity_pre']} -> {d['severity_post']}"
                                if pr is not None and po is not None else "")
        for c in WIDE_NUM:
            a = float(pr[c]) if pr is not None and c in pr and pd.notna(pr[c]) else NAN
            b = float(po[c]) if po is not None and c in po and pd.notna(po[c]) else NAN
            d[f"{c}_pre"], d[f"{c}_post"] = a, b
            d[f"{c}_delta"] = b - a
            d[f"{c}_pct_change"] = 100.0 * (b - a) / abs(a) if (np.isfinite(a) and abs(a) > 1e-9) else NAN
        d["pairing_flag"] = "" if (pr is not None and po is not None) else "UNPAIRED"
        rows.append(d)
    return pd.DataFrame(rows)


def write_outputs(ds: Dataset, cfg: Config, out_dir: Path, data_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = long_table(ds)
     # -------------------------------------------------------------------------
      # PRIMARY OUTCOMES
    # -------------------------------------------------------------------------
    # Export a small CSV containing only the two primary outcome measures.
    #
    # One row = one subject x condition x limb.
    #
    # Peak push-off power is already normalized to body mass (W/kg).
    # Second-rocker quasi-stiffness is also normalized to body mass
    # (Nm/kg/deg), so neither variable requires body mass here.
    # -------------------------------------------------------------------------
    outcome_cols = [
        "subject",
        "condition",
        "limb",
        "peak_pushoff_power_W_per_kg",
        "second_rocker_k_Nm_per_kg_per_deg",
    ]
     # -------------------------------------------------------------------------
    # EXISTING DETAILED OUTPUTS
    # -------------------------------------------------------------------------
    # Keep all of the original output files. These are useful for QC,
    # classification, secondary analyses, and checking how individual
    # outcome values were obtained.


    df.reindex(columns=outcome_cols).to_csv(
        out_dir / "primary_outcomes.csv",
        index=False,
        encoding="utf-8"
    )
    
    df.reindex(columns=STIFF_COLS).to_csv(out_dir / "ankle_quasi_stiffness.csv", index=False,
                                          encoding="utf-8")
    df.reindex(columns=CLASS_COLS).to_csv(out_dir / "itw_classification.csv", index=False,
                                          encoding="utf-8")
    wide_table(df).to_csv(out_dir / "limb_summary.csv", index=False, encoding="utf-8")
    lines = [f"ITW analysis run log - {_dt.datetime.now().isoformat(timespec='seconds')}",
             f"Data folder : {data_dir}", f"Output folder: {out_dir}",
             f"Subjects found ({len(ds.subjects)}): {', '.join(ds.subjects)}",
             f"Files read ({len(ds.found_files)}): {', '.join(ds.found_files)}",
             f"Records (subject x condition x limb): {len(ds.records)}", "",
             "Missing files / unpaired:"] + [f"  {m}" for m in ds.missing or ["  none"]]
    lines += ["", "Ignored files (name did not match the pattern):"]
    lines += [f"  {m}" for m in ds.ignored] or ["  none"]
    lines += ["", "Files skipped because of parse problems:"]
    lines += [f"  {n}: {e}" for n, e in ds.failed] or ["  none"]
    lines += ["", "QC flags per record:"]
    for r in ds.records:
        lines.append(f"  {r.label:<14} {r.res['severity']:<36} {';'.join(r.res['flags']) or '-'}")
    lines += ["", "Parameters used (Config):"] + [f"  {k} = {v}" for k, v in asdict(cfg).items()]
    (out_dir / "run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    return df


# =============================================================================
# PLOTTING (shared by the viewer and the PDF export)
# =============================================================================
KIND_META = {"angle": ("Ankle angle", "Angle (deg, + = dorsiflexion)", "% gait cycle"),
             "moment": ("Ankle plantarflexor moment", "Moment (Nm/kg)", "% gait cycle"),
             "power": ("Ankle power", "Power (W/kg)", "% gait cycle"),
             "mal": ("Moment-angle loop (quasi-stiffness)", "Moment (Nm/kg)", "Angle (deg)")}
PLOT_TYPES = [("1 Angle vs %GC", "angle"), ("2 Moment vs %GC", "moment"),
              ("3 Power vs %GC", "power"), ("4 Moment-angle loop", "mal"),
              ("5 Pre vs Post overlay", "prepost"), ("6a Stiffness by type", "g_stiff"),
              ("6b Criteria heatmap", "g_heat"), ("6c AM1/AM2 vs 3rd rocker", "g_scatter"),
              ("6d Pre->Post stiffness", "g_change"), ("6e Push-off power pre/post", "g_pushoff")]
PER_RECORD = {"angle", "moment", "power", "mal", "prepost"}
PAGED = {"g_heat", "g_change", "g_pushoff"}
DEFAULT_OPTS = {"overlay_prepost": False, "overlay_both_limbs": False, "markers": True,
                "regression": True, "thresholds": True}
BOX = (0.30, 0.76, 0.93, 0.10)   # left, right, top, bottom of the plot area


def _legend(ax, **kw) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, **kw)


def _plot_kind(ax, r: Record, kind: str, **kw) -> None:
    n = r.n_use
    if kind == "mal":
        ax.plot(r.angle[:n], r.moment[:n], **kw)
    else:
        ax.plot(r.pct[:n], getattr(r, kind)[:n], **kw)


def _finish(ax, rec: Record, kind: str, pos, extra: str = "") -> None:
    title, ylab, xlab = KIND_META[kind]
    t = f"{rec.label} - {title}"
    if pos:
        t += f"   (record {pos[0]} of {pos[1]})"
    ax.set_title(t + extra, fontsize=9)
    ax.set_xlabel(xlab, fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    if kind != "mal":
        ax.set_xlim(0, 100)
    _legend(ax)


def _secondary(ax, rec: Record, ds: Dataset, opts: dict, kind: str) -> None:
    if opts.get("overlay_prepost"):
        o = ds.get(rec.subject, "Post" if rec.condition == "Pre" else "Pre", rec.limb)
        if o is not None:
            _plot_kind(ax, o, kind, color="gray", ls="--", lw=1.3, label=f"{o.condition} (same limb)")
    if opts.get("overlay_both_limbs"):
        o = ds.get(rec.subject, rec.condition, "R" if rec.limb == "L" else "L")
        if o is not None:
            _plot_kind(ax, o, kind, color=LIMB_COL[o.limb], ls=":", lw=1.3, alpha=0.7,
                       label=f"other limb ({o.limb})")


def draw_angle(ax, rec, ds, opts, pos=None, finish=True):
    r, ev = rec.res, rec.res.get("_idx")
    _plot_kind(ax, rec, "angle", color=LIMB_COL[rec.limb], lw=2, label=rec.label)
    _secondary(ax, rec, ds, opts, "angle")
    if ev and opts.get("markers", True):
        if r["_crit"][0]:
            ax.axvspan(0, rec.pct[ev["fr"]], color="tab:green", alpha=0.2, label="1st rocker")
        ax.axvspan(r["second_rocker_start_pctGC"], r["second_rocker_end_pctGC"],
                   color="tab:orange", alpha=0.15, label="2nd rocker")
        ax.axvline(r["third_rocker_onset_pctGC"], color="tab:red", lw=1.6, label="3rd rocker onset")
        ax.axvline(r["toe_off_pctGC"], color="k", ls="--", lw=1, label="toe-off")
        ax.plot(r["third_rocker_onset_pctGC"], r["peak_dorsiflexion_angle_deg"], "v", color="tab:red", ms=8)
    if ev and opts.get("thresholds", True):
        ax.axvline(CONFIG.early_third_rocker_pct, color="tab:red", ls=":", alpha=0.6, label="30% GC cutoff")
        ax.axvline(CONFIG.first_rocker_end_pct, color="tab:green", ls=":", alpha=0.6, label="12% GC (1st-rocker window)")
        ax.axhline(-CONFIG.ic_plantarflexion_threshold_deg, color="gray", ls=":", label="-5 deg IC threshold")
    if finish:
        _finish(ax, rec, "angle", pos)


def draw_moment(ax, rec, ds, opts, pos=None, finish=True):
    r, ev = rec.res, rec.res.get("_idx")
    _plot_kind(ax, rec, "moment", color=LIMB_COL[rec.limb], lw=2, label=rec.label)
    _secondary(ax, rec, ds, opts, "moment")
    if ev and r["kinetics_valid"] and opts.get("markers", True):
        ax.plot(r["am1_time_pctGC"], r["am1_Nm_per_kg"], "o", color="tab:green", ms=8,
                label=f"AM1 = {r['am1_Nm_per_kg']:.2f}")
        ax.plot(r["am2_time_pctGC"], r["am2_Nm_per_kg"], "s", color="tab:purple", ms=8,
                label=f"AM2 = {r['am2_Nm_per_kg']:.2f}")
        ax.axvline(r["toe_off_pctGC"], color="k", ls="--", lw=1, label="toe-off")
        ax.text(0.98, 0.95, f"AM1/AM2 = {r['am1_am2_ratio']:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, bbox=dict(fc="w", alpha=0.8))
    if ev and r["kinetics_valid"] and opts.get("thresholds", True):
        ax.axhline(r["am2_Nm_per_kg"], color="tab:purple", ls=":", alpha=0.6, label="AM2 level (AM1>AM2 if above)")
        ax.axvline(CONFIG.am1_end_pct, color="gray", ls=":", alpha=0.6, label="35% GC (AM1/AM2 split)")
    if ev and not r["kinetics_valid"]:
        ax.text(0.5, 0.6, "No valid kinetics (moment < 0.1 Nm/kg)", transform=ax.transAxes,
                ha="center", color="tab:red", fontsize=11)
    if finish:
        _finish(ax, rec, "moment", pos)


def draw_power(ax, rec, ds, opts, pos=None, finish=True):
    r, ev = rec.res, rec.res.get("_idx")
    n = rec.n_use
    pct, p = rec.pct[:n], rec.power[:n]
    _plot_kind(ax, rec, "power", color=LIMB_COL[rec.limb], lw=2, label=rec.label)
    _secondary(ax, rec, ds, opts, "power")
    ax.axhline(0, color="k", lw=0.6)
    if ev and opts.get("markers", True):
        k = ev["toe"] + 1
        ax.fill_between(pct[:k], p[:k], 0, where=p[:k] >= 0, color="tab:green", alpha=0.3, label="generation")
        ax.fill_between(pct[:k], p[:k], 0, where=p[:k] < 0, color="tab:red", alpha=0.3, label="absorption")
        ax.axvline(r["toe_off_pctGC"], color="k", ls="--", lw=1, label="toe-off")
        if np.isfinite(r.get("peak_pushoff_power_W_per_kg", NAN)):
            ax.plot(r["pushoff_power_time_pctGC"], r["peak_pushoff_power_W_per_kg"], "*",
                    color="gold", mec="k", ms=15, label="push-off peak")
            ax.annotate(f"{r['peak_pushoff_power_W_per_kg']:.2f} W/kg @ {r['pushoff_power_time_pctGC']:.0f}%",
                        (r["pushoff_power_time_pctGC"], r["peak_pushoff_power_W_per_kg"]),
                        xytext=(8, 8), textcoords="offset points", fontsize=8)
        ax.axvline(CONFIG.pushoff_start_pct, color="gray", ls=":", alpha=0.5, label="push-off search start")
    if finish:
        _finish(ax, rec, "power", pos)


def draw_mal(ax, rec, ds, opts, pos=None, finish=True):
    r, ev = rec.res, rec.res.get("_idx")
    n = rec.n_use
    a, m = rec.angle[:n], rec.moment[:n]
    _secondary(ax, rec, ds, opts, "mal")
    if not ev:
        ax.plot(a, m, color=LIMB_COL[rec.limb], label=rec.label)
    else:
        toe = ev["toe"]
        ax.plot(a[toe:], m[toe:], "-", color="lightgray", lw=1, label="swing / after toe-off")
        phases = []
        if r["_crit"][0]:
            phases.append(("1st rocker", np.arange(0, ev["fr"] + 1), "tab:green"))
        phases.append(("2nd rocker", np.arange(ev["sr_start"], ev["peak"] + 1), "tab:orange"))
        phases.append(("powered PF", np.arange(ev["peak"], toe + 1), "tab:purple"))
        for name, idx, col in phases:
            ax.plot(a[idx], m[idx], "-o", color=col, ms=3, lw=1.8, label=name)
        for k in range(0, toe, 3):
            ax.annotate("", xy=(a[k + 1], m[k + 1]), xytext=(a[k], m[k]),
                        arrowprops=dict(arrowstyle="->", color="k", lw=0.7))
        ax.plot(a[0], m[0], "k^", ms=8, label="initial contact")
        if opts.get("regression", True):
            for key, col, nm in (("second", "tab:orange", "2nd rocker"), ("first", "tab:green", "1st rocker"),
                                 ("powered", "tab:purple", "powered PF")):
                f = r["_fits"].get(key)
                if f and np.isfinite(f["slope"]):
                    xs = np.array([f["xmin"], f["xmax"]])
                    ax.plot(xs, f["slope"] * xs + f["intercept"], "--", color=col, lw=2.2,
                            label=f"{nm}: k={f['slope']:.3f} Nm/kg/deg, R2={fmt(f['r2'], '{:.2f}')}")
        if not r["kinetics_valid"]:
            ax.text(0.5, 0.6, "No valid kinetics", transform=ax.transAxes, ha="center", color="tab:red")
    if finish:
        _finish(ax, rec, "mal", pos)


DRAW: Dict[str, Callable] = {"angle": draw_angle, "moment": draw_moment, "power": draw_power, "mal": draw_mal}


def info_text(rec: Record, pos=None) -> str:
    r = rec.res
    L = [rec.label + (f"   [record {pos[0]} of {pos[1]}]" if pos else ""),
         f"Severity: {r['severity']}", ""]
    if not r.get("_ok"):
        return "\n".join(L + ["Analysis failed", "Flags: " + ";".join(r["flags"])])
    L += ["Alvarez criteria",
          f" 1st ankle rocker : {r['first_rocker_present']}",
          f" Early 3rd rocker : {r['early_third_rocker']} ({fmt(r['third_rocker_onset_pctGC'], '{:.1f}')} %GC, cutoff 30)",
          f" AM1 > AM2        : {r['am1_gt_am2']} (ratio {fmt(r['am1_am2_ratio'], '{:.2f}')})", "",
          "Events (%GC)",
          f" IC angle         : {fmt(r['ic_angle_deg'], '{:.1f}')} deg",
          f" 2nd rocker       : {fmt(r['second_rocker_start_pctGC'], '{:.0f}')} -> {fmt(r['second_rocker_end_pctGC'], '{:.1f}')}",
          f" Peak DF          : {fmt(r['peak_dorsiflexion_angle_deg'], '{:.1f}')} deg",
          f" Toe-off          : {fmt(r['toe_off_pctGC'], '{:.0f}')}",
          f" AM1 / AM2        : {fmt(r['am1_Nm_per_kg'], '{:.2f}')} @ {fmt(r['am1_time_pctGC'], '{:.0f}')} / "
          f"{fmt(r['am2_Nm_per_kg'], '{:.2f}')} @ {fmt(r['am2_time_pctGC'], '{:.0f}')}", "",
          "Quasi-stiffness (Nm/kg/deg)",
          f" 2nd rocker k     : {fmt(r['second_rocker_k_Nm_per_kg_per_deg'])}  R2 {fmt(r['second_rocker_R2'], '{:.2f}')}",
          f"   n={fmt(r['second_rocker_n_points'], '{:.0f}')}  ROM={fmt(r['second_rocker_ROM_deg'], '{:.1f}')} deg",
          f" sub-window k     : {fmt(r['second_rocker_subwindow_k_Nm_per_kg_per_deg'])}",
          f" secant k         : {fmt(r['second_rocker_secant_k_Nm_per_kg_per_deg'])}",
          f" 1st rocker k     : {fmt(r['first_rocker_k_Nm_per_kg_per_deg'])}",
          f" powered-PF k     : {fmt(r['powered_pf_k_Nm_per_kg_per_deg'])}",
          f" 2nd rocker k /rad: {fmt(r['second_rocker_k_Nm_per_kg_per_rad'], '{:.2f}')} Nm/kg/rad",
          f" k (Nm/deg)       : {fmt(r['second_rocker_k_Nm_per_deg'])}", "",
          "Work (J/kg)",
          f" absorbed (2nd)   : {fmt(r['work_absorbed_second_rocker_J_per_kg'])}",
          f" released (PF)    : {fmt(r['work_released_powered_pf_J_per_kg'])}",
          f" release/absorb   : {fmt(r['work_release_to_absorb_ratio'], '{:.2f}')}", "",
          "Push-off power",
          f" peak             : {fmt(r['peak_pushoff_power_W_per_kg'], '{:.2f}')} W/kg @ {fmt(r['pushoff_power_time_pctGC'], '{:.0f}')} %GC", "",
          "QC flags: " + (", ".join(r["flags"]) or "none")]
    return "\n".join(L)


# ---- group views ----------------------------------------------------------------------------
def n_pages(kind: str, recs: List[Record], ds: Dataset, limb: str, cfg: Config) -> int:
    if kind == "g_heat":
        total = len(recs)
    elif kind in ("g_change", "g_pushoff"):
        total = len(ds.pairs(limb))
    else:
        return 1
    return max(1, math.ceil(total / cfg.rows_per_page))


def draw_g_stiff(ax, recs: List[Record]) -> None:
    rng = np.random.default_rng(0)
    for xi, t in enumerate(TYPE_ORDER):
        sel = [r for r in recs if r.res["severity"] == t]
        vals = np.array([r.res.get("second_rocker_k_Nm_per_kg_per_deg", NAN) for r in sel], float)
        ok = np.isfinite(vals)
        if sel and ok.any():
            for cond, mk in (("Pre", "o"), ("Post", "s")):
                v = np.array([x for x, r in zip(vals, sel) if r.condition == cond and np.isfinite(x)])
                if v.size:
                    ax.scatter(xi + rng.uniform(-0.2, 0.2, v.size), v, marker=mk, color=TYPE_COL[t],
                               edgecolor="k", alpha=0.8, label=cond if xi == 0 else None)
            ax.hlines(np.nanmean(vals), xi - 0.3, xi + 0.3, color="k", lw=2.5)
        ax.text(xi, 1.0, f"n={int(ok.sum())}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.set_xticks(range(len(TYPE_ORDER)))
    ax.set_xticklabels([t.replace("Unclassifiable - no valid kinetics", "No kinetics") for t in TYPE_ORDER], fontsize=8)
    ax.set_ylabel("2nd-rocker quasi-stiffness (Nm/kg/deg)", fontsize=8)
    ax.set_title("Quasi-stiffness by Alvarez severity type (o = Pre, square = Post, bar = mean)", fontsize=9)
    ax.grid(alpha=0.25)


def draw_g_heat(ax, recs: List[Record], page: int, cfg: Config) -> str:
    npg = max(1, math.ceil(len(recs) / cfg.rows_per_page))
    page = min(max(page, 0), npg - 1)
    sub = recs[page * cfg.rows_per_page:(page + 1) * cfg.rows_per_page]
    M = np.zeros((len(sub), 3))
    for i, r in enumerate(sub):
        for j, v in enumerate(r.res["_crit"]):
            M[i, j] = 0 if v is None else (2 if v else 1)
    ax.imshow(M, cmap=ListedColormap(["#e0e0e0", "#ffffff", "#7fb3d5"]), vmin=-0.5, vmax=2.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(3):
            ax.text(j, i, ["n/a", "No", "Yes"][int(M[i, j])], ha="center", va="center", fontsize=7)
        ax.text(3.0, i, short_sev(sub[i].res["severity"]), ha="left", va="center", fontsize=8,
                color=TYPE_COL.get(sub[i].res["severity"], "k"), fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1st rocker", "Early 3rd rocker", "AM1 > AM2"], fontsize=8)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([r.label for r in sub], fontsize=7)
    ax.set_xlim(-0.5, 3.4)
    ax.set_title(f"Alvarez criteria matrix - page {page + 1}/{npg}  (T1/T2/T3, U = unclassified, NK = no kinetics)", fontsize=9)
    return f"page {page + 1}/{npg}"


def draw_g_scatter(ax, recs: List[Record]) -> None:
    for t in TYPE_ORDER:
        sel = [r for r in recs if r.res["severity"] == t and np.isfinite(r.res.get("am1_am2_ratio", NAN))]
        if sel:
            ax.scatter([r.res["third_rocker_onset_pctGC"] for r in sel], [r.res["am1_am2_ratio"] for r in sel],
                       color=TYPE_COL[t], edgecolor="k", label=f"{t} (n={len(sel)})", alpha=0.85)
    ax.axvline(CONFIG.early_third_rocker_pct, color="gray", ls=":", label="30% GC cutoff")
    ax.axhline(1.0, color="gray", ls="--", label="AM1 = AM2")
    ax.set_xlabel("3rd-rocker onset (% GC)", fontsize=8)
    ax.set_ylabel("AM1 / AM2", fontsize=8)
    ax.set_title("AM1/AM2 vs timing of 3rd-rocker onset", fontsize=9)
    ax.grid(alpha=0.25)
    _legend(ax)


def draw_g_paired(fig, gs, ds: Dataset, metric: str, ylabel: str, title: str, page: int, limb: str,
                  cfg: Config) -> Tuple[list, str]:
    pairs = ds.pairs(limb)
    npg = max(1, math.ceil(len(pairs) / cfg.rows_per_page))
    page = min(max(page, 0), npg - 1)
    sub = pairs[page * cfg.rows_per_page:(page + 1) * cfg.rows_per_page]
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    names, deltas, cols = [], [], []
    for a, b in sub:
        va, vb = a.res.get(metric, NAN), b.res.get(metric, NAN)
        col = LIMB_COL[a.limb]
        ax1.plot([0, 1], [va, vb], "-o", color=col, alpha=0.7)
        names.append(f"{a.subject} {a.limb}  {short_sev(a.res['severity'])}->{short_sev(b.res['severity'])}")
        deltas.append(vb - va)
        cols.append(col)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Pre", "Post"])
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylabel(ylabel, fontsize=8)
    ax1.set_title(f"{title} (blue = L, red = R)", fontsize=9)
    ax1.grid(alpha=0.25)
    ax2.barh(range(len(sub)), [0 if not np.isfinite(d) else d for d in deltas], color=cols)
    ax2.set_yticks(range(len(sub)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.invert_yaxis()
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("Post - Pre", fontsize=8)
    ax2.set_title(f"Change per limb and severity type - page {page + 1}/{npg}", fontsize=9)
    ax2.grid(alpha=0.25)
    if not pairs:
        ax1.text(0.5, 0.5, "No subjects with both Pre and Post", transform=ax1.transAxes, ha="center")
    return [ax1, ax2], f"page {page + 1}/{npg}  ({len(pairs)} paired limbs)"


def _text_axes(fig, rect, text: str):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.text(0, 1, text, va="top", ha="left", family="monospace", fontsize=8)
    return ax


def render(fig, kind: str, rec: Optional[Record], recs: List[Record], ds: Dataset, opts: dict,
           page: int = 0, box=BOX, limb: str = "Both", pos=None, cfg: Config = CONFIG):
    """Draw one plot type into fig. Returns (axes created, info text)."""
    l, r_, t, b = box
    if kind in ("angle", "moment", "power", "mal"):
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=l, right=r_, top=t, bottom=b)[0])
        DRAW[kind](ax, rec, ds, opts, pos)
        return [ax], info_text(rec, pos)
    if kind == "prepost":
        gs = fig.add_gridspec(2, 2, left=l, right=r_, top=t, bottom=b, hspace=0.35, wspace=0.28)
        axes = []
        for i, k in enumerate(("angle", "moment", "power", "mal")):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            for cond in ("Pre", "Post"):
                o = ds.get(rec.subject, cond, rec.limb)
                if o is None:
                    continue
                _plot_kind(ax, o, k, color=COND_COL[cond], lw=2, label=f"{cond}: {short_sev(o.res['severity'])}")
                if k == "angle" and o.res.get("_idx"):
                    ax.axvline(o.res["third_rocker_onset_pctGC"], color=COND_COL[cond], ls=":", lw=1.2)
            if k == "power":
                ax.axhline(0, color="k", lw=0.6)
            _finish(ax, rec, k, None)
            ax.set_title(f"{rec.subject} {rec.limb} - {KIND_META[k][0]}: Pre vs Post", fontsize=8)
            axes.append(ax)
        txt = "\n\n".join(info_text(ds.get(rec.subject, c, rec.limb)) for c in ("Pre", "Post")
                          if ds.get(rec.subject, c, rec.limb) is not None)
        return axes, txt
    if kind == "dash":
        gs = fig.add_gridspec(2, 2, left=0.05, right=0.66, top=0.93, bottom=0.07, hspace=0.35, wspace=0.25)
        axes = []
        for i, k in enumerate(("angle", "moment", "power", "mal")):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            DRAW[k](ax, rec, ds, opts, pos if i == 0 else None)
            axes.append(ax)
        axes.append(_text_axes(fig, [0.69, 0.05, 0.30, 0.90], info_text(rec, pos)))
        return axes, ""
    ax_list: list
    info = ""
    if kind == "g_stiff":
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=l, right=r_, top=t, bottom=b)[0])
        draw_g_stiff(ax, recs)
        ax_list = [ax]
        info = f"{len(recs)} records in view"
    elif kind == "g_heat":
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=l + 0.04, right=r_, top=t, bottom=b)[0])
        info = draw_g_heat(ax, recs, page, cfg)
        ax_list = [ax]
    elif kind == "g_scatter":
        ax = fig.add_subplot(fig.add_gridspec(1, 1, left=l, right=r_, top=t, bottom=b)[0])
        draw_g_scatter(ax, recs)
        ax_list = [ax]
        info = f"{len(recs)} records in view"
    elif kind in ("g_change", "g_pushoff"):
        gs = fig.add_gridspec(1, 2, left=l, right=r_, top=t, bottom=b, wspace=0.55, width_ratios=[1, 1.3])
        if kind == "g_change":
            ax_list, info = draw_g_paired(fig, gs, ds, "second_rocker_k_Nm_per_kg_per_deg",
                                          "2nd-rocker stiffness (Nm/kg/deg)", "Stiffness Pre -> Post", page, limb, cfg)
        else:
            ax_list, info = draw_g_paired(fig, gs, ds, "peak_pushoff_power_W_per_kg",
                                          "Peak push-off power (W/kg)", "Push-off power Pre -> Post", page, limb, cfg)
    else:
        raise ValueError(kind)
    return ax_list, info


def export_pdf(ds: Dataset, cfg: Config, path: Path, opts: Optional[dict] = None) -> Path:
    """One dashboard page per record + one Pre-vs-Post page per paired limb + all group pages."""
    from matplotlib.backends.backend_pdf import PdfPages
    opts = dict(DEFAULT_OPTS if opts is None else opts)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = len(ds.records)
    with PdfPages(path) as pdf:
        for i, rec in enumerate(ds.records, 1):
            fig = Figure(figsize=(16, 9))
            render(fig, "dash", rec, ds.records, ds, opts, pos=(i, N))
            pdf.savefig(fig)
        for a, _b in ds.pairs():
            fig = Figure(figsize=(16, 9))
            axes, txt = render(fig, "prepost", a, ds.records, ds, opts, box=(0.05, 0.66, 0.93, 0.07))
            _text_axes(fig, [0.69, 0.05, 0.30, 0.90], txt)
            pdf.savefig(fig)
        for kind in ("g_stiff", "g_heat", "g_scatter", "g_change", "g_pushoff"):
            for pg in range(n_pages(kind, ds.records, ds, "Both", cfg)):
                fig = Figure(figsize=(16, 9))
                render(fig, kind, None, ds.records, ds, opts, page=pg, box=(0.08, 0.96, 0.93, 0.08), cfg=cfg)
                pdf.savefig(fig)
    return path


# =============================================================================
# INTERACTIVE VIEWER
# =============================================================================
class Viewer:
    """Matplotlib viewer. Keeps references to all widgets so they are not garbage collected."""

    def __init__(self, ds: Dataset, cfg: Config, out_dir: Path) -> None:
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, TextBox
        self._Button, self._Check, self._Radio, self._TextBox = Button, CheckButtons, RadioButtons, TextBox
        self.ds, self.cfg, self.out_dir = ds, cfg, out_dir
        self.opts = dict(DEFAULT_OPTS)
        self.cond, self.limb = "Both", "Both"
        self.types = {"Type 1": True, "Type 2": True, "Type 3": True, "Other": True}
        self.pt, self.page = 0, 0
        self.cur: Optional[Record] = ds.records[0] if ds.records else None
        self._w: list = []
        self.plot_axes: list = []
        plt.rcParams["keymap.save"] = []
        plt.rcParams["keymap.back"] = ["backspace"]
        plt.rcParams["keymap.forward"] = []
        plt.rcParams["keymap.home"] = ["home"]
        self.fig = plt.figure(figsize=(16, 9))
        try:
            self.fig.canvas.manager.set_window_title("ITW ankle analysis viewer")
        except Exception:  # noqa: BLE001
            pass
        self._build_widgets()
        self.ax_text = self.fig.add_axes([0.78, 0.08, 0.21, 0.85])
        self.status = self.fig.text(0.30, 0.02, "", fontsize=8)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    # ---- widgets ----------------------------------------------------------------------------
    def _radio(self, rect, labels, active, cb, title=None, fs=8):
        ax = self.fig.add_axes(rect)
        if title:
            ax.set_title(title, fontsize=8, loc="left")
        rb = self._Radio(ax, labels, active=active)
        for t in rb.labels:
            t.set_fontsize(fs)
        rb.on_clicked(cb)
        self._w.append(rb)
        return rb

    def _check(self, rect, labels, states: dict, cb, title=None):
        ax = self.fig.add_axes(rect)
        if title:
            ax.set_title(title, fontsize=8, loc="left")
        ck = self._Check(ax, labels, [states[k] for k in labels])
        for t in ck.labels:
            t.set_fontsize(8)
        ck.on_clicked(cb)
        self._w.append(ck)
        return ck

    def _button(self, rect, label, cb):
        ax = self.fig.add_axes(rect)
        bt = self._Button(ax, label)
        bt.label.set_fontsize(8)
        bt.on_clicked(cb)
        self._w.append(bt)
        return bt

    def _build_widgets(self) -> None:
        self._radio([0.01, 0.58, 0.21, 0.37], [n for n, _ in PLOT_TYPES], 0, self._set_plot, "Plot type")
        self._radio([0.01, 0.47, 0.09, 0.09], ["Both", "Pre", "Post"], 0, self._set_cond, "Condition")
        self._radio([0.12, 0.47, 0.09, 0.09], ["Both", "L", "R"], 0, self._set_limb, "Limb")
        self._check([0.01, 0.35, 0.10, 0.10], list(self.types), self.types, self._tog_type, "Severity")
        labels = {"Pre-vs-Post overlay": "overlay_prepost", "Both limbs": "overlay_both_limbs",
                  "Event markers": "markers", "Regression line": "regression", "Alvarez thresholds": "thresholds"}
        self._ov_map = labels
        self._check([0.12, 0.28, 0.17, 0.17], list(labels), {k: self.opts[v] for k, v in labels.items()},
                    self._tog_overlay, "Overlays")
        self._button([0.01, 0.22, 0.07, 0.045], "< Prev", lambda e: self.step(-1))
        self._button([0.09, 0.22, 0.07, 0.045], "Next >", lambda e: self.step(+1))
        self._button([0.17, 0.22, 0.10, 0.045], "Shuffle", lambda e: self.shuffle())
        self._button([0.01, 0.165, 0.07, 0.045], "< Subj", lambda e: self.jump_subject(-1))
        self._button([0.09, 0.165, 0.07, 0.045], "Subj >", lambda e: self.jump_subject(+1))
        tax = self.fig.add_axes([0.20, 0.165, 0.07, 0.045])
        self.tb = self._TextBox(tax, "Go to ", initial="", label_pad=0.1)
        self.tb.label.set_fontsize(8)
        self.tb.on_submit(self.on_submit)
        self._w.append(self.tb)
        self._button([0.01, 0.11, 0.12, 0.045], "Save figure (S)", lambda e: self.save_fig())
        self._button([0.15, 0.11, 0.12, 0.045], "Export PDF (E)", lambda e: self.export_all())

    # ---- state ------------------------------------------------------------------------------------
    def _recs(self) -> List[Record]:
        out = []
        for r in self.ds.records:
            if self.cond != "Both" and r.condition != self.cond:
                continue
            if self.limb != "Both" and r.limb != self.limb:
                continue
            if not self.types.get(sev_key(r.res["severity"]), True):
                continue
            out.append(r)
        return out

    def _pos(self, recs: List[Record]) -> int:
        if not recs:
            return 0
        if self.cur not in recs:
            self.cur = recs[0]
        return recs.index(self.cur)

    def _kind(self) -> str:
        return PLOT_TYPES[self.pt][1]

    def _set_plot(self, label):
        self.pt = [n for n, _ in PLOT_TYPES].index(label)
        self.page = 0
        self.redraw()

    def _set_cond(self, label):
        self.cond = label
        self.page = 0
        self.redraw()

    def _set_limb(self, label):
        self.limb = label
        self.page = 0
        self.redraw()

    def _tog_type(self, label):
        self.types[label] = not self.types[label]
        self.page = 0
        self.redraw()

    def _tog_overlay(self, label):
        k = self._ov_map[label]
        self.opts[k] = not self.opts[k]
        self.redraw()

    def step(self, d: int) -> None:
        recs = self._recs()
        kind = self._kind()
        if kind in PAGED:
            npg = n_pages(kind, recs, self.ds, self.limb, self.cfg)
            self.page = (self.page + d) % npg
        elif recs:
            self.cur = recs[(self._pos(recs) + d) % len(recs)]
        self.redraw()

    def shuffle(self) -> None:
        recs = self._recs()
        if recs:
            self.cur = random.choice(recs)
        self.redraw()

    def change_plot(self, d: int) -> None:
        self.pt = (self.pt + d) % len(PLOT_TYPES)
        self.page = 0
        self.redraw()

    def jump_subject(self, d: int) -> None:
        recs = self._recs()
        if not recs:
            return
        subs: List[str] = []
        for r in recs:
            if r.subject not in subs:
                subs.append(r.subject)
        cur = self.cur.subject if self.cur in recs else subs[0]
        tgt = subs[(subs.index(cur) + d) % len(subs)]
        self._goto_subject(tgt, recs)

    def _goto_subject(self, subject: str, recs: List[Record]) -> None:
        cands = [r for r in recs if r.subject == subject]
        same = [r for r in cands if self.cur and r.condition == self.cur.condition and r.limb == self.cur.limb]
        self.cur = (same or cands)[0]
        self.redraw()

    def on_submit(self, text: str) -> None:
        digits = re.findall(r"\d+", text)
        if not digits:
            self._say("Type a subject number, e.g. 3 or S03")
            return
        n = int(digits[0])
        recs = self._recs()
        cands = [r for r in recs if natural_key(r.subject)[0] == n]
        if not cands:
            self._say(f"Subject {text!r} not found (or filtered out)")
            return
        self._goto_subject(cands[0].subject, recs)

    def _say(self, msg: str) -> None:
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    def save_fig(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = (self.cur.label.replace(" ", "_") if self.cur and self._kind() in PER_RECORD else "group")
        p = self.out_dir / f"view_{name}_{self._kind()}.png"
        self.fig.savefig(p, dpi=150)
        self._say(f"Saved {p}")
        print("Saved", p)

    def export_all(self) -> None:
        self._say("Exporting PDF ... please wait")
        self.fig.canvas.draw()
        p = export_pdf(self.ds, self.cfg, self.out_dir / "all_figures.pdf", self.opts)
        self._say(f"Exported {p}")
        print("Exported", p)

    def on_key(self, ev) -> None:
        if getattr(self.tb, "capturekeystrokes", False):
            return
        k = ev.key
        if k == "right":
            self.step(+1)
        elif k == "left":
            self.step(-1)
        elif k == "up":
            self.change_plot(-1)
        elif k == "down":
            self.change_plot(+1)
        elif k == "pagedown":
            self.jump_subject(+1)
        elif k == "pageup":
            self.jump_subject(-1)
        elif k in ("r", "R"):
            self.shuffle()
        elif k in ("s", "S"):
            self.save_fig()
        elif k in ("e", "E"):
            self.export_all()

    # ---- drawing ----------------------------------------------------------------------------------------
    def redraw(self) -> None:
        for ax in self.plot_axes:
            try:
                self.fig.delaxes(ax)
            except Exception:  # noqa: BLE001
                pass
        self.plot_axes = []
        self.ax_text.clear()
        self.ax_text.axis("off")
        recs = self._recs()
        kind = self._kind()
        msg = ""
        if not recs and (kind in PER_RECORD or kind in ("g_stiff", "g_heat", "g_scatter")):
            self.ax_text.text(0, 1, "No records match the current filters.", va="top", color="tab:red", fontsize=10)
            ax = self.fig.add_axes([BOX[0], BOX[3], BOX[1] - BOX[0], BOX[2] - BOX[3]])
            ax.axis("off")
            ax.text(0.5, 0.5, "No records match the current filters.\nChange Condition / Limb / Severity filters.",
                    ha="center", va="center", fontsize=13, color="tab:red")
            self.plot_axes = [ax]
            self.fig.canvas.draw_idle()
            return
        try:
            if kind in PER_RECORD:
                i = self._pos(recs)
                axes, msg = render(self.fig, kind, self.cur, recs, self.ds, self.opts, pos=(i + 1, len(recs)))
            else:
                npg = n_pages(kind, recs, self.ds, self.limb, self.cfg)
                self.page = min(self.page, npg - 1)
                axes, msg = render(self.fig, kind, None, recs, self.ds, self.opts, page=self.page,
                                   limb=self.limb, cfg=self.cfg)
                msg = f"{PLOT_TYPES[self.pt][0]}\n{msg}\n(Left/Right = page)"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            axes, msg = [], f"Plot error: {e}"
        self.plot_axes = axes
        self.ax_text.text(0, 1, msg, va="top", ha="left", family="monospace", fontsize=8)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


# =============================================================================
# SELF-TEST
# =============================================================================
def run_selftest(data_dir: Path, cfg: Config) -> bool:
    ok_all = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok_all
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        ok_all &= bool(cond)

    print("Self-test")
    # 1. regression recovers a known slope
    x = np.linspace(-10, 10, 21)
    f = _ols(x, 0.07 * x + 0.5)
    check("OLS recovers known slope (0.07)", abs(f["slope"] - 0.07) / 0.07 < 0.01, f"-> {f['slope']:.5f}")
    # 2. synthetic limb through the whole pipeline
    idx = np.arange(101)
    ang = np.interp(idx, [0, 40, 60, 100], [-20, 10, -30, -30]).astype(float)
    mom = np.clip(0.05 * (ang + 20), 0, None)
    res = analyze_limb(ang, mom, np.zeros(101), cfg)
    k = res["second_rocker_k_Nm_per_kg_per_deg"]
    check("Synthetic limb: second-rocker k = 0.05 within 1%", abs(k - 0.05) / 0.05 < 0.01, f"-> {k:.5f}")
    check("Synthetic limb: first rocker absent, early 3rd rocker not required", res["first_rocker_present"] == "No")
    # 3. data-based checks
    ds = build_dataset(data_dir, cfg)
    print(f"  Data folder: {data_dir}  ({len(ds.subjects)} subjects, {len(ds.records)} records)")
    rec = ds.get("S05", "Pre", "L")
    if rec is None:
        print("  [SKIP] S05_Precast Left hand-check: file not found")
    else:
        r = rec.res
        check("S05 Pre L: 1st rocker absent", r["first_rocker_present"] == "No")
        check("S05 Pre L: peak DF ~ -9.87 deg at ~18 %GC",
              abs(r["peak_dorsiflexion_angle_deg"] + 9.87) < 0.1 and abs(r["third_rocker_onset_pctGC"] - 18) < 1.5,
              f"-> {r['peak_dorsiflexion_angle_deg']:.2f} deg @ {r['third_rocker_onset_pctGC']:.1f}%")
        check("S05 Pre L: early 3rd rocker", r["early_third_rocker"] == "Yes")
        check("S05 Pre L: AM1/AM2 ~ 0.95, AM1 not > AM2",
              abs(r["am1_am2_ratio"] - 0.95) < 0.02 and r["am1_gt_am2"] == "No", f"-> {r['am1_am2_ratio']:.3f}")
        check("S05 Pre L: Type 2", r["severity"] == "Type 2", f"-> {r['severity']}")
        check("S05 Pre L: push-off ~ 3.6 W/kg near 52 %GC",
              abs(r["peak_pushoff_power_W_per_kg"] - 3.6) < 0.1 and abs(r["pushoff_power_time_pctGC"] - 52) < 2,
              f"-> {r['peak_pushoff_power_W_per_kg']:.2f} @ {r['pushoff_power_time_pctGC']:.1f}%")
    s6 = [ds.get("S06", "Post", lb) for lb in LIMBS]
    if s6[0] is None:
        print("  S06_Postcast not found - invalid-kinetics check skipped")
    else:
        check("S06 Post flagged as invalid kinetics", all(not r.res["kinetics_valid"] for r in s6))
    # 4. comparison with Alvarez Table 3
    if ds.records:
        df = long_table(ds)
        ref = {"third_rocker_onset_pctGC": (45.6, 24.5, 18.0), "am1_am2_ratio": (0.6, 0.8, 1.4),
               "am1_time_pctGC": (29.8, 23.9, 15.2)}
        print("  Comparison with Alvarez Table 3 (mean +/- SD, Types 1/2/3; small sample => rough only)")
        for col, refv in ref.items():
            parts = []
            for t, rv in zip(("Type 1", "Type 2", "Type 3"), refv):
                v = df.loc[df.severity == t, col].dropna()
                parts.append(f"{t}: {v.mean():.2f}+/-{v.std():.2f} (n={len(v)}, paper {rv})" if len(v) else
                             f"{t}: n=0 (paper {rv})")
            print(f"    {col}: " + " | ".join(parts))
    print("Self-test", "PASSED" if ok_all else "FAILED")
    return ok_all


# =============================================================================
# MAIN
# =============================================================================
def select_backend(no_gui: bool) -> bool:
    """Pick an interactive backend (QtAgg, TkAgg, MacOSX); fall back to Agg. True if GUI works."""
    if no_gui:
        plt.switch_backend("Agg")
        return False
    headless = sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not headless:
        for name in ("QtAgg", "TkAgg", "MacOSX"):
            try:
                plt.switch_backend(name)
                f = plt.figure()
                plt.close(f)
                print(f"Using interactive matplotlib backend: {name}")
                return True
            except Exception:  # noqa: BLE001
                continue
    plt.switch_backend("Agg")
    print("No interactive matplotlib backend available (headless session or no Qt/Tk installed).\n"
          "Running with --no-gui: figures are saved to the output folder instead.\n"
          "Tip: pip install PyQt6   (or install tkinter) to enable the viewer.")
    return False


def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def resolve_data_dir(arg: Optional[str], cfg: Config) -> Path:
    cands = [Path(arg)] if arg else [_script_dir(), _script_dir() / "data", Path.cwd(), Path.cwd() / "data"]
    for c in cands:
        if c.is_dir() and discover_files(c, cfg)[0]:
            return c.resolve()
    return cands[0].resolve()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ITW ankle rocker / quasi-stiffness / Alvarez classification")
    ap.add_argument("--data-dir", default=None, help="folder with S01_Precast.txt ... (default: script folder, then ./data)")
    ap.add_argument("--out", default=None, help="output folder (default: ./results next to the script)")
    ap.add_argument("--body-mass", default=None, help="CSV with columns subject,mass_kg (for Nm/deg)")
    ap.add_argument("--no-gui", action="store_true", help="skip the viewer; export CSVs and PDF only")
    ap.add_argument("--selftest", action="store_true", help="run validation checks and exit")
    ap.add_argument("--regex", default=None, help="override the filename regex")
    args = ap.parse_args(argv)
    cfg = CONFIG
    if args.regex:
        cfg.file_regex = args.regex
    data_dir = resolve_data_dir(args.data_dir, cfg)
    out_dir = Path(args.out).resolve() if args.out else _script_dir() / "results"
    print(f"Data folder  : {data_dir}\nOutput folder: {out_dir}")
    if args.selftest:
        plt.switch_backend("Agg")
        return 0 if run_selftest(data_dir, cfg) else 1
    gui = select_backend(args.no_gui)
    mass = read_body_mass(Path(args.body_mass)) if args.body_mass else {}
    ds = build_dataset(data_dir, cfg, mass)
    if not ds.records:
        print("No matching data files found. Expected names like S01_Precast.txt in", data_dir)
        return 1
    print(f"Subjects: {len(ds.subjects)} ({', '.join(ds.subjects)}); records: {len(ds.records)}")
    for m in ds.missing:
        print("  missing:", m)
    for n, e in ds.failed:
        print(f"  skipped {n}: {e}")
    write_outputs(ds, cfg, out_dir, data_dir)
    print("Wrote primary_outcomes.csv, ankle_quasi_stiffness.csv, itw_classification.csv, limb_summary.csv, run_log.txt")
    if not gui:
        p = export_pdf(ds, cfg, out_dir / "all_figures.pdf")
        print("Wrote", p)
        return 0
    Viewer(ds, cfg, out_dir).show()
    return 0


if __name__ == "__main__":
    sys.exit(main())


