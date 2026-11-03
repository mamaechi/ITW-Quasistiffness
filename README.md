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
