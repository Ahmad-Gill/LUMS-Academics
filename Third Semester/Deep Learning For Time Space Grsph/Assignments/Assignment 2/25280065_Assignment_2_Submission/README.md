# AI651 Assignment 2: Deep Learning for Space, Time and Graphs

**Muhammad Ahmad Gill (25280065)** · LUMS, Fall 2026
Full code: <https://github.com/Ahmad-Gill/LUMS-Academics/tree/main/Third%20Semester/Deep%20Learning%20For%20Time%20Space%20Grsph/Assignments/Assignment%202>

This assignment has two parts. **Question 1** designs a deep-learning flood-forecasting system for a river network: a written proposal with no training. **Question 2** trains video-prediction models and tests whether they really learned how things move. The full answers are in the report; this page explains what is in the folder and what each model does.

## What is in this folder

| File / folder | What it is |
|---|---|
| `25280065_Assignment_2_Report.pdf` | **The report**: both questions, all tables and figures, reproducibility notes, references, use of AI |
| `25280065_Assignment_2_Report.docx` | Editable copy of the same report |
| `25280065_Question_2.ipynb` | All Question 2 code: data, models, training, evaluation, Tables 1–6, Figures 1–6 |
| `figures/` | Every figure in the report, named after its number in the report (see below) |

**Figure names:** `Q1_Fig03_…` is Figure Q1.3 and `Q2_Fig5_…` is Figure 5 of Question 2. `Q2_D04_…` is Diagram D4 and `Q2_P02_…` is Plot P2. The `.gif` files are animated versions of Figures 3 and 6(a).

## Question 1: River forecasting proposal

**Task.** A water authority wants, every 10 minutes, a forecast of the water level **one hour ahead** at four river stations (S1–S4). The river starts at a reservoir with controlled releases and has a monitored tributary (T1), an unmonitored tributary, and two rain gauges (R1, R2). One side of the valley has no rain gauge at all.

**Proposed model: the Three-way Decomposed River-Graph Forecaster** (Figure Q1.1).
- **River as a graph.** The river is a 7-node graph (reservoir, S1, T1, S2, S3, S4, plus a *virtual* node U for the unmonitored tributary) with edges pointing downstream only (Figure Q1.2). Rain gauges are mapped to the parts of the catchment they cover.
- **Input.** Every 10 minutes, the last 24 hours of levels, rainfall and releases are aligned onto one time grid, giving a tensor of shape `[batch, 144 steps, 7 nodes, 8 features]`. Missing or late readings are marked with a mask and a "data age" feature instead of being guessed silently.
- **Encoder.** A shared GRU reads each node's history.
- **Lag attention.** *Directed lag attention* lets each station look at its upstream neighbours at different time lags, so the model **learns travel times**, which get shorter when the river is high (Figure Q1.5).
- **Announced releases.** Release schedules announced for the next hour are added at the reservoir node.
- **Three heads.** Three heads split each forecast into **base flow**, a **gauged-rain** contribution and a **hidden-rain** contribution (Figure Q1.6). The hidden part is triggered by "surprise": water arriving that no gauge explains. This is how the design copes with the ungauged side of the valley and the unmonitored tributary.
- **Output.** For each station, a 10 %, 50 % and 90 % forecast of the level one hour ahead, so the authority gets an uncertainty band, not just a single number. The band widens when sensors are offline or hidden rain is suspected (Figure Q1.12).

The report also covers the assumptions, the exact data specification, the equations and training loss, how missing or delayed data and release-schedule changes are handled, why it can run every 10 minutes on a CPU, its limitations, and the extra data it would request. All Question 1 plots use synthetic data to *illustrate* the design and are labelled as such: they are not results.

## Question 2: Motion or memory?

**Task.** A ConvLSTM is trained on **Moving MNIST** (videos of two handwritten digits bouncing around a 64×64 box): it sees 5 frames and predicts the next 5. Each model is then tested on videos that are **mirrored**, **rotated by 90°** or **played 2× or 3× faster**. A model that truly learned motion should cope with all of them.

**Models**

| Model | What it is | What it tests |
|---|---|---|
| **A** | Baseline ConvLSTM: 2 layers, 64 channels, 3×3 kernels, 480,784 parameters, trained on normal-speed clips | Does a standard model learn motion? |
| **B** | Same as A, trained on randomly mirrored and rotated clips | Is the mirror/rotation failure about the digits' *appearance*? |
| **C** | Same as A, trained on 1×, 2× and 3× speed clips | Is the speed failure about *unfamiliar* motion? |
| **D** | Like C, but layer 2 uses a dilated convolution so information can travel further per frame (12 px instead of 8 px) | Is the speed failure about the network's *reach*? |
| **Final** | Model D wrapped in a **symmetrised forecaster**: predict on all 8 mirror/rotation versions of the clip, undo each transformation, average | Built-in, exact mirror/rotation consistency |

**Main findings** (test set; full numbers in Tables 2–6)
- Model A beats the simple baselines (error 0.0223, versus 0.0426 for an all-black prediction), but it is **slightly worse on rotated and mirrored clips** and **falls apart at 2× speed**.
- The mirror/rotation problem is about **appearance**: the training data already contains motion in every direction. Training with mirrored/rotated clips (B) removes the extra error, and symmetrisation makes the model exactly consistent (gap about 10⁻¹⁷, i.e. floating-point noise).
- The 2× failure is about **unfamiliar motion**: training on faster clips (C) cuts the 2× error from 0.0432 to 0.0291. At 3×, the larger reach of D helps only slightly beyond C.
- **Still unsolved:** when two digits overlap, every model merges them into one blob and cannot separate them again.

## Reproducing the results

- Run `25280065_Question_2.ipynb` top to bottom. It downloads Moving MNIST with `torchvision` and trains every model with seeds 0 and 1 (about 20 s per epoch on an Apple M2 Pro, 30 epochs per model).
- The scripts that regenerate the Question 1 figures and the Question 2 diagrams and plots are in the GitHub repository linked above (`Question1/figures/src/run_all.py` and `Question2/make_report_figures.py`).
