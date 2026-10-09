[NeuralDecoderREADME.md](https://github.com/user-attachments/files/33265324/NeuralDecoderREADME.md)
# Motor Cortex Neural Decoding

Predicting a monkey's hand movements from the activity of motor-cortex neurons, using the public **MC_Maze** dataset from the [Neural Latents Benchmark](https://github.com/neurallatents/nlb_tools).

**Status:** Phase 1 (loading, cleaning, and exploring the data) is complete. Phase 2 (building decoders) is next.

![PSTHs of four strongly direction-tuned neurons](figures/03_psths.png)

---

## Project goal

Neurons in motor cortex change their firing depending on which way the arm is about to move. A **decoder** is a model that reads the activity of many neurons at once and predicts the movement from it. Brain-computer interfaces use the same idea to let people control a cursor or a robotic arm.

This project builds that pipeline step by step:

1. Load and clean the neural and movement recordings, and line them up in time.
2. Confirm that individual neurons carry information about reach direction.
3. Train decoders that predict **hand velocity** from population activity, starting simple (linear regression) and working up (Kalman filter, recurrent neural network).

---

## Dataset

| | |
| --- | --- |
| Source | [DANDI Archive, Dandiset 000140](https://dandiarchive.org/dandiset/000140) (MC_Maze_Small, Neural Latents Benchmark) |
| Subject | Monkey "Jenkins" (rhesus macaque) |
| Task | Reaching to targets on a screen after an instructed delay, sometimes around virtual barriers ("maze") |
| Neural data | Spiking activity of **142 neurons** recorded in primary motor cortex (M1) and dorsal premotor cortex (PMd) |
| Behavioral data | Hand position and velocity, cursor position, eye position (x and y for each) |
| Trials used | **100** (the training file, with `split_heldout=False`) |
| Raw resolution | 1 ms (293,666 time points × 150 columns: 142 neurons + 8 behavioral signals) |

The raw data is **not** stored in this repository (`data/` and `*.nwb` are in `.gitignore`). See [Setup](#setup) to download it.

---

## Repository structure

```
motor-cortex-neural-decoding/
├── data/                  # DANDI download goes here (ignored by Git)
├── figures/               # Figures saved by the notebooks
│   ├── 01_single_trial.png
│   ├── 02_hand_paths.png
│   ├── 03_psths.png
│   └── 04_tuning.png
├── notebooks/
│   └── 01_explore.ipynb   # Phase 1: load, clean, align, explore
├── src/                   # Reusable code (added in later phases)
├── requirements.txt
└── README.md
```

---

## Setup

Tested with Python 3.11 in a conda environment on Windows (VS Code).

**1. Create the environment and install packages**

```bash
conda create -n neurodecode python=3.11 -y
conda activate neurodecode
pip install -r requirements.txt
pip install --no-deps nlb_tools
```

`nlb_tools` was written for older versions of pandas and NumPy, so `requirements.txt` pins `pandas==1.5.3` and `numpy<2`. Installing `nlb_tools` with `--no-deps` stops it from replacing those pinned versions.

**2. Download the data** (run from the repository root)

```bash
cd data
dandi download DANDI:000140
cd ..
```

This creates `data/000140/sub-Jenkins/`, which the notebook reads.

**3. Run the notebook**

Open `notebooks/01_explore.ipynb` in VS Code or Jupyter, select the `neurodecode` kernel, and run all cells. Figures are saved to `figures/`.

---

## Phase 1: what has been done

### 1. Load the data

`nlb_tools.NWBDataset` reads the NWB file and combines spikes and behavior into one pandas table, indexed by time at 1 ms resolution, plus a `trial_info` table with one row per trial (start and end times, movement onset, number of barriers, and so on).

### 2. Fix the time index

The combined time index was not in clean, evenly spaced order. It jumped backward by about 290 seconds at one point and had gaps of 15–25 ms, so pandas could not infer a regular sampling rate and `resample()` failed with `Inferred frequency None ... does not conform to passed frequency`.

The fix rounds every timestamp to the nearest millisecond and groups rows by timestamp, which also sorts them:

```python
dataset.data.index = dataset.data.index.round("ms")
dataset.data = dataset.data.groupby(level=0).first()
dataset.data.index.name = "clock_time"
```

Afterward, every step between rows is exactly 1 ms, and no rows were lost (293,666 before and after).

### 3. Bin and smooth the spikes

- **Binning:** spike counts are summed into 20 ms bins (`dataset.resample(20)`). Decoders typically work at this resolution.
- **Smoothing:** a smoothed copy of the spikes (`spikes_smth`) is made with a 40 ms Gaussian kernel (`dataset.smooth_spk(40)`). It is used only for plotting average firing rates.

### 4. Align trials to movement onset

Each trial is cut to a 700 ms window, from 250 ms before to 450 ms after the hand starts moving (`make_trial_data(align_field="move_onset_time", align_range=(-250, 450))`). Lining trials up on the same event lets them be averaged.

### 5. Measure reach direction

For each trial, reach direction is the angle of the hand's net displacement across the window, computed with `np.arctan2(dy, dx)`. Angles are in the 2D plane of the screen: 0° is rightward, 90° is upward, and ±180° is leftward. For plotting, trials are grouped into eight 45° direction bins. Only five of the eight bins contain trials in this dataset (centered at −158°, −22°, 22°, 112°, and 158°).

### 6. Explore with figures

**Figure 1: one trial's spikes** (`figures/01_single_trial.png`)

![Spike counts on one trial](figures/01_single_trial.png)

Spike counts for all 142 neurons on a single trial, in 20 ms bins; the red line marks movement onset. Activity is spread thinly across the population, which is why decoders combine many neurons instead of relying on any single one.

**Figure 2: hand paths** (`figures/02_hand_paths.png`)

![Hand paths colored by reach direction](figures/02_hand_paths.png)

The hand's path on each of the 100 trials, colored by reach direction. Curved paths are reaches around the maze barriers.

**Figure 3: average firing rates by direction (PSTHs)** (`figures/03_psths.png`)

Each panel shows one neuron's average firing rate over time, one line per reach direction (a peri-stimulus time histogram, or PSTH). The four neurons shown are those whose average rate changes most across directions and time (largest modulation depth). For example, neuron 1651 peaks near 39 Hz shortly after movement onset for 112° reaches but stays below about 7 Hz for ±22° reaches. Neurons 2232 and 1072 reach their peak before the hand starts moving.

**Figure 4: directional tuning** (`figures/04_tuning.png`)

![Directional tuning of neuron 1651](figures/04_tuning.png)

Neuron 1651's average firing rate during the first 300 ms of movement on each trial, plotted against reach direction. It fires at roughly 30–67 Hz on many reaches near 120–140° and at 0–10 Hz on reaches between −30° and 45°. This **directional tuning** is the signal a decoder learns to read.

---

## Key terms

- **Spike count / binning:** the number of times a neuron fires within a time window (here, 20 ms).
- **Firing rate (Hz):** spikes per second, computed as the spike count divided by the window length.
- **PSTH:** a neuron's firing rate over time, averaged across trials aligned to the same event.
- **Modulation depth:** how much a neuron's average firing rate varies across conditions and time (maximum minus minimum).
- **Directional tuning:** a neuron firing more for some movement directions than others.
- **Decoder:** a model that predicts behavior (here, hand velocity) from neural activity.

---

## Roadmap

- [x] **Phase 1:** load, clean, align, and explore the data (Sep 2026)
- [ ] **Phase 2:** ridge regression baseline predicting hand velocity from binned spikes, scored with R² on held-out trials
- [ ] **Phase 3:** Kalman filter decoder
- [ ] **Phase 4:** recurrent neural network decoder
- [ ] **Later:** decode behavior from mouse Neuropixels recordings, using pose-tracking (SLEAP) and behavior-segmentation (Keypoint MoSeq) outputs

---

## Data and tools

- Data: [DANDI Archive, Dandiset 000140](https://dandiarchive.org/dandiset/000140). See the Dandiset page for the original data citation.
- Benchmark and loading code: [Neural Latents Benchmark, `nlb_tools`](https://github.com/neurallatents/nlb_tools); Pei et al. (2021), *Neural Latents Benchmark '21*, [arXiv:2109.04463](https://arxiv.org/abs/2109.04463).
- Python libraries: pandas, NumPy, matplotlib, scikit-learn, pynwb, dandi.

---

## Author

**James Abednego Delinois** · [LinkedIn](https://www.linkedin.com/in/james-delinois-619a481b7) · [GitHub](https://github.com/jamesdelinois)
