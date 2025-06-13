# 🫀 Pan-Tompkins QRS Detection with LMS Adaptive Threshold

This project implements the classic Pan-Tompkins algorithm for QRS detection in ECG signals and enhances it with an LMS-based adaptive thresholding mechanism to improve performance, especially under noisy conditions.

---

## 📌 Project Summary

The goal is to compare:
- **Static thresholding** from the original Pan-Tompkins algorithm
- **LMS-based adaptive thresholding**, which dynamically adjusts based on the ECG signal envelope

Both methods are evaluated on real ECG records from the MIT-BIH Arrhythmia Database.

---

## 📊 Evaluation Results

| Dataset       | Method  | TP   | FP  | FN  | Sensitivity | Precision | F1 Score |
|---------------|---------|------|-----|-----|-------------|-----------|----------|
| Clean (101)   | Static  | 1866 |  2  |  8  | 0.996       | 0.999     | 0.997    |
| Clean (101)   | LMS     | 1865 |  2  |  9  | 0.995       | 0.999     | 0.997    |
| Noisy (108)   | Static  | 1748 | 191 | 76  | 0.958       | 0.901     | 0.929    |
| Noisy (108)   | LMS     | 1746 |  53 | 78  | 0.957       | 0.971     | **0.964**|

> LMS-based thresholding performs comparably to static methods in clean signals and significantly better in noisy recordings by reducing false positives.

---

## 🚀 How to Run

### 1. Install dependencies

Make sure you have Python 3 and the following libraries:

```bash
pip install -r requirements.txt
```

### 2. Prepare ECG data

Download records `101` and `108` from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) and place them here:

```
data/
├── clean_signals/101.hea, 101.dat, 101.atr
└── noisy_signals/108.hea, 108.dat, 108.atr
```

### 3. Run the script

```bash
python Pan_Tompkins_QRS.py
```

---

## 📈 Features

- 🧠 Implements full Pan-Tompkins signal flow:
  - Bandpass filter (LP + HP)
  - Differentiation
  - Squaring
  - Moving window integration
  - Static thresholding
  -  Adaptive LMS threshold using signal envelope
  -  DSP analysis of filters (magnitude, phase, pole-zero, group delay)
  -  Evaluation vs annotated QRS beats
  -  Visual comparison of thresholds and QRS detections

---

## 📚 References

- Pan, J., & Tompkins, W. J. (1985). *A Real-Time QRS Detection Algorithm*. IEEE Transactions on Biomedical Engineering, 32(3), 230–236.
- MIT-BIH Arrhythmia Database: [PhysioNet MITDB](https://physionet.org/content/mitdb/1.0.0/)

