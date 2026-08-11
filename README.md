# YadYar Lite — Phase 1: T-19 Text Difficulty Assessment

**Team:** Hannaneh Ahmadi _ Sana Bavari 
**Course:** Artificial Intelligence (Spring 1404-1405)  
**Instructor:** Dr. Koohzadi  

---

## Project Overview
This is the first phase of the YadYar Lite project for the **T-19** topic:  
**Text Difficulty Assessment Across Domains**.

We build a baseline model using **TF-IDF + Logistic Regression** to classify texts into three difficulty levels:
- Elementary (0)
- Intermediate (1)
- Advanced (2)

---

## Dataset
- **Source:** OneStopEnglishCorpus (`texts.zip`)
- **Samples:** 567 (189 articles × 3 difficulty levels)
- **Split:** GroupShuffleSplit by article (no data leakage)
  - Train: 151 articles (453 samples)
  - Test: 38 articles (114 samples)

---

## Model
- **Feature Extraction:** TF-IDF (`ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`)
- **Classifier:** Logistic Regression (`C=2.0`, `max_iter=2000`)

---

## Visualizations
All plots are saved in the `phase1_outputs/` folder:
- `length_distribution.png` — Text length distribution by class
- `confusion_matrix_heatmap.png` — Confusion matrix heatmap
- `per_class_performance.png` — Precision, Recall, F1 per class

---

## How to Run
1. Make sure `texts.zip` is in the project root.
2. Install dependencies:
   pip install -r requirements.txt
   Run the script:
3. Run the script
   python phase1.py
4. All outputs will be saved in the phase1_outputs/ folder.

