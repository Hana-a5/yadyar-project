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
1. Download OneStopEnglishCorpus from:
(https://github.com/nishkalavallabhi/OneStopEnglishCorpus/tree/master)
2. Extract texts.zip
3. Put texts.zip in the project root
4. Install dependencies:
   pip install -r requirements.txt
5. Run:
   python baseline.py
Team 14 - T-19 Text Difficulty Assessment (Phase 2)

‫نحوه اجرا:
‫1. فایل texts.zip را در کنار فایل Phase2_Code.ipynb قرار دهید.
‫2. فایل Phase2_Code.ipynb را در Jupyter Notebook یا Google Colab باز کنید.
‫3. تمام سلول‌ها را به ترتیب از بالا به پایین اجرا کنید (Run All Cells).

‫پیش‌نیازها (کتابخانه‌های مورد نیاز):
‫- pandas
‫- numpy
‫- matplotlib
‫- seaborn
‫- scikit-learn

‫نکته: تمام کتابخانه‌های بالا با دستور pip قابل نصب هستند.

‫خروجی‌های تولیدشده توسط کد:
‫- confusion_matrix.png (تصویر ماتریس درهم‌ریختگی)
‫- accuracy_by_length.png (تصویر نمودار دقت بر اساس طول متن)
‫- phase2_results.csv (نتایج اصلی: Accuracy, Macro-F1, MAE)
‫- phase2_test_predictions.csv (پیش‌بینی‌های مدل روی داده‌های تست)
‫- phase2_error_analysis.csv (تحلیل خطاها)
‫- phase2_short_long_results.csv (نتایج تحلیل Short/Long)

‫توضیح دیتاست:
‫دیتاست OneStopEnglish از فایل texts.zip بارگذاری می‌شود که شامل 189 فایل CSV است.
