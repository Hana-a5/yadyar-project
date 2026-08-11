import zipfile
import io
import os
import colorama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init, Fore, Back, Style
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)
colorama.init(autoreset=True)

# SETTINGS
ZIP_PATH = "texts.zip"
RANDOM_STATE = 42
OUTPUT_DIR = "phase1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = {
    "Elementary": 0,
    "Intermediate": 1,
    "Advanced": 2
}


# READ DATASET
def load_dataset(zip_path):
    records = []

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        csv_files = [
            name for name in zip_file.namelist()
            if name.lower().endswith(".csv")
        ]

        print("Number of CSV files:", len(csv_files))

        for filename in csv_files:
            file_data = zip_file.read(filename)
            df = pd.read_csv(
                io.BytesIO(file_data),
                encoding="cp1252"
            )

            df.columns = [
                str(column).strip()
                for column in df.columns
            ]

            article_name = filename.rsplit(".", 1)[0]

            for level in LABELS:
                if level not in df.columns:
                    continue

                paragraphs = (
                    df[level]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                text = " ".join(paragraphs)
                text = " ".join(text.split())

                if text != "":
                    records.append({
                        "article": article_name,
                        "level": level,
                        "text": text
                    })

    return pd.DataFrame(records)


# LOAD DATA
print(colorama.Fore.YELLOW + "---> LOADING DATASET\n" + Style.RESET_ALL)
data = load_dataset(ZIP_PATH)

# BASIC INFORMATION
print("\nDataset shape:")
print(data.shape)

# CREATE LABEL
data["label"] = data["level"].map(LABELS)

# DATASET STATISTICS
print(colorama.Fore.YELLOW + "\n---> DATASET STATISTICS\n" + Style.RESET_ALL)

print("Number of samples:", len(data))
print("Number of articles:", data["article"].nunique())

print("\nClass distribution:")
print(data["level"].value_counts())

# TEXT LENGTH ANALYSIS
data["word_count"] = (
    data["text"]
    .str.split()
    .str.len()
)

print(colorama.Fore.YELLOW + "\n---> TEXT LENGTH\n" + Style.RESET_ALL)

length_stats = (
    data
    .groupby("level")["word_count"]
    .agg(["count", "mean", "median", "min", "max"])
)
print(length_stats)

# PREPARE DATA
X = data["text"]
y = data["label"]
groups = data["article"]

# TRAIN / TEST SPLIT
print(colorama.Fore.YELLOW + "\n---> TRAIN / TEST SPLIT\n" + Style.RESET_ALL)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_index, test_index = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]
y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

train_articles = groups.iloc[train_index].unique()
test_articles = groups.iloc[test_index].unique()

print("Training articles:", len(train_articles))
print("Test articles:", len(test_articles))
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# CHECK DATA LEAKAGE
common_articles = set(train_articles).intersection(set(test_articles))
print("\nCommon articles:", len(common_articles))

if len(common_articles) == 0:
    print("No data leakage detected.")
else:
    print("WARNING: Data leakage detected!")

# BUILD BASELINE MODEL
print(colorama.Fore.YELLOW + "\n---> BUILDING BASELINE MODEL\n" + Style.RESET_ALL)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=30000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            C=2.0,
            max_iter=2000
        )
    )
])

# TRAIN MODEL
print("Training...")
model.fit(X_train, y_train)
print("Training completed.")

# PREDICTION
y_pred = model.predict(X_test)

# EVALUATION
print(colorama.Fore.YELLOW + "\n--->BASELINE RESULTS\n" + Style.RESET_ALL)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
mae = mean_absolute_error(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))
print("MAE:", round(mae, 4))

# CLASSIFICATION REPORT
print(colorama.Fore.YELLOW + "\n---> CLASSIFICATION REPORT\n" + Style.RESET_ALL)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Elementary", "Intermediate", "Advanced"]
    )
)

# 16. CONFUSION MATRIX
print(colorama.Fore.YELLOW + "\n---> CONFUSION MATRIX\n" + Style.RESET_ALL)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Elementary", "Actual Intermediate", "Actual Advanced"],
    columns=["Pred Elementary", "Pred Intermediate", "Pred Advanced"]
)
print(cm_df)

# SAVE BASELINE RESULTS
results = pd.DataFrame({
    "Metric": ["Accuracy", "Macro-F1", "MAE"],
    "Value": [accuracy, macro_f1, mae]
})
results.to_csv(
    os.path.join(OUTPUT_DIR, "phase1_results.csv"),
    index=False,
    encoding="utf-8-sig"
)
print("\nResults saved to phase1_outputs/phase1_results.csv")

# SAVE TEST PREDICTIONS
test_data = data.iloc[test_index].copy()
test_data["predicted_label"] = y_pred
test_data["predicted_level"] = (
    test_data["predicted_label"]
    .map({0: "Elementary", 1: "Intermediate", 2: "Advanced"})
)
test_data["correct"] = (
    test_data["label"] == test_data["predicted_label"]
)
test_data.to_csv(
    os.path.join(OUTPUT_DIR, "phase1_test_predictions.csv"),
    index=False,
    encoding="utf-8-sig"
)
print("Test predictions saved to phase1_outputs/phase1_test_predictions.csv")

# NEW: COEFFICIENT ANALYSIS
print(Fore.YELLOW + "\n---> COEFFICIENT ANALYSIS (Top Words per Class)\n" + Style.RESET_ALL)

feature_names = model.named_steps["tfidf"].get_feature_names_out()
coefs = model.named_steps["classifier"].coef_
class_names = ["Elementary", "Intermediate", "Advanced"]
class_colors = {
    "Elementary": Fore.LIGHTBLUE_EX,
    "Intermediate": Fore.LIGHTRED_EX,          # یا Fore.LIGHTYELLOW_EX
    "Advanced": Fore.LIGHTGREEN_EX
}

for i, class_name in enumerate(class_names):
    coef = coefs[i]

    top_positive_idx = np.argsort(coef)[-10:][::-1]
    top_positive_words = [feature_names[idx] for idx in top_positive_idx]
    top_positive_scores = [coef[idx] for idx in top_positive_idx]

    top_negative_idx = np.argsort(coef)[:10]
    top_negative_words = [feature_names[idx] for idx in top_negative_idx]
    top_negative_scores = [coef[idx] for idx in top_negative_idx]

    color = class_colors[class_name]
    print(color + f"\n--- {class_name} ---" + Style.RESET_ALL)
    print(color + "Top positive words (strongly associated):" + Style.RESET_ALL)
    for word, score in zip(top_positive_words, top_positive_scores):
        print(color + f"  {word}: {score:.4f}" + Style.RESET_ALL)

    print(color + "Top negative words (weakly associated):" + Style.RESET_ALL)
    for word, score in zip(top_negative_words, top_negative_scores):
        print(color + f"  {word}: {score:.4f}" + Style.RESET_ALL)

coef_df = pd.DataFrame(
    coefs.T,
    index=feature_names,
    columns=class_names
)
coef_df.to_csv(
    os.path.join(OUTPUT_DIR, "coefficients.csv"),
    encoding="utf-8-sig"
)
print("\nCoefficients saved to phase1_outputs/coefficients.csv")

# NEW: VISUALIZATIONS
print(colorama.Fore.YELLOW + "\n---> GENERATING VISUALIZATIONS\n" + Style.RESET_ALL)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


plt.figure()
for i, level in enumerate(["Elementary", "Intermediate", "Advanced"]):
    subset = data[data["level"] == level]
    sns.histplot(
        subset["word_count"],
        label=level,
        alpha=0.5,
        kde=True
    )
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.title("Text Length Distribution by Difficulty Level")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "length_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print("  - length_distribution.png saved")

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Elementary", "Intermediate", "Advanced"],
    yticklabels=["Elementary", "Intermediate", "Advanced"],
    cbar=True
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Baseline Model")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print("  - confusion_matrix_heatmap.png saved")

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=["Elementary", "Intermediate", "Advanced"],
    output_dict=True
)

metrics_df = pd.DataFrame({
    "Class": ["Elementary", "Intermediate", "Advanced"],
    "Precision": [
        report_dict["Elementary"]["precision"],
        report_dict["Intermediate"]["precision"],
        report_dict["Advanced"]["precision"]
    ],
    "Recall": [
        report_dict["Elementary"]["recall"],
        report_dict["Intermediate"]["recall"],
        report_dict["Advanced"]["recall"]
    ],
    "F1": [
        report_dict["Elementary"]["f1-score"],
        report_dict["Intermediate"]["f1-score"],
        report_dict["Advanced"]["f1-score"]
    ]
})

metrics_melted = metrics_df.melt(
    id_vars="Class",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=metrics_melted,
    x="Class",
    y="Score",
    hue="Metric",
    palette="Set2"
)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Per-Class Performance Comparison")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "per_class_performance.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()
print("  - per_class_performance.png saved")

print(f"\nAll outputs saved in: {OUTPUT_DIR}")
print(colorama.Fore.MAGENTA+"End!" + Style.RESET_ALL)