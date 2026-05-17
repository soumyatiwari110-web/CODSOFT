# ============================================================
# TITANIC SURVIVAL PREDICTION - MODEL TRAINING
# Author: Soumya Tiwari
# Description: Train, compare, and save the best ML model
# ============================================================

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score)
from preprocess import load_and_preprocess

# ----------------------------
# LOAD PREPROCESSED DATA
# ----------------------------
X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess()

# ----------------------------
# DEFINE ALL MODELS
# ----------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200,
                                                   max_depth=8,
                                                   min_samples_split=5,
                                                   random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=7)
}

# ----------------------------
# TRAIN AND EVALUATE ALL MODELS
# ----------------------------
results = {}
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]
                        if hasattr(model, 'predict_proba')
                        else model.decision_function(X_test))
    results[name] = {'model': model, 'accuracy': acc, 'auc': auc, 'predictions': y_pred}
    print(f"\n📊 {name}")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   AUC Score: {auc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# ----------------------------
# SELECT BEST MODEL
# ----------------------------
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']
best_acc = results[best_name]['accuracy']

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_name}")
print(f"   Accuracy: {best_acc*100:.2f}%")
print("="*60)

# ----------------------------
# CONFUSION MATRIX FOR BEST MODEL
# ----------------------------
plt.style.use('dark_background')
GOLD = '#C9A84C'
NAVY = '#0D1B2A'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(NAVY)
fig.suptitle(f'Model Evaluation — {best_name}\nBy Soumya Tiwari',
             color=GOLD, fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, results[best_name]['predictions'])
ax1 = axes[0]
ax1.set_facecolor(NAVY)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'],
            ax=ax1, linewidths=1, linecolor=NAVY,
            annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_title('Confusion Matrix', color=GOLD, fontsize=12, fontweight='bold')
ax1.set_xlabel('Predicted', color='white')
ax1.set_ylabel('Actual', color='white')
ax1.tick_params(colors='white')

# Model Accuracy Comparison
ax2 = axes[1]
ax2.set_facecolor(NAVY)
names = list(results.keys())
accs = [results[n]['accuracy'] * 100 for n in names]
colors = ['#C9A84C' if n == best_name else '#4A90E2' for n in names]
bars = ax2.barh(names, accs, color=colors, edgecolor='white', linewidth=0.8)
ax2.set_title('Model Accuracy Comparison', color=GOLD, fontsize=12, fontweight='bold')
ax2.set_xlabel('Accuracy (%)', color='white')
ax2.tick_params(colors='white')
ax2.set_xlim(70, 100)
for bar, acc in zip(bars, accs):
    ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             f'{acc:.2f}%', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('static/images/model_evaluation.png', dpi=150,
            bbox_inches='tight', facecolor=NAVY)
plt.show()

# ----------------------------
# FEATURE IMPORTANCE (Random Forest)
# ----------------------------
if best_name == 'Random Forest':
    fi = best_model.feature_importances_
    fi_df = dict(zip(feature_cols, fi))
    fi_sorted = dict(sorted(fi_df.items(), key=lambda x: x[1], reverse=True))

    fig2, ax = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.barh(list(fi_sorted.keys()), list(fi_sorted.values()),
            color=GOLD, edgecolor='white', linewidth=0.8)
    ax.set_title('Feature Importance — Random Forest', color=GOLD,
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=150,
                bbox_inches='tight', facecolor=NAVY)
    plt.show()

# ----------------------------
# SAVE BEST MODEL WITH PICKLE
# Why: So Flask app can load and reuse it without retraining
# ----------------------------
with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n✅ Model saved to models/model.pkl")
print(f"   Model type: {best_name}")
print(f"   Accuracy  : {best_acc*100:.2f}%")
print("\n🚀 Ready for Flask deployment!")