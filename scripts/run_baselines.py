import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

def load_data(path):
    texts, bin_labels, cat_labels = [], [], []
    cat_to_id = {"Gender": 0, "Political": 1, "Religion": 2, "Geo-political": 3, "Violence": 4, "Others": 5}
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["Comment"])
            bl = 1 if d["Label"] == "Hate" else 0
            bin_labels.append(bl)
            
            c = d.get("Hate Category", "")
            cl = cat_to_id.get(c, -1)
            cat_labels.append(cl)
    return texts, bin_labels, cat_labels

def main():
    print("Running Baselines...")
    tr_txt, tr_bin, tr_cat = load_data('data/splits/train.jsonl')
    te_txt, te_bin, te_cat = load_data('data/splits/test.jsonl')
    
    # Majority Binary
    # Train distribution is roughly 50/50, let's find exact majority
    maj_bin = 1 if sum(tr_bin) > len(tr_bin)/2 else 0
    maj_bin_preds = [maj_bin] * len(te_bin)
    maj_bin_f1 = f1_score(te_bin, maj_bin_preds, average='macro')
    print(f"Majority Binary Macro-F1: {maj_bin_f1:.4f}")
    
    # Majority Category (only for True Hate)
    hate_idx = [i for i, b in enumerate(tr_bin) if b == 1]
    tr_hate_cats = [tr_cat[i] for i in hate_idx]
    
    # Find majority category
    cat_counts = {}
    for c in tr_hate_cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    maj_cat = max(cat_counts, key=cat_counts.get)
    
    te_hate_idx = [i for i, b in enumerate(te_bin) if b == 1]
    te_hate_cats = [te_cat[i] for i in te_hate_idx]
    maj_cat_preds = [maj_cat] * len(te_hate_idx)
    maj_cat_f1 = f1_score(te_hate_cats, maj_cat_preds, average='macro')
    print(f"Majority Category Macro-F1: {maj_cat_f1:.4f}")
    
    # TF-IDF + Logistic Regression (Binary)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_tr = vectorizer.fit_transform(tr_txt)
    X_te = vectorizer.transform(te_txt)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, tr_bin)
    lr_preds = clf.predict(X_te)
    lr_f1 = f1_score(te_bin, lr_preds, average='macro')
    print(f"TF-IDF+LR Binary Macro-F1: {lr_f1:.4f}")

    results = {
        "majority_binary_macro_f1": maj_bin_f1,
        "majority_category_macro_f1": maj_cat_f1,
        "tfidf_lr_binary_macro_f1": lr_f1
    }
    
    import os
    os.makedirs('reports', exist_ok=True)
    with open('reports/m4_baselines.json', 'w') as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main()
