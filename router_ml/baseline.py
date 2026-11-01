import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from router_ml import data, metrics

def main() -> None:
    ds = data.load()
    names = data.label_names(ds)
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )
    model = make_pipeline(features, LogisticRegression(C=20, max_iter=2000))
    start = time.perf_counter()
    model.fit(ds["train"]["text"], ds["train"]["label"])
    train_s = time.perf_counter() - start
    val = model.score(ds["validation"]["text"], ds["validation"]["label"])
    probs = model.predict_proba(ds["test"]["text"])
    card = metrics.scorecard(ds["test"]["label"], probs, names)
    val = model.score(ds["validation"]["text"], ds["validation"]["label"])
    probs = model.predict_proba(ds["test"]["text"])
    card = metrics.scorecard(ds["test"]["label"], probs, names)
    metrics.save("tfidf_logreg", card, val_accuracy=round(val, 4), train_s=train_s)

if __name__ == "__main__":
    main()