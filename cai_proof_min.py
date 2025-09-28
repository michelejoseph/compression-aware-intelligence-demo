
# cai minimal "proof" kit - michele style
# goal: show compression -> info loss -> task perf drops. isr tracks perf.
# notes: keep it simple. fast. reproducible. comments in lowercase (lol).

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# seed so results arent all over the place
np.random.seed(42)

def run():
    # output folder
    out = Path("./cai_proof_self")
    out.mkdir(parents=True, exist_ok=True)

    # 1) synth data (binary clf, enough signal but not too easy)
    X, y = make_classification(
        n_samples=6000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        class_sep=1.25,
        flip_y=0.01,
        random_state=42,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # standardize (dont forget this or pca/logreg get weird)
    ss = StandardScaler()
    X_trs = ss.fit_transform(X_tr)
    X_tes = ss.transform(X_te)

    # baseline on full dim (aka "no compression")
    base = LogisticRegression(max_iter=2000)
    base.fit(X_trs, y_tr)
    yb = base.predict(X_tes)
    acc_base = accuracy_score(y_te, yb)
    f1_base = f1_score(y_te, yb)

    # avg var for normalization (unitless nmse)
    data_var = np.mean(np.var(X_trs, axis=0))

    rows = []
    comps = [50, 40, 30, 20, 15, 10, 8, 6, 4, 3, 2, 1]  # include full 50 to have a clean top line

    for k in comps:
        # compress
        p = PCA(n_components=k, random_state=42)
        Z_tr = p.fit_transform(X_trs)
        Z_te = p.transform(X_tes)

        # classify in compressed space
        c = LogisticRegression(max_iter=2000)
        c.fit(Z_tr, y_tr)
        yp = c.predict(Z_te)

        acc = accuracy_score(y_te, yp)
        f1 = f1_score(y_te, yp)

        # reconstruct to estimate info loss
        Xh_tr = p.inverse_transform(Z_tr)
        Xh_te = p.inverse_transform(Z_te)

        mse_te = np.mean((X_tes - Xh_te) ** 2)
        nmse_te = float(mse_te / data_var)

        # info sufficiency ratio (simple version)
        isr = 1.0 - nmse_te

        # little print so ppl see whats going on
        print(f"k={k:02d}/{X_trs.shape[1]}  acc={acc:.3f}  f1={f1:.3f}  isr={isr:.3f}")

        rows.append({
            "n_components": k,
            "compression_ratio": k / X_trs.shape[1],
            "accuracy": acc,
            "f1": f1,
            "nmse_test": nmse_te,
            "ISR": isr,
        })

    # results table
    df = pd.DataFrame(rows).sort_values("n_components", ascending=False).reset_index(drop=True)

    # sanity: baseline row (from full dim calc above). keep isr=1 at top.
    if df.loc[0, "n_components"] != X_trs.shape[1]:
        top = pd.DataFrame([{
            "n_components": X_trs.shape[1],
            "compression_ratio": 1.0,
            "accuracy": acc_base,
            "f1": f1_base,
            "nmse_test": 0.0,
            "ISR": 1.0,
        }])
        df = pd.concat([top, df], ignore_index=True)

    # save csv
    csv_path = out / "results.csv"
    df.to_csv(csv_path, index=False)

    # plots (no special colors/styles)
    plt.figure(figsize=(7,5))
    plt.plot(df["n_components"], df["accuracy"], marker="o")
    plt.title("accuracy vs pca components")
    plt.xlabel("pca components")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.3)
    fig1 = out / "accuracy_vs_components.png"
    plt.savefig(fig1, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(df["ISR"], df["accuracy"], marker="o")
    plt.title("accuracy tracks isr")
    plt.xlabel("isr = 1 - normalized mse")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.3)
    fig2 = out / "accuracy_vs_isr.png"
    plt.savefig(fig2, bbox_inches="tight")
    plt.close()

    # quick corr (yeah pearson is fine)
    r = float(np.corrcoef(df["ISR"], df["accuracy"])[0,1])

    # tiny readme (keep it casual)
    readme = f"""
# cai mini proof (quick + dirty)

what it shows:
- more compression -> higher recon loss -> lower clf accuracy.
- isr (1 - nmse) tracks accuracy. on this run r ≈ {r:.3f}.

how to run:
- `python cai_proof_min.py` from this folder. it prints lines and saves png/csv under `cai_proof_self/`.

files:
- results.csv
- accuracy_vs_components.png
- accuracy_vs_isr.png

notes:
- data is synthetic (50 feats, 30 informative). pca used as compressor. swap in whatever and check if isr still tracks.
- baseline row (no compression) sits at isr=1.0 by design.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    # final tiny summary print (so ppl can screenshot)
    last = df.tail(1).to_dict(orient="records")[0]
    print("---")
    print(f"baseline_acc={acc_base:.3f} | best_acc={df['accuracy'].max():.3f} | corr(isr,acc)={r:.3f}")
    print(f"last_run: k={int(last['n_components'])} acc={last['accuracy']:.3f} isr={last['ISR']:.3f}")
    print(f"saved -> {csv_path.name}, {fig1.name}, {fig2.name}")

if __name__ == "__main__":
    run()
