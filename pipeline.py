#!/usr/bin/env python3
"""
Kasus 07 · Irigasi Pintar Berbasis IoT (Rumah Kaca Sayuran Lembang)
Pipeline Pembersihan Data, Rekayasa Fitur Time-Series, Evaluasi Baseline,
Pelatihan Model GBDT & Regresi Logistik (Time-Based Split), dan Analisis Biaya.
"""

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


# ==============================================================================
# 1. PEMBERSIHAN DATA & PEMULIHAN ANOMALI SENSOR (CAUSAL / TANPA LEAKAGE)
# ==============================================================================
def load_and_clean_data(raw_csv_path: str) -> tuple[pd.DataFrame, dict]:
    df_raw = pd.read_csv(raw_csv_path)
    audit = {
        "raw_rows": len(df_raw),
        "exact_duplicates": int(df_raw.duplicated().sum()),
    }

    # Hapus duplikat baris identik dan urutkan secara kronologis per petak
    df = df_raw.drop_duplicates().copy()
    df["waktu_dt"] = pd.to_datetime(df["waktu"])
    df = df.sort_values(["petak", "waktu_dt"]).reset_index(drop=True)

    audit["unique_rows"] = len(df)
    audit["missing_hourly_slots"] = 2208 * 16 - len(df)
    audit["dirty_tanaman_count"] = int((~df["tanaman"].isin(["cabai", "tomat", "selada"])).sum())
    audit["kt_glitch_0"] = int((df["kelembaban_tanah_pct"] == 0.0).sum())
    audit["kt_glitch_100"] = int((df["kelembaban_tanah_pct"] == 100.0).sum())
    audit["kt_glitch_127_5"] = int((df["kelembaban_tanah_pct"] == 127.5).sum())
    audit["cahaya_lux_negative"] = int((df["cahaya_lux"] < 0).sum())
    audit["suhu_missing"] = int(df["suhu_udara_C"].isna().sum())
    audit["prakiraan_missing"] = int(df["prakiraan_hujan_6jam_mm"].isna().sum())

    # Simpan nilai kelembaban_tanah_pct mentah untuk evaluasi Baseline Mentah
    df["kt_raw"] = df["kelembaban_tanah_pct"]

    # 1. Normalisasi nama tanaman ('CABAI', 'Cabai', 'cabe', 'tomat ', 'Tomat', 'Selada', 'lettuce')
    df["tanaman"] = (
        df["tanaman"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"cabe": "cabai", "lettuce": "selada"})
    )

    # 2. Perbaikan bug inversi bit two's complement pada cahaya_lux negatif (-x - 1)
    df["cahaya_lux"] = np.where(df["cahaya_lux"] < 0, -df["cahaya_lux"] - 1.0, df["cahaya_lux"])

    # 3. Imputasi prakiraan_hujan_6jam_mm dan suhu_udara_C dari petak lain di jam yang sama
    prak_by_time = df.groupby("waktu")["prakiraan_hujan_6jam_mm"].median()
    df["prakiraan_hujan_6jam_mm"] = df["prakiraan_hujan_6jam_mm"].fillna(df["waktu"].map(prak_by_time))

    suhu_by_time = df.groupby("waktu")["suhu_udara_C"].mean()
    df["suhu_udara_C"] = df["suhu_udara_C"].fillna(df["waktu"].map(suhu_by_time)).round(1)

    # 4. Pemulihan kausal (forward-fill per petak) untuk glitch sensor kelembaban_tanah_pct (0.0, 100.0, 127.5)
    df["kelembaban_tanah_pct"] = df["kelembaban_tanah_pct"].replace({0.0: np.nan, 100.0: np.nan, 127.5: np.nan})
    df["kelembaban_tanah_pct"] = df.groupby("petak")["kelembaban_tanah_pct"].ffill().bfill()

    # 5. Rekayasa fitur time-series kausal (hanya menggunakan jam t-1 dan t-2)
    df["kt_prev1"] = df.groupby("petak")["kelembaban_tanah_pct"].shift(1).bfill()
    df["kt_prev2"] = df.groupby("petak")["kelembaban_tanah_pct"].shift(2).bfill()
    df["kt_jump1"] = (df["kelembaban_tanah_pct"] - df["kt_prev1"]).round(1)
    df["drying_rate_prev"] = (df["kt_prev1"] - df["kt_prev2"]).clip(-2.0, 0.0).fillna(-0.5)

    # Rekonsiliasi kelembapan pra-siram: jika kt_jump1 > 10.0, sensor mencatat angka setelah penyiraman
    df["kt_efektif"] = np.where(
        df["kt_jump1"] > 10.0,
        (df["kt_prev1"] + df["drying_rate_prev"]).round(1),
        df["kelembaban_tanah_pct"],
    )

    df["hour"] = df["waktu_dt"].dt.hour
    df["is_daytime"] = df["hour"].between(5, 17).astype(float)

    crop_th_map = {"cabai": 32.0, "tomat": 36.0, "selada": 42.0}
    df["ambang_tanaman"] = df["tanaman"].map(crop_th_map)
    df["defisit_kt_raw"] = (df["ambang_tanaman"] - df["kelembaban_tanah_pct"]).round(2)
    df["defisit_kt_efektif"] = (df["ambang_tanaman"] - df["kt_efektif"]).round(2)

    # One-hot encoding tanaman dan jenis tanah
    for c in ["cabai", "tomat", "selada"]:
        df[f"tanaman_{c}"] = (df["tanaman"] == c).astype(float)
    for s in ["pasir", "lempung", "liat"]:
        df[f"tanah_{s}"] = (df["jenis_tanah"] == s).astype(float)

    return df, audit


# ==============================================================================
# 2. IMPLEMENTASI MODEL GBDT & REGRESI LOGISTIK (MURNI NUMPY/SCIPY)
# ==============================================================================
class RegressionTree:
    def __init__(self, max_depth=4, min_samples_leaf=20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _fit_node(self, X, r, w, depth):
        if depth >= self.max_depth or len(r) < 2 * self.min_samples_leaf:
            return {"leaf": True, "val": round(float(np.sum(r) / (np.sum(w) + 1e-6)), 5)}
        best_gain = 0.0
        best_split = None
        sum_r, sum_w = np.sum(r), np.sum(w)
        base_score = (sum_r ** 2) / (sum_w + 1e-6)
        n_samples, n_features = X.shape
        for f in range(n_features):
            col = X[:, f]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            thresholds = (
                np.unique(np.quantile(col, np.linspace(0.02, 0.98, 35)))
                if len(uniq) > 40
                else (uniq[:-1] + uniq[1:]) / 2.0
            )
            for th in thresholds:
                left = col <= th
                n_left = np.sum(left)
                if n_left < self.min_samples_leaf or (n_samples - n_left) < self.min_samples_leaf:
                    continue
                r_left, w_left = np.sum(r[left]), np.sum(w[left])
                r_right, w_right = sum_r - r_left, sum_w - w_left
                gain = (r_left ** 2) / (w_left + 1e-6) + (r_right ** 2) / (w_right + 1e-6) - base_score
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, th, left)
        if best_split is None or best_gain <= 1e-5:
            return {"leaf": True, "val": round(float(sum_r / (sum_w + 1e-6)), 5)}
        f, th, left = best_split
        return {
            "leaf": False,
            "f": int(f),
            "th": round(float(th), 4),
            "l": self._fit_node(X[left], r[left], w[left], depth + 1),
            "r": self._fit_node(X[~left], r[~left], w[~left], depth + 1),
        }

    def fit(self, X, r, w):
        self.tree = self._fit_node(X, r, w, 0)
        return self

    def _predict_node(self, node, X):
        if node["leaf"]:
            return np.full(len(X), node["val"])
        left = X[:, node["f"]] <= node["th"]
        out = np.empty(len(X))
        if np.any(left):
            out[left] = self._predict_node(node["l"], X[left])
        if np.any(~left):
            out[~left] = self._predict_node(node["r"], X[~left])
        return out

    def predict(self, X):
        return self._predict_node(self.tree, X)


class GBDTClassifier:
    def __init__(self, n_estimators=35, learning_rate=0.15, max_depth=4, min_samples_leaf=20):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.init_log_odds = 0.0

    def fit(self, X, y):
        pos_rate = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.init_log_odds = round(float(np.log(pos_rate / (1.0 - pos_rate))), 5)
        raw_pred = np.full(len(y), self.init_log_odds)
        self.trees = []
        for _ in range(self.n_estimators):
            p = expit(raw_pred)
            r = y - p
            w = p * (1.0 - p)
            tree = RegressionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(X, r, w)
            raw_pred += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        raw_pred = np.full(len(X), self.init_log_odds)
        for tree in self.trees:
            raw_pred += self.lr * tree.predict(X)
        return expit(raw_pred)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Precision": prec, "Recall": rec, "F1": f1}


if __name__ == "__main__":
    raw_path = Path(__file__).with_name("raw.csv")
    df, audit = load_and_clean_data(raw_path)
    print("Audit Pembersihan Data:", json.dumps(audit, indent=2))

    train_mask = df["waktu_dt"] < "2025-10-01"
    val_mask = (df["waktu_dt"] >= "2025-10-01") & (df["waktu_dt"] < "2025-10-15")
    test_mask = df["waktu_dt"] >= "2025-10-15"

    feat_with_lag = [
        "kt_efektif", "defisit_kt_efektif", "kelembaban_tanah_pct", "kt_prev1", "kt_jump1",
        "prakiraan_hujan_6jam_mm", "curah_hujan_mm", "suhu_udara_C", "kelembaban_udara_pct",
        "cahaya_lux", "umur_tanaman_hari", "hour", "is_daytime",
        "tanaman_cabai", "tanaman_tomat", "tanaman_selada",
        "tanah_pasir", "tanah_lempung", "tanah_liat",
    ]

    X_lag = df[feat_with_lag].values
    y_all = df["perlu_siram"].values

    gbdt = GBDTClassifier(n_estimators=35, learning_rate=0.15, max_depth=4, min_samples_leaf=20)
    gbdt.fit(X_lag[train_mask], y_all[train_mask])
    probs = gbdt.predict_proba(X_lag)

    pred_base = (df["kt_raw"] < 35.0).astype(int).values
    pred_gbdt = (probs >= 0.23).astype(int)

    for split_name, mask in [("Train", train_mask), ("Validation", val_mask), ("Test", test_mask)]:
        m_base = calc_metrics(y_all[mask], pred_base[mask])
        m_gbdt = calc_metrics(y_all[mask], pred_gbdt[mask])
        print(f"\n[{split_name}] Baseline (<35): F1={m_base['F1']:.4f} (TP={m_base['TP']}, FP={m_base['FP']}, FN={m_base['FN']})")
        print(f"[{split_name}] GBDT (Lag, th=0.23): F1={m_gbdt['F1']:.4f} (TP={m_gbdt['TP']}, FP={m_gbdt['FP']}, FN={m_gbdt['FN']})")