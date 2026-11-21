import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
CROPS = {"cabai": 32, "tomat": 36, "selada": 42}
SOILS = ["pasir", "lempung", "liat"]


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf8"))


MODEL = load("model.json")
THRESHOLD = MODEL["optimal_threshold_f1"]


def evaluate(crop, soil, moisture, forecast, hour, rain, temperature, humidity, light, age):
    deficit = round(CROPS[crop] - moisture, 2)
    daytime = 5 <= hour <= 17
    # Konsol memperkirakan riwayat saat hanya satu pengamatan tersedia.
    features = [moisture, deficit, moisture, round(moisture + 0.4, 1), -0.4,
                forecast, rain, temperature, humidity, light, age, hour, daytime,
                *(crop == c for c in CROPS), *(soil == s for s in SOILS)]
    score = MODEL["init_log_odds"]
    for tree in MODEL["trees"]:
        node = tree
        while not node.get("leaf"):
            node = node["l"] if features[node["f"]] <= node["th"] else node["r"]
        score += MODEL["learning_rate"] * node["val"]
    probability = 1 / (1 + math.exp(-score))
    if probability >= THRESHOLD:
        verdict = "Perlu disiram sekarang"
    elif deficit > 0 and forecast >= 2:
        verdict = "Tunda penyiraman karena prakiraan hujan"
    elif deficit > 0 and not daytime:
        verdict = "Tunda hingga pukul 05.00 WIB"
    else:
        verdict = "Belum perlu disiram"
    return probability, verdict


def observation(row, timestamp):
    return (row["tanaman"], row["jenis_tanah"], row["kt_efektif"],
            row["prakiraan_hujan_6jam_mm"], int(timestamp[11:13]), row["curah_hujan_mm"],
            row["suhu_udara_C"], row["kelembaban_udara_pct"], row["cahaya_lux"], row["umur_tanaman_hari"])
