import pandas as pd
import streamlit as st
from model import CROPS, SOILS, MODEL, THRESHOLD, evaluate, load, observation

st.set_page_config(page_title="Irigasi Pintar", page_icon="🌳", layout="wide")
st.title("🌳 Irigasi Pintar")
st.caption("Rumah kaca Lembang · Rekomendasi penyiraman per petak")
console, plots, evaluation = st.tabs(["Konsol", "16 Petak", "Evaluasi"])
snapshots = load("petak.json")
presets = {
    "Selada kering": ("P05", "selada", "lempung", 38.5, 0, 13, 0, 23.5, 74, 54000, 65),
    "Tahan hujan": ("P02", "cabai", "pasir", 29.5, 3.8, 11, 0, 22, 82, 38000, 78),
    "Cabai aman": ("P01", "cabai", "lempung", 33.5, 0, 10, 0, 22.5, 76, 48000, 55),
    "Malam kering": ("P04", "tomat", "liat", 33.8, 0, 21, 0, 17.5, 93, 0, 80),
}
fields = ["crop", "soil", "moisture", "forecast", "hour", "rain", "temperature", "humidity", "light", "age"]
controls = [
    ("Kelembapan tanah (%)", 0, 100), ("Prakiraan hujan 6 jam (mm)", 0, 100),
    ("Jam pengamatan (WIB)", 0, 23), ("Curah hujan (mm)", 0, 100),
    ("Suhu udara (°C)", -10, 60), ("Kelembapan udara (%)", 0, 100),
    ("Cahaya (lux)", 0, 200000), ("Umur tanaman (hari)", 0, 365),
]


def set_plot(plot, values):
    st.session_state["plot"] = plot
    for key, value in zip(fields, values):
        st.session_state[key] = value if key in ("crop", "soil") else float(value)
    st.session_state["hour"] = int(st.session_state["hour"])


def apply_preset():
    plot, *values = presets[st.session_state["preset"]]
    set_plot(plot, values)


meta = {r["petak"]: r for r in next(iter(snapshots.values()))["rows"]}


def change_plot():
    row = meta[st.session_state.plot]
    st.session_state.update(crop=row["tanaman"], soil=row["jenis_tanah"])


if "plot" not in st.session_state:
    set_plot("P05", presets["Selada kering"][1:])

with console:
    st.selectbox("Contoh kondisi", presets.keys(), key="preset", on_change=apply_preset)
    cols = st.columns(3)
    cols[0].selectbox("Petak", list(meta), key="plot", on_change=change_plot)
    crop = cols[1].selectbox("Tanaman", list(CROPS), key="crop")
    soil = cols[2].selectbox("Jenis tanah", SOILS, key="soil")
    cols = st.columns(2)
    for i, (key, (label, low, high)) in enumerate(zip(fields[2:], controls)):
        if key == "hour":
            cols[i % 2].slider(label, low, high, key=key)
        else:
            cols[i % 2].number_input(label, float(low), float(high), key=key)
    values = [st.session_state[k] for k in fields]
    probability, verdict = evaluate(*values)
    st.subheader(verdict)
    cols = st.columns(3)
    cols[0].metric("Probabilitas perlu siram", f"{probability:.1%}")
    cols[1].metric("Ambang kelembapan tanaman", f"{CROPS[crop]}%")
    cols[2].metric("Baseline tanah <35%", "Siram" if values[2] < 35 else "Tunda")
    st.caption(f"Ambang model {THRESHOLD:.0%}. Selisih kelembapan terhadap ambang tanaman {values[2]-CROPS[crop]:+.1f} poin persentase.")
    st.subheader("Glosarium")
    for term, definition in load("glosarium.json").items():
        with st.expander(term):
            st.write(definition)

with plots:
    scenario = st.selectbox("Kondisi historis", list(snapshots), format_func=lambda k: k.replace("_", " ").title())
    snapshot = snapshots[scenario]
    st.caption(snapshot["waktu"] + " WIB")
    records = []
    for row in snapshot["rows"]:
        score, decision = evaluate(*observation(row, snapshot["waktu"]))
        records.append({"Petak": row["petak"], "Tanaman": row["tanaman"], "Tanah": row["jenis_tanah"],
                        "Kelembapan efektif (%)": row["kt_efektif"], "Peluang siram (%)": round(score*100, 1),
                        "Rekomendasi": decision, "Label agronom": row["perlu_siram"]})
    st.dataframe(records, hide_index=True)
    selected = st.selectbox("Petak untuk diperiksa di Konsol", snapshot["rows"], format_func=lambda r: r["petak"])
    st.button("Muat kondisi ke Konsol", on_click=set_plot,
              args=(selected["petak"], observation(selected, snapshot["waktu"])))
    st.caption("Setelah memuat kondisi, buka tab Konsol untuk mengubah pembacaan.")

with evaluation:
    st.subheader("Hasil uji 15–31 Oktober 2025")
    st.write("Pada ambang 0,23, model menemukan 131 dari 175 kebutuhan siram dengan 7 alarm keliru. F1 model mencapai 0,8371 dan F1 baseline 0,1522.")
    curve = pd.DataFrame(MODEL["threshold_curve"])
    test = curve[curve.split == "test"].set_index("th").sort_index()
    st.line_chart(test[["Precision", "Recall", "F1"]])
    st.subheader("Simulasi biaya kesalahan")
    threshold = st.select_slider("Ambang simulasi", options=list(test.index), value=min(test.index, key=lambda t:abs(t-THRESHOLD)))
    fn_cost = st.number_input("Biaya kebutuhan terlewat (Rp)", min_value=0, value=75000)
    fp_cost = st.number_input("Biaya alarm keliru (Rp)", min_value=0, value=25000)
    costs = test.FN * fn_cost + test.FP * fp_cost
    baseline = 133 * fn_cost + 335 * fp_cost
    st.dataframe(test.loc[[threshold], ["TP","FP","FN","TN","Precision","Recall","F1"]])
    st.write(f"Biaya model Rp{costs.loc[threshold]:,.0f} · Baseline Rp{baseline:,.0f} · Selisih Rp{baseline-costs.loc[threshold]:,.0f}")
    st.caption(f"Ambang dengan biaya terendah pada kurva uji ini {costs.idxmin():.2f}.")

st.divider()
st.caption("Juan Stevan Immanuel · Naail Safwan Saleh")
