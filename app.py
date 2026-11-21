from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from model import MODEL, evaluate, load, observation

st.set_page_config(page_title="Irigasi Pintar", layout="wide", initial_sidebar_state="collapsed")
st.html("""<style>
header[data-testid="stHeader"], [data-testid="stToolbar"] {display:none}
.stMainBlockContainer {padding:0!important;max-width:none!important}
iframe {display:block;border:0}
</style>""")
dashboard = components.declare_component("dashboard_irigasi", path=str(Path(__file__).parent / "ui"))


@st.cache_data
def historical_plots():
    snapshots = load("petak.json")
    for snapshot in snapshots.values():
        for row in snapshot["rows"]:
            row["score"], _ = evaluate(*observation(row, snapshot["waktu"]))
    return snapshots


previous = st.session_state.get("reading")
probability = None
if previous:
    keys = ("tanaman", "jenis_tanah", "kt", "prak", "hour", "curah", "suhu", "rh", "lux", "umur")
    probability, _ = evaluate(*(previous[k] for k in keys))
reading = dashboard(metrics={k: MODEL[k] for k in ("optimal_threshold_f1", "threshold_curve")},
                    snapshots=historical_plots(), input=previous, score=probability, key="irigasi")
if reading and reading != previous:
    st.session_state.reading = reading
    st.rerun()
