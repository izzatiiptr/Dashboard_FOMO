import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard FOMO dan Kesejahteraan Mahasiswa",
    layout="wide",
)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Data Eda Threeasure_Updated.csv")
    df.columns = df.columns.str.strip().str.lower()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Standardise categorical text
    if "fakultas" in data.columns:
        data["fakultas"] = (
            data["fakultas"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )
        data["fakultas"] = data["fakultas"].replace(
            {
                "Fakultas Imu Sosial Budaya Dan Politik": "Fakultas Ilmu Sosial Budaya Dan Politik",
                "Fakultas Ilmu Sosial Budaya Dan Politik": "Fakultas Ilmu Sosial Budaya Dan Politik",
            }
        )

    if "program_studi" in data.columns:
        data["program_studi"] = (
            data["program_studi"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
        )

    # Ensure numeric columns are numeric
    numeric_cols = [
        "rata-rata_uang_saku_perbulan",
        "pengeluaran_untuk_fomo_per_bulan",
        "kemampuan_mengelola_keuangan",
        "frekuensi_fomo_pengeluaran",
        "pengaruh_fomo_terhadap_emosi",
        "frekuensi_stres_karena_finansial",
        "frekuensi_hilang_semangat_kuliah_karena_tekanan_finansial",
        "frekuensi_stres_fomo",
        "frekuensi_kegiatan_karena_fomo",
        "kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis_numerik",
        "skor_psikologis",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "timestamp" in data.columns:
        data["hari"] = data["timestamp"].dt.day_name()
        data["jam"] = data["timestamp"].dt.hour
        data["minggu"] = data["timestamp"].dt.to_period("W").astype(str)

    # Derived features to make the visuals richer
    if {"pengeluaran_untuk_fomo_per_bulan", "rata-rata_uang_saku_perbulan"}.issubset(data.columns):
        uang = data["rata-rata_uang_saku_perbulan"].replace(0, np.nan)
        proporsi = data["pengeluaran_untuk_fomo_per_bulan"] / uang
        data["proporsi_fomo"] = proporsi.clip(lower=0)
        data["sisa_uang_saku"] = data["rata-rata_uang_saku_perbulan"] - data["pengeluaran_untuk_fomo_per_bulan"]
        data["kategori_proporsi"] = pd.cut(
            data["proporsi_fomo"],
            bins=[0, 0.2, 0.5, np.inf],
            labels=["Rendah (<20%)", "Sedang (20-50%)", "Tinggi (>50%)"],
            include_lowest=True,
        )

    if "kemampuan_mengelola_keuangan" in data.columns:
        data["kategori_keuangan"] = pd.cut(
            data["kemampuan_mengelola_keuangan"],
            bins=[0, 2.5, 3.5, 5],
            labels=["Kurang", "Cukup", "Baik"],
            include_lowest=True,
        )

    if "sering_merasa_fomo" in data.columns:
        data["kategori_fomo"] = (
            data["sering_merasa_fomo"].fillna("Tidak").str.strip().str.lower().map({"ya": "Sering", "tidak": "Jarang"})
        )
        data["kategori_fomo"] = data["kategori_fomo"].fillna("Jarang")

    if {"frekuensi_fomo_pengeluaran", "frekuensi_kegiatan_karena_fomo"}.issubset(data.columns):
        data["skor_fomo_relatif"] = (
            data[["frekuensi_fomo_pengeluaran", "frekuensi_kegiatan_karena_fomo"]].sum(axis=1)
        ) / 2

    stress_cols = [
        "pengaruh_fomo_terhadap_emosi",
        "frekuensi_stres_karena_finansial",
        "frekuensi_hilang_semangat_kuliah_karena_tekanan_finansial",
        "frekuensi_stres_fomo",
    ]
    stress_ok = [c for c in stress_cols if c in data.columns]
    if stress_ok:
        data["indeks_stres"] = data[stress_ok].mean(axis=1)

    return data

df = preprocess_data(load_data())

# Warna Palet
SALMON = "#FA8072"
SKYBLUE = "#87CEEB"
GRADIENT = ["#C94A44", "#E96A5F", "#FF8C75", "#FFB5A7", "#FFE5E0"]
MINT = "#7FC8A9"
GRADIENT_SCALE = [
    [0.0, GRADIENT[0]],
    [0.25, GRADIENT[1]],
    [0.5, GRADIENT[2]],
    [0.75, GRADIENT[3]],
    [1.0, GRADIENT[4]],
]

def set_page(page: str) -> None:
    st.session_state["page"] = page

def render_plot(fig, container=None):
    """Helper to render Plotly figures responsively."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(250, 128, 114, 0.04)",
        plot_bgcolor="rgba(250, 128, 114, 0.02)",
        font=dict(color="#343A40", family="Poppins, sans-serif"),
        title=dict(font=dict(color=SALMON, size=20), x=0.5, xanchor="center", pad=dict(b=12)),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0)", font=dict(color="#343A40")),
        colorway=[
            "rgba(255,245,240,0.55)",
            "rgba(254,224,210,0.55)",
            "rgba(252,187,161,0.55)",
            "rgba(252,146,114,0.55)",
            "rgba(251,106,74,0.55)",
            "rgba(239,59,44,0.55)"
        ],
    )
    fig.update_coloraxes(colorscale=GRADIENT_SCALE)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(201, 74, 68, 0.15)", zeroline=False, linecolor="rgba(201, 74, 68, 0.3)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(201, 74, 68, 0.15)", zeroline=False, linecolor="rgba(201, 74, 68, 0.3)")
    target = container if container is not None else st
    target.plotly_chart(fig, config={"responsive": True})

def load_local_css() -> None:
    css_path = Path("styles/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

def render_banner(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="banner">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

load_local_css()

# Sidebar Navigasi
st.sidebar.header("Navigasi")
pages = ["Pendahuluan", "Analisis & Visualisasi", "Eksplorasi Interaktif", "Kesimpulan"]
if "page" not in st.session_state:
    st.session_state["page"] = pages[0]

for page in pages:
    btn_key = f"nav_{page.lower().replace(' ', '_').replace('&', 'and')}"
    st.sidebar.button(
        page,
        key=btn_key,
        width="stretch",
        on_click=set_page,
        args=(page,),
        disabled=st.session_state["page"] == page,
    )

menu = st.session_state["page"]

# Halaman 1: Pendahuluan
if menu == "Pendahuluan":
    render_banner(
        "Pendahuluan",
        "Ringkasan proyek analisis FOMO mahasiswa dan gambaran struktur dataset yang digunakan."
    )

    st.markdown(
        """
        <div class="card">
            <h2>Gambaran Proyek</h2>
            <p>
                Dashboard ini merangkum hasil survei mengenai fenomena <em>Fear of Missing Out</em> (FOMO)
                di kalangan mahasiswa. Fokus utamanya adalah menggali hubungan antara tekanan sosial,
                perilaku konsumtif, dan kondisi kesejahteraan psikologis.
            </p>
            <p>
                Data yang dianalisis mencakup 153 responden lintas fakultas di UPN Veteran Jawa Timur.
                Survei menyoroti seberapa sering mahasiswa mengalami FOMO, besaran alokasi pengeluaran
                yang terdorong oleh FOMO, hingga kemampuan mereka mengelola keuangan pribadi.
            </p>
            <h3>Struktur Dataset</h3>
            <ul>
                <li><strong>Identitas:</strong> timestamp, fakultas, program studi.</li>
                <li><strong>FOMO &amp; Emosi:</strong> frekuensi FOMO, pengaruh terhadap emosi, indeks stres.</li>
                <li><strong>Keuangan:</strong> uang saku bulanan, pengeluaran karena FOMO, kemampuan mengelola keuangan.</li>
                <li><strong>Kesejahteraan:</strong> skor psikologis, kebutuhan dukungan emosional.</li>
            </ul>
            <p>
                Dengan menggabungkan visualisasi interaktif dan ringkasan statistik,
                dashboard ini diharapkan mampu menjadi referensi untuk merancang program pendampingan
                maupun kebijakan peningkatan literasi finansial dan kesehatan mental mahasiswa.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responden", f"{len(df):,}")

    if "fakultas" in df.columns and df["fakultas"].notna().any():
        col2.metric(
            "Fakultas Terbanyak",
            df["fakultas"].mode().iat[0],
        )

    if "proporsi_fomo" in df.columns and df["proporsi_fomo"].notna().any():
        col3.metric(
            "Rata-rata Proporsi FOMO",
            f"{df['proporsi_fomo'].mean() * 100:,.1f}%",
            help="Proporsi pengeluaran FOMO dibandingkan uang saku bulanan.",
        )

    if "pengeluaran_untuk_fomo_per_bulan" in df.columns:
        col4.metric(
            "Pengeluaran FOMO Median",
            f"Rp {df['pengeluaran_untuk_fomo_per_bulan'].median():,.0f}",
        )

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        st.caption(
            f"Periode survei: {df['timestamp'].dropna().min().date()} - {df['timestamp'].dropna().max().date()}"
        )

elif menu == "Analisis & Visualisasi":
    render_banner(
        "Analisis & Visualisasi",
        "Eksplorasi statistik utama terkait distribusi responden, perilaku FOMO, dan kesejahteraan mahasiswa."
    )
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Responden Harian", "Fakultas", "FOMO dan Keuangan", "Psikologis"]
    )

    # Tab 1
    with tab1:
        st.subheader("Tren Jumlah Responden Harian")
        if "timestamp" in df.columns:
            data_ts = df.dropna(subset=["timestamp"]).copy()
            data_ts["tanggal"] = data_ts["timestamp"].dt.date
            daily = (
                data_ts.groupby("tanggal")
                .size()
                .reset_index(name="Jumlah Responden")
                .sort_values("tanggal")
            )
            col_ts_1, col_ts_2 = st.columns(2)
            with col_ts_1:
                fig = px.line(
                    daily,
                    x="tanggal",
                    y="Jumlah Responden",
                    markers=True,
                    color_discrete_sequence=[SALMON],
                    title="Tren Jumlah Responden per Hari",
                )
                fig.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Responden")
                render_plot(fig)

            with col_ts_2:
                daily["Kumulatif Responden"] = daily["Jumlah Responden"].cumsum()
                fig_cum = px.area(
                    daily,
                    x="tanggal",
                    y="Kumulatif Responden",
                    color_discrete_sequence=[SALMON],
                    title="Akumulasi Responden Selama Periode Survei",
                )
                fig_cum.update_traces(
                    line=dict(color=SALMON),
                    fillcolor="rgba(250,128,114,0.25)",
                )
                fig_cum.update_layout(xaxis_title="Tanggal", yaxis_title="Responden Kumulatif")
                render_plot(fig_cum)

            weekly = (
                data_ts.assign(minggu=data_ts["timestamp"].dt.to_period("W").astype(str))
                .groupby("minggu")
                .size()
                .reset_index(name="Responden")
            )
            col_ts_3, col_ts_4 = st.columns(2)
            with col_ts_3:
                fig_week = px.bar(
                    weekly,
                    x="minggu",
                    y="Responden",
                    text="Responden",
                    color_discrete_sequence=[SALMON],
                    title="Distribusi Responden per Minggu",
                )
                fig_week.update_traces(textposition="outside")
                fig_week.update_layout(xaxis_title="Minggu (Periode)", yaxis_title="Jumlah Responden")
                render_plot(fig_week)

            with col_ts_4:
                if {"hari", "jam"}.issubset(data_ts.columns):
                    order_hari = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    heat = (
                        data_ts.groupby(["hari", "jam"])
                        .size()
                        .reset_index(name="Responden")
                    )
                    heat["hari"] = pd.Categorical(heat["hari"], categories=order_hari, ordered=True)
                    heat = heat.sort_values(["hari", "jam"])
                    fig_heat = px.density_heatmap(
                        heat,
                        x="jam",
                        y="hari",
                        z="Responden",
                        color_continuous_scale=GRADIENT,
                        title="Kepadatan Responden Berdasarkan Hari dan Jam",
                    )
                    fig_heat.update_layout(xaxis_title="Jam", yaxis_title="Hari")
                    render_plot(fig_heat)
                else:
                    st.info("Data jam responden belum tersedia untuk heatmap.")
        else:
            st.warning("Kolom 'timestamp' tidak ditemukan.")

    # Tab 2
    with tab2:
        st.subheader("Distribusi Responden per Fakultas")
        if "fakultas" in df.columns:
            fak = (
                df.groupby("fakultas")
                .size()
                .reset_index(name="Jumlah")
                .sort_values("Jumlah", ascending=True)
            ).rename(columns={"fakultas": "Fakultas"})
            col_fac_top = st.columns(2)
            with col_fac_top[0]:
                fig = px.bar(
                    fak,
                    x="Jumlah",
                    y="Fakultas",
                    orientation="h",
                    color_discrete_sequence=[SKYBLUE],
                    text="Jumlah",
                    title="Distribusi Responden Berdasarkan Fakultas",
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                render_plot(fig)

            with col_fac_top[1]:
                if "proporsi_fomo" in df.columns and df["proporsi_fomo"].notna().any():
                    top_proporsi = (
                        df.groupby("fakultas")["proporsi_fomo"]
                        .mean()
                        .reset_index()
                        .sort_values("proporsi_fomo", ascending=False)
                        .head(5)
                    ).rename(columns={"fakultas": "Fakultas", "proporsi_fomo": "Proporsi FOMO"})
                    top_proporsi["Label"] = top_proporsi["Proporsi FOMO"].apply(lambda x: f"{x*100:,.1f}%")
                    fig_prop = px.bar(
                        top_proporsi,
                        x="Fakultas",
                        y="Proporsi FOMO",
                        color="Proporsi FOMO",
                        text="Label",
                        color_continuous_scale=GRADIENT,
                        title="Top 5 Fakultas dengan Proporsi Pengeluaran FOMO Tertinggi",
                    )
                    fig_prop.update_layout(
                        xaxis_title="Fakultas", yaxis_title="Proporsi FOMO Rata-rata", uniformtext_minsize=10
                    )
                    render_plot(fig_prop)
                else:
                    st.info("Data proporsi pengeluaran FOMO belum tersedia.")

            col_fac_bottom = st.columns(2)
            with col_fac_bottom[0]:
                if {"fakultas", "pengeluaran_untuk_fomo_per_bulan"}.issubset(df.columns):
                    box_data = df.dropna(subset=["fakultas", "pengeluaran_untuk_fomo_per_bulan"])
                    if not box_data.empty:
                        fig_box = px.box(
                            box_data,
                            x="fakultas",
                            y="pengeluaran_untuk_fomo_per_bulan",
                            points="all",
                            title="Sebaran Pengeluaran FOMO Bulanan per Fakultas",
                            color="fakultas",
                            color_discrete_sequence=GRADIENT,
                        )
                        fig_box.update_layout(
                            xaxis_title="Fakultas", yaxis_title="Pengeluaran FOMO per Bulan (Rp)", showlegend=False
                        )
                        render_plot(fig_box)
                    else:
                        st.info("Data pengeluaran FOMO per fakultas belum tersedia.")
                else:
                    st.info("Kolom pengeluaran FOMO belum tersedia.")

            with col_fac_bottom[1]:
                if {"fakultas", "kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis"}.issubset(df.columns):
                    dukungan = (
                        df.groupby(["fakultas", "kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis"])
                        .size()
                        .reset_index(name="Responden")
                    )
                    if not dukungan.empty:
                        fig_support = px.bar(
                            dukungan,
                            x="fakultas",
                            y="Responden",
                            color="kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis",
                            color_discrete_map={"Ya": SALMON, "Tidak": SKYBLUE},
                            title="Kebutuhan Dukungan Emosional per Fakultas",
                            barmode="stack",
                        )
                        fig_support.update_layout(
                            xaxis_title="Fakultas", yaxis_title="Jumlah Responden", legend_title="Butuh Dukungan"
                        )
                        render_plot(fig_support)
                    else:
                        st.info("Data kebutuhan dukungan emosional belum tersedia.")
                else:
                    st.info("Kolom kebutuhan dukungan emosional belum tersedia.")

            if "program_studi" in df.columns:
                st.caption("Klik pada visualisasi untuk memperbesar rincian per program studi.")
                treemap_source = (
                    df.groupby(["fakultas", "program_studi"])
                    .size()
                    .reset_index(name="Responden")
                )
                if not treemap_source.empty:
                    fig_tree = px.treemap(
                        treemap_source,
                        path=["fakultas", "program_studi"],
                        values="Responden",
                        color="Responden",
                        color_continuous_scale=GRADIENT,
                        title="Pemetaan Responden per Fakultas dan Program Studi",
                    )
                    render_plot(fig_tree)
                else:
                    st.info("Data program studi belum tersedia.")
        else:
            st.warning("Kolom 'fakultas' tidak tersedia.")

    # Tab 3
    with tab3:
        st.subheader("Proporsi Pengeluaran FOMO dari Uang Saku")
        row_fomo_top = st.columns(2)
        with row_fomo_top[0]:
            if "kategori_proporsi" in df.columns and df["kategori_proporsi"].notna().any():
                pie_source = df.dropna(subset=["kategori_proporsi"])
                fig = px.pie(
                    pie_source,
                    names="kategori_proporsi",
                    title="Proporsi Pengeluaran FOMO dari Uang Saku",
                    color="kategori_proporsi",
                    color_discrete_map={
                        "Rendah (<20%)": "#FFE5E0",
                        "Sedang (20-50%)": "#FF8C75",
                        "Tinggi (>50%)": "#C94A44",
                    },
                )
                fig.update_traces(textinfo="percent+label")
                render_plot(fig)
            else:
                st.info("Data kategori proporsi FOMO belum tersedia.")

        with row_fomo_top[1]:
            if {"kategori_fomo", "pengeluaran_untuk_fomo_per_bulan"}.issubset(df.columns):
                avg_spend = (
                    df.groupby("kategori_fomo")["pengeluaran_untuk_fomo_per_bulan"]
                    .mean()
                    .reset_index()
                    .rename(
                        columns={
                            "kategori_fomo": "Kategori FOMO",
                            "pengeluaran_untuk_fomo_per_bulan": "Rata-rata Pengeluaran",
                        }
                    )
                )
                fig_avg_spend = px.bar(
                    avg_spend,
                    x="Kategori FOMO",
                    y="Rata-rata Pengeluaran",
                    color="Kategori FOMO",
                    text=avg_spend["Rata-rata Pengeluaran"].apply(lambda x: f"Rp {x:,.0f}"),
                    color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
                    title="Rata-rata Pengeluaran FOMO Berdasarkan Kategori FOMO",
                )
                fig_avg_spend.update_layout(showlegend=False, yaxis_title="Rata-rata Pengeluaran (Rp)")
                render_plot(fig_avg_spend)
            else:
                st.info("Data kategori FOMO belum lengkap untuk perbandingan pengeluaran.")

        row_fomo_bottom = st.columns(2)
        with row_fomo_bottom[0]:
            if {"rata-rata_uang_saku_perbulan", "pengeluaran_untuk_fomo_per_bulan"}.issubset(df.columns):
                scatter_budget = px.scatter(
                    df,
                    x="rata-rata_uang_saku_perbulan",
                    y="pengeluaran_untuk_fomo_per_bulan",
                    size="pengeluaran_untuk_fomo_per_bulan",
                    color="kategori_fomo" if "kategori_fomo" in df.columns else None,
                    color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
                    hover_data=["fakultas"] if "fakultas" in df.columns else None,
                    title="Uang Saku vs Pengeluaran FOMO",
                )
                scatter_budget.update_layout(
                    xaxis_title="Rata-rata Uang Saku per Bulan (Rp)",
                    yaxis_title="Pengeluaran FOMO per Bulan (Rp)",
                )
                render_plot(scatter_budget)
            else:
                st.info("Kolom uang saku dan pengeluaran FOMO diperlukan untuk scatter plot.")

        with row_fomo_bottom[1]:
            if {"kategori_keuangan", "proporsi_fomo"}.issubset(df.columns):
                violin_prop = px.violin(
                    df,
                    x="kategori_keuangan",
                    y="proporsi_fomo",
                    color="kategori_keuangan",
                    color_discrete_sequence=[SALMON, SKYBLUE, MINT],
                    box=True,
                    points="all",
                    title="Sebaran Proporsi FOMO berdasarkan Kategori Keuangan",
                )
                violin_prop.update_layout(
                    xaxis_title="Kategori Keuangan",
                    yaxis_title="Proporsi Pengeluaran FOMO",
                    showlegend=False,
                )
                render_plot(violin_prop)
            else:
                st.info("Data proporsi FOMO dan kategori keuangan dibutuhkan untuk violin plot.")

        st.markdown("---")
        st.subheader("Hubungan FOMO dan Kemampuan Keuangan")
        if {"kategori_fomo", "kategori_keuangan"}.issubset(df.columns):
            crosstab = pd.crosstab(
                df["kategori_fomo"],
                df["kategori_keuangan"],
                normalize="index",
            ) * 100
            fig = px.imshow(
                crosstab,
                text_auto=".1f",
                color_continuous_scale=GRADIENT,
                title="Heatmap FOMO vs Keuangan (Proporsi per Kategori FOMO)",
            )
            fig.update_layout(coloraxis_colorbar_title="%")
            render_plot(fig)
        else:
            st.info("Diperlukan data kategori FOMO dan keuangan untuk menampilkan heatmap.")

    # Tab 4
    with tab4:
        st.subheader("Distribusi Skor Psikologis Mahasiswa")
        if "skor_psikologis" in df.columns:
            psy_row_top = st.columns(2)
            with psy_row_top[0]:
                fig = px.box(
                    df,
                    y="skor_psikologis",
                    points="all",
                    color_discrete_sequence=[SALMON],
                    title="Distribusi Skor Psikologis Mahasiswa",
                )
                render_plot(fig)

            with psy_row_top[1]:
                hist = px.histogram(
                    df,
                    x="skor_psikologis",
                    nbins=15,
                    color_discrete_sequence=[SALMON],
                    title="Histogram Skor Psikologis",
                )
                hist.update_layout(xaxis_title="Skor Psikologis", yaxis_title="Jumlah Responden")
                render_plot(hist)

            psy_row_bottom = st.columns(2)
            with psy_row_bottom[0]:
                if "kategori_fomo" in df.columns:
                    violin_psy = px.violin(
                        df,
                        x="kategori_fomo",
                        y="skor_psikologis",
                        color="kategori_fomo",
                        color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
                        box=True,
                        points="all",
                        title="Skor Psikologis berdasarkan Kategori FOMO",
                    )
                    violin_psy.update_layout(
                        xaxis_title="Kategori FOMO",
                        yaxis_title="Skor Psikologis",
                        showlegend=False,
                    )
                    render_plot(violin_psy)
                else:
                    st.info("Kategori FOMO belum tersedia untuk perbandingan skor psikologis.")

            with psy_row_bottom[1]:
                if {"indeks_stres", "kategori_keuangan"}.issubset(df.columns) and df["indeks_stres"].notna().any():
                    stres_summary = (
                        df.groupby("kategori_keuangan")["indeks_stres"]
                        .mean()
                        .reset_index(name="Indeks Stres Rata-rata")
                    )
                    if not stres_summary.empty:
                        fig_stress = px.bar(
                            stres_summary,
                            x="kategori_keuangan",
                            y="Indeks Stres Rata-rata",
                            color="Indeks Stres Rata-rata",
                            color_continuous_scale=GRADIENT,
                            title="Rata-rata Indeks Stres per Kategori Keuangan",
                        )
                        fig_stress.update_layout(xaxis_title="Kategori Keuangan", yaxis_title="Indeks Stres Rata-rata")
                        render_plot(fig_stress)
                    else:
                        st.info("Indeks stres belum dapat dihitung.")
                else:
                    st.info("Perlu data indeks stres dan kategori keuangan.")

        st.markdown("---")
        st.subheader("Korelasi Antar Variabel Psikologis dan Keuangan")
        cols = [
            "kemampuan_mengelola_keuangan",
            "pengaruh_fomo_terhadap_emosi",
            "frekuensi_stres_karena_finansial",
            "frekuensi_hilang_semangat_kuliah_karena_tekanan_finansial",
            "frekuensi_stres_fomo",
            "skor_psikologis",
        ]
        rel_row = st.columns(2)
        with rel_row[0]:
            if {"proporsi_fomo", "skor_psikologis"}.issubset(df.columns):
                scatter = px.scatter(
                    df,
                    x="proporsi_fomo",
                    y="skor_psikologis",
                    color="kategori_fomo" if "kategori_fomo" in df.columns else None,
                    color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
                    hover_data=["fakultas"] if "fakultas" in df.columns else None,
                    trendline="ols",
                    title="Proporsi Pengeluaran FOMO vs Skor Psikologis",
                )
                scatter.update_layout(
                    xaxis_tickformat="%",
                    xaxis_title="Proporsi Pengeluaran FOMO",
                    yaxis_title="Skor Psikologis",
                )
                render_plot(scatter)
            else:
                st.info("Data proporsi FOMO dan skor psikologis belum lengkap.")

        with rel_row[1]:
            if {"skor_psikologis", "kemampuan_mengelola_keuangan"}.issubset(df.columns):
                fig_density = px.density_contour(
                    df,
                    x="kemampuan_mengelola_keuangan",
                    y="skor_psikologis",
                    color="kategori_fomo" if "kategori_fomo" in df.columns else None,
                    color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
                    title="Kepadatan Skor Keuangan dan Psikologis",
                )
                fig_density.update_layout(
                    xaxis_title="Kemampuan Mengelola Keuangan",
                    yaxis_title="Skor Psikologis",
                )
                render_plot(fig_density)
            else:
                st.info("Data kemampuan keuangan dan skor psikologis dibutuhkan untuk kontur kepadatan.")

# Halaman 3: Eksplorasi Interaktif
elif menu == "Eksplorasi Interaktif":
    render_banner(
        "Eksplorasi Interaktif",
        "Gunakan filter dinamis untuk meninjau hubungan antar variabel sesuai kebutuhan analisis."
    )
    st.markdown("Gunakan filter di bawah untuk menjelajahi data secara dinamis:")

    col1, col2, col3, col4 = st.columns(4)
    fakultas = (
        col1.selectbox(
            "Fakultas", ["Semua"] + sorted(df["fakultas"].dropna().unique().tolist())
        )
        if "fakultas" in df.columns
        else "Semua"
    )
    fomo = (
        col2.selectbox(
            "Tingkat FOMO", ["Semua"] + sorted(df["kategori_fomo"].dropna().unique().tolist())
        )
        if "kategori_fomo" in df.columns
        else "Semua"
    )
    keuangan = (
        col3.selectbox(
            "Kategori Keuangan",
            ["Semua"] + sorted(df["kategori_keuangan"].dropna().unique().tolist()),
        )
        if "kategori_keuangan" in df.columns
        else "Semua"
    )
    if "proporsi_fomo" in df.columns and df["proporsi_fomo"].notna().any():
        prop_max_val = float(df["proporsi_fomo"].max())
        if not np.isfinite(prop_max_val) or prop_max_val <= 0:
            prop_max_val = 1.0
        step_size = max(round(prop_max_val / 20, 2), 0.05)
        prop_range = col4.slider(
            "Rentang Proporsi Pengeluaran FOMO",
            0.0,
            prop_max_val,
            (0.0, prop_max_val),
            step=step_size,
        )
    else:
        prop_range = None

    if "program_studi" in df.columns:
        program = st.multiselect(
            "Program Studi", sorted(df["program_studi"].dropna().unique().tolist()), default=[]
        )
    else:
        program = None

    data = df.copy()
    if fakultas != "Semua":
        data = data[data["fakultas"] == fakultas]
    if fomo != "Semua" and "kategori_fomo" in df.columns:
        data = data[data["kategori_fomo"] == fomo]
    if keuangan != "Semua" and "kategori_keuangan" in df.columns:
        data = data[data["kategori_keuangan"] == keuangan]
    if prop_range and "proporsi_fomo" in df.columns:
        lower, upper = prop_range
        data = data[data["proporsi_fomo"].between(lower, upper)]
    if program:
        data = data[data["program_studi"].isin(program)]

    st.write(f"Menampilkan {len(data)} responden sesuai filter.")

    col1, col2 = st.columns(2)
    if {"skor_fomo_relatif", "skor_psikologis"}.issubset(data.columns):
        fig1 = px.scatter(
            data,
            x="skor_fomo_relatif",
            y="skor_psikologis",
            color="kategori_fomo" if "kategori_fomo" in data.columns else None,
            color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
            title="Skor FOMO Relatif vs Skor Psikologis",
        )
        fig1.update_layout(
            xaxis_title="Skor FOMO Relatif",
            yaxis_title="Skor Psikologis",
        )
        render_plot(fig1, container=col1)
    elif {"frekuensi_fomo_pengeluaran", "skor_psikologis"}.issubset(data.columns):
        fig_alt = px.scatter(
            data,
            x="frekuensi_fomo_pengeluaran",
            y="skor_psikologis",
            color="kategori_fomo" if "kategori_fomo" in data.columns else None,
            color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
            title="Frekuensi Pengeluaran karena FOMO vs Skor Psikologis",
        )
        fig_alt.update_layout(xaxis_title="Frekuensi Pengeluaran karena FOMO", yaxis_title="Skor Psikologis")
        render_plot(fig_alt, container=col1)

    if {"kemampuan_mengelola_keuangan", "skor_psikologis"}.issubset(data.columns):
        fig2 = px.scatter(
            data,
            x="kemampuan_mengelola_keuangan",
            y="skor_psikologis",
            color="kategori_keuangan" if "kategori_keuangan" in data.columns else None,
            color_discrete_sequence=[SALMON, SKYBLUE, MINT],
            title="Kemampuan Mengelola Keuangan vs Skor Psikologis",
        )
        fig2.update_layout(xaxis_title="Kemampuan Mengelola Keuangan", yaxis_title="Skor Psikologis")
        render_plot(fig2, container=col2)

    extra_col1, extra_col2 = st.columns(2)
    if {"kategori_fomo", "indeks_stres"}.issubset(data.columns):
        stress_breakdown = (
            data.groupby("kategori_fomo")["indeks_stres"]
            .mean()
            .reset_index(name="Indeks Stres Rata-rata")
        )
        if not stress_breakdown.empty:
            fig_inter_stress = px.bar(
                stress_breakdown,
                x="kategori_fomo",
                y="Indeks Stres Rata-rata",
                color="kategori_fomo",
                color_discrete_map={"Sering": SALMON, "Jarang": SKYBLUE},
                title="Indeks Stres Rata-rata per Kategori FOMO (Filter Aktif)",
            )
            fig_inter_stress.update_layout(xaxis_title="Kategori FOMO", showlegend=False)
            render_plot(fig_inter_stress, container=extra_col1)
        else:
            with extra_col1:
                st.info("Tidak ada data indeks stres untuk filter saat ini.")
    else:
        with extra_col1:
            st.info("Data indeks stres tidak tersedia untuk visualisasi.")

    if {"kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis"}.issubset(data.columns):
        support_filtered = (
            data["kebutuhan_akan_dukungan_emosional_dan_bantuan_psikologis"]
            .value_counts()
            .reset_index()
        )
        support_filtered.columns = ["Kebutuhan Dukungan", "Responden"]
        if not support_filtered.empty:
            fig_support_filtered = px.pie(
                support_filtered,
                names="Kebutuhan Dukungan",
                values="Responden",
                color="Kebutuhan Dukungan",
                color_discrete_map={"Ya": SALMON, "Tidak": SKYBLUE},
                title="Proporsi Kebutuhan Dukungan Emosional (Filter Aktif)",
            )
            render_plot(fig_support_filtered, container=extra_col2)
        else:
            with extra_col2:
                st.info("Tidak ada data dukungan emosional untuk filter saat ini.")
    else:
        with extra_col2:
            st.info("Data dukungan emosional tidak tersedia.")

    if "proporsi_fomo" in data.columns:
        st.markdown("### Ringkasan Tabel")
        st.dataframe(
            data[
                [
                    col
                    for col in [
                        "nama_lengkap",
                        "fakultas",
                        "program_studi",
                        "kategori_fomo",
                        "kategori_keuangan",
                        "proporsi_fomo",
                        "skor_psikologis",
                    ]
                    if col in data.columns
                ]
            ].rename(columns={"proporsi_fomo": "Proporsi FOMO", "skor_psikologis": "Skor Psikologis"}),
            width="stretch",
        )

# Halaman 4: Kesimpulan
else:
    render_banner(
        "Kesimpulan",
        "Ringkasan temuan utama dan rekomendasi tindak lanjut dari hasil analisis dashboard."
    )
    st.title("Kesimpulan Umum")
    st.markdown("""
    Berdasarkan hasil analisis:
    - Mahasiswa dengan **tingkat FOMO tinggi** cenderung memiliki **kemampuan pengelolaan keuangan yang rendah**.
    - Semakin **baik kemampuan finansial**, semakin **stabil kesejahteraan psikologis**.
    - Diperlukan peningkatan **literasi keuangan dan kesadaran digital** di kalangan mahasiswa.
    """)
