import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Download required NLTK data (runs once, cached after)
@st.cache_resource
def load_nltk():
    try:
        nltk.download("vader_lexicon", quiet=True, raise_on_error=False)
        nltk.download("stopwords", quiet=True, raise_on_error=False)
    except Exception:
        pass
    return SentimentIntensityAnalyzer(), set(stopwords.words("english"))

sia, NLTK_STOPWORDS = load_nltk()

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Student Feedback & Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(145deg, #0a0a0f, #0f1923, #0a0f1a); color: #e2e8f0; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.03); border-right: 1px solid rgba(255,255,255,0.07); }
[data-testid="metric-container"] { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 18px; }
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; color: #38bdf8 !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
.hero { background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(168,85,247,0.1)); border: 1px solid rgba(56,189,248,0.25); border-radius: 20px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; }
.hero-title { font-size: 2rem; font-weight: 700; color: #fff; margin: 0; }
.hero-sub { color: #38bdf8; font-size: 0.9rem; margin-top: 0.3rem; }
.section-title { font-weight: 700; font-size: 0.95rem; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin: 1.8rem 0 0.8rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid rgba(56,189,248,0.2); }
.sentiment-positive { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); border-radius: 12px; padding: 16px; text-align: center; }
.sentiment-negative { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); border-radius: 12px; padding: 16px; text-align: center; }
.sentiment-neutral { background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); border-radius: 12px; padding: 16px; text-align: center; }
.feedback-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; font-size: 0.88rem; line-height: 1.6; }
.keyword-pill { display: inline-block; background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.3); border-radius: 20px; padding: 3px 12px; margin: 3px; font-size: 0.8rem; color: #38bdf8; }
.insight-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(56,189,248,0.15); border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; }
.insight-box h4 { color: #38bdf8; margin: 0 0 6px 0; font-size: 0.9rem; }
.insight-box p { color: #cbd5e1; margin: 0; font-size: 0.85rem; line-height: 1.6; }
.risk-high { border-left: 4px solid #f87171 !important; }
.risk-medium { border-left: 4px solid #fbbf24 !important; }
.risk-low { border-left: 4px solid #34d399 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="DM Sans"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
)
GRADE_COLORS = {"A+": "#34d399", "A": "#38bdf8", "B": "#fbbf24", "C": "#f97316", "F": "#f87171"}
SENT_COLORS  = {"Positive": "#34d399", "Negative": "#f87171", "Neutral": "#fbbf24"}
STOP_WORDS = {
    "the","a","an","is","it","in","on","at","to","and","or","but","of","for","with",
    "this","that","was","are","we","i","my","our","very","so","they","have","has","be",
    "been","as","by","not","no","its","from","had","he","she","me","him","her","his",
    "their","there","here","what","which","who","how","when","where","all","also","just",
    "can","will","would","should","could","do"
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def preprocess(text):
    """Clean text exactly like your reference code — lowercase, remove URLs, punctuation, stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in NLTK_STOPWORDS]
    return " ".join(words)

def get_sentiment(text):
    """VADER-based sentiment — same logic as your reference code."""
    score = sia.polarity_scores(str(text))["compound"]
    if score >= 0.05:   return "Positive", score, "😊"
    elif score <= -0.05: return "Negative", score, "😞"
    else:                return "Neutral",  score, "😐"

def extract_keywords(cleaned_text):
    """Extract meaningful words from already-cleaned text."""
    words = re.findall(r'\b[a-z]{4,}\b', cleaned_text)
    return words  # stopwords already removed in preprocess()

def assign_grade(pct):
    if pct >= 90: return "A+"
    elif pct >= 75: return "A"
    elif pct >= 60: return "B"
    elif pct >= 40: return "C"
    else: return "F"

def process_marks(df, subject_cols):
    df = df.copy()
    df["Total"]      = df[subject_cols].sum(axis=1)
    df["Percentage"] = (df["Total"] / (len(subject_cols) * 100) * 100).round(2)
    df["Grade"]      = df["Percentage"].apply(assign_grade)
    df["Status"]     = df["Percentage"].apply(lambda x: "Pass" if x >= 40 else "Fail")
    return df

def load_df(uploaded_file, default_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None

def make_wordcloud(text, colormap, title):
    if not text.strip(): return None
    wc = WordCloud(width=600, height=300, background_color=None, mode="RGBA",
                   colormap=colormap, max_words=40,
                   prefer_horizontal=0.8).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, color="#e2e8f0", fontsize=12, pad=10)
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 CHRIST University")
    st.markdown("**Student Feedback & Performance**")
    st.markdown("---")

    st.markdown("#### 🔍 Filters")
    branch_filter = st.selectbox("Branch", ["All", "CSE", "ECE", "MBA"])
    status_filter = st.radio("Status", ["All", "Pass", "Fail"], horizontal=True)

    st.markdown("---")
    st.markdown("#### 📂 Upload Data")
    uploaded_marks    = st.file_uploader("📊 Marks CSV", type=["csv"], key="marks_upload")
    uploaded_feedback = st.file_uploader("💬 Feedback CSV", type=["csv"], key="feedback_upload")

    st.markdown("---")
    st.markdown("""
    <small style='color:#64748b;'>
    📌 <b>Marks CSV:</b> Name, Branch, subject cols, Attendance<br><br>
    📌 <b>Feedback CSV:</b> Name, Branch, Course, Feedback
    </small>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
# Default paths — files sit in same folder as app.py (C3/)
MARKS_PATH    = os.path.join(os.path.dirname(__file__), "sample_data.csv")
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "feedback_data.csv")

df_raw = load_df(uploaded_marks, MARKS_PATH)
if df_raw is None:
    st.warning("⚠️ Marks CSV not found. Place `sample_data.csv` in the C3 folder or drag & drop above.")
    st.stop()

NON_SUBJECT = {"name","branch","semester","attendance","roll no","roll_no","id","status","grade","total","percentage"}
subject_cols = [c for c in df_raw.columns if c.lower() not in NON_SUBJECT and pd.api.types.is_numeric_dtype(df_raw[c])]

if not subject_cols:
    st.error("No subject columns found in Marks CSV.")
    st.stop()

has_attendance = "Attendance" in df_raw.columns
df_marks = process_marks(df_raw, subject_cols)

df_fb_raw = load_df(uploaded_feedback, FEEDBACK_PATH)
if df_fb_raw is None:
    st.warning("⚠️ Feedback CSV not found. Place `feedback_data.csv` in the C3 folder or drag & drop above.")
    st.stop()

for col in ["Name","Branch","Course","Feedback"]:
    if col not in df_fb_raw.columns:
        st.error(f"Feedback CSV missing column: `{col}`")
        st.stop()

df_fb_raw["Cleaned"]  = df_fb_raw["Feedback"].apply(preprocess)
df_fb_raw["Sentiment"], df_fb_raw["Score"], df_fb_raw["Emoji"] = zip(*df_fb_raw["Cleaned"].apply(get_sentiment))
df_fb_raw["Keywords"] = df_fb_raw["Cleaned"].apply(extract_keywords)

# ─────────────────────────────────────────────
#  APPLY FILTERS
# ─────────────────────────────────────────────
mask_m = pd.Series([True]*len(df_marks))
if branch_filter != "All": mask_m &= df_marks["Branch"] == branch_filter
if status_filter != "All": mask_m &= df_marks["Status"] == status_filter
df_f = df_marks[mask_m]

df_fb = df_fb_raw.copy()
if branch_filter != "All": df_fb = df_fb[df_fb["Branch"] == branch_filter]

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 Smart Student Feedback & Performance Dashboard</div>
    <div class="hero-sub">CHRIST (Deemed to be University) · NLP Feedback Analysis + Academic Analytics</div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "📝 Feedback Analysis", "🔗 Combined Insights"])

# ═══════════════════════════════════════════════
#  TAB 1 — PERFORMANCE
# ═══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📊 Key Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("👥 Students",  len(df_f))
    k2.metric("📈 Class Avg", f"{df_f['Percentage'].mean():.1f}%")
    k3.metric("🏆 Highest",   f"{df_f['Percentage'].max():.1f}%")
    k4.metric("✅ Pass Rate", f"{(df_f['Status']=='Pass').mean()*100:.1f}%")
    k5.metric("⚠️ At Risk",   len(df_f[df_f['Percentage'] < 40]))

    st.markdown("---")
    st.markdown('<div class="section-title">📉 Performance Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_bar = px.bar(df_f.sort_values("Percentage", ascending=False),
            x="Name", y="Percentage", color="Grade", color_discrete_map=GRADE_COLORS,
            title="Student-wise Performance", text="Percentage",
            hover_data=["Branch","Grade","Status"])
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(**PLOTLY_THEME, height=380, xaxis_tickangle=-35, title_font_size=13)
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        grade_counts = df_f["Grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade","Count"]
        fig_pie = px.pie(grade_counts, names="Grade", values="Count",
            color="Grade", color_discrete_map=GRADE_COLORS,
            title="Grade Distribution", hole=0.5)
        fig_pie.update_layout(**PLOTLY_THEME, height=380, title_font_size=13)
        st.plotly_chart(fig_pie, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        subj_avg = df_f[subject_cols].mean().reset_index()
        subj_avg.columns = ["Subject","Average"]
        fig_subj = px.bar(subj_avg.sort_values("Average"), y="Subject", x="Average",
            orientation="h", title="Subject-wise Average", color="Average",
            color_continuous_scale=["#f87171","#fbbf24","#34d399"], text="Average")
        fig_subj.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_subj.update_layout(**PLOTLY_THEME, height=320, title_font_size=13, coloraxis_showscale=False)
        st.plotly_chart(fig_subj, use_container_width=True)
    with c4:
        if has_attendance:
            fig_att = px.scatter(df_f, x="Attendance", y="Percentage",
                color="Grade", color_discrete_map=GRADE_COLORS,
                title="Attendance vs Performance", hover_data=["Name","Branch"])
            fig_att.update_layout(**PLOTLY_THEME, height=320, title_font_size=13)
            st.plotly_chart(fig_att, use_container_width=True)
        else:
            branch_avg = df_f.groupby("Branch")["Percentage"].mean().reset_index()
            fig_br = px.bar(branch_avg, x="Branch", y="Percentage",
                title="Branch-wise Average", color="Branch", text="Percentage")
            fig_br.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_br.update_layout(**PLOTLY_THEME, height=320, title_font_size=13, showlegend=False)
            st.plotly_chart(fig_br, use_container_width=True)

    st.markdown("---")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**🏆 Top 5 Students**")
        st.dataframe(df_f.nlargest(5,"Percentage")[["Name","Branch","Percentage","Grade"]], use_container_width=True, hide_index=True)
    with t2:
        st.markdown("**⚠️ At-Risk Students (< 40%)**")
        at_risk = df_f[df_f["Percentage"] < 40][["Name","Branch","Percentage","Grade"]]
        if at_risk.empty: st.success("🎉 No students below 40%!")
        else: st.dataframe(at_risk, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
#  TAB 2 — FEEDBACK ANALYSIS
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">✍️ Analyze New Feedback</div>', unsafe_allow_html=True)
    f1, f2 = st.columns([3,1])
    with f1:
        user_feedback = st.text_area("Enter student feedback here:",
            placeholder="e.g. The professor was very helpful and the course was well structured...", height=100)
    with f2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    if analyze_btn and user_feedback.strip():
        cleaned   = preprocess(user_feedback)
        sentiment, score, emoji = get_sentiment(cleaned)
        keywords  = extract_keywords(cleaned)
        color = "#34d399" if sentiment=="Positive" else "#f87171" if sentiment=="Negative" else "#fbbf24"
        sc = f"sentiment-{sentiment.lower()}"
        s1,s2,s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class="{sc}"><div style='font-size:2rem;'>{emoji}</div>
                <div style='font-size:1.2rem;font-weight:700;color:{color};'>{sentiment}</div>
                <div style='font-size:0.8rem;color:#94a3b8;'>Sentiment</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="{sc}"><div style='font-size:2rem;'>📊</div>
                <div style='font-size:1.2rem;font-weight:700;color:{color};'>{score:.2f}</div>
                <div style='font-size:0.8rem;color:#94a3b8;'>Polarity Score</div></div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="{sc}"><div style='font-size:2rem;'>🔑</div>
                <div style='font-size:1.2rem;font-weight:700;color:{color};'>{len(keywords)}</div>
                <div style='font-size:0.8rem;color:#94a3b8;'>Keywords Found</div></div>""", unsafe_allow_html=True)
        if keywords:
            st.markdown("**🔑 Extracted Keywords:**")
            st.markdown("".join([f'<span class="keyword-pill">{w}</span>' for w in set(keywords)]), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Bulk Feedback Analytics</div>', unsafe_allow_html=True)

    if df_fb.empty:
        st.info("No feedback data for selected filters.")
    else:
        sent_counts = df_fb["Sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment","Count"]
        b1, b2 = st.columns(2)
        with b1:
            fig_sent = px.pie(sent_counts, names="Sentiment", values="Count",
                color="Sentiment", color_discrete_map=SENT_COLORS,
                title="Overall Sentiment Distribution", hole=0.5)
            fig_sent.update_layout(**PLOTLY_THEME, height=350, title_font_size=13)
            st.plotly_chart(fig_sent, use_container_width=True)
        with b2:
            branch_sent = df_fb.groupby(["Branch","Sentiment"]).size().reset_index(name="Count")
            fig_bsent = px.bar(branch_sent, x="Branch", y="Count", color="Sentiment",
                color_discrete_map=SENT_COLORS, title="Sentiment by Branch", barmode="group")
            fig_bsent.update_layout(**PLOTLY_THEME, height=350, title_font_size=13)
            st.plotly_chart(fig_bsent, use_container_width=True)

        st.markdown('<div class="section-title">☁️ Word Cloud</div>', unsafe_allow_html=True)
        pos_text = " ".join(df_fb[df_fb["Sentiment"]=="Positive"]["Cleaned"].tolist())
        neg_text = " ".join(df_fb[df_fb["Sentiment"]=="Negative"]["Cleaned"].tolist())
        wc1, wc2 = st.columns(2)
        with wc1:
            fig_wc = make_wordcloud(pos_text, "Greens", "✅ Positive Feedback Words")
            if fig_wc: st.pyplot(fig_wc, use_container_width=True); plt.close()
        with wc2:
            fig_wc2 = make_wordcloud(neg_text, "Reds", "❌ Negative Feedback Words")
            if fig_wc2: st.pyplot(fig_wc2, use_container_width=True); plt.close()

        st.markdown('<div class="section-title">🔑 Top Keywords</div>', unsafe_allow_html=True)
        all_keywords = [kw for kws in df_fb["Keywords"] for kw in kws]
        word_freq = Counter(all_keywords).most_common(15)
        if word_freq:
            df_words = pd.DataFrame(word_freq, columns=["Word","Frequency"])
            fig_kw = px.bar(df_words, x="Frequency", y="Word", orientation="h",
                color="Frequency", color_continuous_scale=["#1e3a5f","#38bdf8"],
                title="Most Frequently Used Words in Feedback")
            fig_kw.update_layout(**PLOTLY_THEME, height=420, title_font_size=13, coloraxis_showscale=False)
            st.plotly_chart(fig_kw, use_container_width=True)

        st.markdown('<div class="section-title">💬 Individual Feedback</div>', unsafe_allow_html=True)
        sent_filter = st.selectbox("Filter by Sentiment", ["All","Positive","Negative","Neutral"])
        shown_fb = df_fb if sent_filter=="All" else df_fb[df_fb["Sentiment"]==sent_filter]
        for _, row in shown_fb.iterrows():
            color = "#34d399" if row["Sentiment"]=="Positive" else "#f87171" if row["Sentiment"]=="Negative" else "#fbbf24"
            st.markdown(f"""
            <div class="feedback-card">
                <b>{row['Name']}</b> · <span style='color:#64748b;'>{row['Branch']} · {row['Course']}</span>
                <span style='float:right;color:{color};font-weight:600;'>{row['Emoji']} {row['Sentiment']} ({row['Score']:.2f})</span><br>
                <span style='color:#cbd5e1;'>"{row['Feedback']}"</span>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  TAB 3 — COMBINED INSIGHTS
# ═══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🔗 Merging Academic Performance + Feedback Sentiment</div>', unsafe_allow_html=True)

    df_combined = df_marks.merge(
        df_fb_raw[["Name","Branch","Course","Sentiment","Score","Feedback"]].rename(columns={"Score":"Sentiment_Score"}),
        on="Name", how="inner"
    )
    # Override Branch with marks branch if duplicate
    if "Branch_x" in df_combined.columns:
        df_combined["Branch"] = df_combined["Branch_x"]
        df_combined.drop(columns=["Branch_x","Branch_y"], inplace=True)

    if branch_filter != "All":
        df_combined = df_combined[df_combined["Branch"] == branch_filter]

    if df_combined.empty:
        st.info("No matching records between marks and feedback. Ensure student names match in both CSVs.")
    else:
        # ── SUMMARY METRICS ──────────────────────────────
        pos_avg = df_combined[df_combined["Sentiment"]=="Positive"]["Percentage"].mean() if "Positive" in df_combined["Sentiment"].values else 0
        neg_avg = df_combined[df_combined["Sentiment"]=="Negative"]["Percentage"].mean() if "Negative" in df_combined["Sentiment"].values else 0
        neu_avg = df_combined[df_combined["Sentiment"]=="Neutral"]["Percentage"].mean()  if "Neutral"  in df_combined["Sentiment"].values else 0
        diff    = pos_avg - neg_avg

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("😊 Positive Avg Score", f"{pos_avg:.1f}%")
        m2.metric("😐 Neutral Avg Score",  f"{neu_avg:.1f}%")
        m3.metric("😞 Negative Avg Score", f"{neg_avg:.1f}%")
        m4.metric("📊 Sentiment-Score Gap", f"{abs(diff):.1f}%", delta=f"{'Pos > Neg' if diff>0 else 'Neg > Pos'}")

        st.markdown("---")

        # ── CHART ROW 1 ──────────────────────────────────
        ci1, ci2 = st.columns(2)
        with ci1:
            sent_perf = df_combined.groupby("Sentiment")["Percentage"].mean().reset_index()
            sent_perf.columns = ["Sentiment","Avg Performance %"]
            fig_sp = px.bar(sent_perf, x="Sentiment", y="Avg Performance %",
                color="Sentiment", color_discrete_map=SENT_COLORS,
                title="Avg Academic Score by Sentiment Group", text="Avg Performance %")
            fig_sp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_sp.update_layout(**PLOTLY_THEME, height=360, title_font_size=13, showlegend=False)
            st.plotly_chart(fig_sp, use_container_width=True)

        with ci2:
            fig_scatter = px.scatter(df_combined, x="Sentiment_Score", y="Percentage",
                color="Sentiment", color_discrete_map=SENT_COLORS,
                hover_data=["Name","Branch","Grade","Course"],
                title="Sentiment Polarity vs Academic Percentage")
            fig_scatter.update_layout(**PLOTLY_THEME, height=360, title_font_size=13)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── CHART ROW 2 ──────────────────────────────────
        ci3, ci4 = st.columns(2)
        with ci3:
            branch_sent_perf = df_combined.groupby(["Branch","Sentiment"])["Percentage"].mean().reset_index()
            fig_bsp = px.bar(branch_sent_perf, x="Branch", y="Percentage",
                color="Sentiment", barmode="group", color_discrete_map=SENT_COLORS,
                title="Branch-wise Score split by Sentiment",
                text="Percentage")
            fig_bsp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bsp.update_layout(**PLOTLY_THEME, height=360, title_font_size=13)
            st.plotly_chart(fig_bsp, use_container_width=True)

        with ci4:
            # Attendance vs Sentiment (if available)
            if has_attendance:
                fig_att_sent = px.box(df_combined, x="Sentiment", y="Attendance",
                    color="Sentiment", color_discrete_map=SENT_COLORS,
                    title="Attendance Distribution by Sentiment Group",
                    points="all", hover_data=["Name"])
                fig_att_sent.update_layout(**PLOTLY_THEME, height=360, title_font_size=13, showlegend=False)
                st.plotly_chart(fig_att_sent, use_container_width=True)
            else:
                grade_sent = df_combined.groupby(["Grade","Sentiment"]).size().reset_index(name="Count")
                fig_gs = px.bar(grade_sent, x="Grade", y="Count", color="Sentiment",
                    color_discrete_map=SENT_COLORS, barmode="stack",
                    title="Grade vs Sentiment Distribution")
                fig_gs.update_layout(**PLOTLY_THEME, height=360, title_font_size=13)
                st.plotly_chart(fig_gs, use_container_width=True)

        # ── STUDENT-LEVEL COMBINED TABLE ─────────────────
        st.markdown('<div class="section-title">📋 Student-Level Combined View</div>', unsafe_allow_html=True)

        display_cols = ["Name","Branch","Percentage","Grade","Status","Sentiment","Course","Feedback"]
        display_cols = [c for c in display_cols if c in df_combined.columns]
        df_display = df_combined[display_cols].copy()

        def color_sentiment(val):
            if val == "Positive": return "color: #34d399; font-weight:600"
            elif val == "Negative": return "color: #f87171; font-weight:600"
            return "color: #fbbf24; font-weight:600"

        st.dataframe(
            df_display.sort_values("Percentage", ascending=False).style.applymap(color_sentiment, subset=["Sentiment"]),
            use_container_width=True, hide_index=True
        )

        # ── ACTIONABLE INSIGHTS ───────────────────────────
        st.markdown('<div class="section-title">💡 Actionable Insights & Judgements</div>', unsafe_allow_html=True)

        # Insight 1: Sentiment-Performance correlation
        if diff > 10:
            label, risk_cls, icon = "Strong Positive Correlation Detected", "risk-low", "✅"
            desc = f"Students with positive feedback score <b>{diff:.1f}% higher</b> than those with negative feedback. This strongly suggests that <b>course satisfaction directly drives academic performance</b>. Faculty engagement and clarity of teaching are likely key factors."
        elif diff > 0:
            label, risk_cls, icon = "Moderate Correlation Detected", "risk-medium", "⚠️"
            desc = f"Students with positive feedback score <b>{diff:.1f}% higher</b> on average. There is a moderate link between satisfaction and performance — <b>improving teaching methods could lift grades across the board</b>."
        else:
            label, risk_cls, icon = "Negative Correlation — Investigate", "risk-high", "🚨"
            desc = "Students with negative feedback are scoring higher. This may indicate <b>high-pressure courses with poor delivery</b> — students may be performing despite dissatisfaction, suggesting a systemic issue worth investigating."
        st.markdown(f'<div class="insight-box {risk_cls}"><h4>{icon} {label}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

        # Insight 2: At-risk students with negative sentiment
        neg_fail = df_combined[(df_combined["Sentiment"]=="Negative") & (df_combined["Status"]=="Fail")]
        if not neg_fail.empty:
            names = ", ".join(neg_fail["Name"].tolist())
            st.markdown(f"""<div class="insight-box risk-high">
                <h4>🚨 High-Priority Students: Failing + Negative Feedback</h4>
                <p><b>{len(neg_fail)} student(s)</b> are both failing academically AND expressing negative feedback: <b>{names}</b>.<br>
                These students need <b>immediate academic counselling and faculty intervention</b>. Dual distress (academic + emotional) greatly increases dropout risk.</p>
            </div>""", unsafe_allow_html=True)

        # Insight 3: Positive feedback but failing (hidden struggle)
        pos_fail = df_combined[(df_combined["Sentiment"]=="Positive") & (df_combined["Status"]=="Fail")]
        if not pos_fail.empty:
            names2 = ", ".join(pos_fail["Name"].tolist())
            st.markdown(f"""<div class="insight-box risk-medium">
                <h4>⚠️ Hidden Struggle: Positive Feedback but Failing</h4>
                <p><b>{len(pos_fail)} student(s)</b> gave positive feedback yet are academically failing: <b>{names2}</b>.<br>
                These students may enjoy the course but face <b>conceptual gaps or exam anxiety</b>. Targeted tutoring or additional practice tests are recommended.</p>
            </div>""", unsafe_allow_html=True)

        # Insight 4: Top performers with negative sentiment
        neg_top = df_combined[(df_combined["Sentiment"]=="Negative") & (df_combined["Percentage"] >= 75)]
        if not neg_top.empty:
            names3 = ", ".join(neg_top["Name"].tolist())
            st.markdown(f"""<div class="insight-box risk-medium">
                <h4>⚠️ High Achievers — Dissatisfied</h4>
                <p><b>{len(neg_top)} high-scoring student(s)</b> gave negative feedback: <b>{names3}</b>.<br>
                Strong students who are dissatisfied often indicate <b>lack of challenge, poor delivery, or irrelevant content</b>. Risk of disengagement — consider advanced content or mentorship roles.</p>
            </div>""", unsafe_allow_html=True)

        # Insight 5: Branch-level judgement
        branch_insight = df_combined.groupby("Branch").agg(
            Avg_Score=("Percentage","mean"),
            Neg_Pct=("Sentiment", lambda x: (x=="Negative").mean()*100)
        ).reset_index()
        worst_branch = branch_insight.loc[branch_insight["Neg_Pct"].idxmax()]
        best_branch  = branch_insight.loc[branch_insight["Avg_Score"].idxmax()]

        st.markdown(f"""<div class="insight-box risk-low">
            <h4>🏆 Branch-Level Judgement</h4>
            <p>
            <b>{best_branch['Branch']}</b> has the highest average academic score ({best_branch['Avg_Score']:.1f}%) — indicating strong faculty–student alignment.<br><br>
            <b>{worst_branch['Branch']}</b> has the highest negative feedback rate ({worst_branch['Neg_Pct']:.0f}% of its students) — 
            faculty in this branch should review their <b>teaching pace, content relevance, and student engagement strategies</b>.
            </p>
        </div>""", unsafe_allow_html=True)

        # Insight 6: Overall recommendation
        overall_neg_pct = (df_combined["Sentiment"]=="Negative").mean()*100
        overall_pass_pct = (df_combined["Status"]=="Pass").mean()*100
        st.markdown(f"""<div class="insight-box">
            <h4>📌 Overall Dashboard Summary</h4>
            <p>
            Across <b>{len(df_combined)} matched students</b>: <b>{overall_pass_pct:.0f}%</b> are passing academically, 
            and <b>{100-overall_neg_pct:.0f}%</b> gave neutral or positive feedback.<br><br>
            The data suggests the university can improve outcomes by addressing courses with high negative sentiment — 
            even small improvements in teaching quality and student satisfaction tend to produce measurable gains in exam performance.
            </p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.8rem;padding:1rem 0;'>
    🎓 Smart Student Feedback & Performance Dashboard · CHRIST (Deemed to be University) · Built with Streamlit, TextBlob & Plotly
</div>""", unsafe_allow_html=True)