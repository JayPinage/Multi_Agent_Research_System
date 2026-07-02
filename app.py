import streamlit as st
import time
from agents import build_scrape_agent, build_search_agent, writer_chain, critic_chain

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:            #0a0d12;
    --surface:       rgba(255,255,255,0.03);
    --surface-hi:    rgba(255,255,255,0.05);
    --border:        rgba(255,255,255,0.08);
    --teal:          #5EEAD4;
    --teal-dim:      rgba(94,234,212,0.16);
    --amber:         #FFB454;
    --amber-dim:     rgba(255,180,84,0.16);
    --slate:         #4B5563;
    --text:          #E7EAF0;
    --text-muted:    #8891A0;
    --text-faint:    #4B5568;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 70% 45% at 15% -8%, rgba(94,234,212,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 55% 35% at 85% 105%, rgba(255,180,84,0.06) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1180px; }

/* ── Hero ── */
.hero { padding: 3rem 0 2.2rem; }
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--teal);
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 1.1rem;
}
.hero-eyebrow::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 8px 1px var(--teal);
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.6rem, 5.2vw, 4.4rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: var(--text);
    margin: 0 0 1rem;
}
.hero h1 span {
    background: linear-gradient(120deg, var(--teal), #8FF3E0);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero-sub {
    font-size: 1.02rem;
    font-weight: 300;
    color: var(--text-muted);
    max-width: 560px;
    line-height: 1.7;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ── Input card ── */
.input-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.2rem;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-dim) !important;
}
.stTextInput > label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--teal) !important;
    font-weight: 500 !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--teal) 0%, #35C4AE 100%) !important;
    color: #05100E !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.8rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 18px rgba(94,234,212,0.22) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(94,234,212,0.32) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* secondary/chip-style buttons (example queries) */
div[data-testid="column"] .stButton > button.chip,
.chip-btn button {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 0.9rem !important;
    width: auto !important;
}
.chip-btn button:hover {
    border-color: var(--teal) !important;
    color: var(--teal) !important;
    transform: none !important;
}

.examples-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-faint);
    letter-spacing: 0.15em;
    margin: 0.4rem 0 0.6rem;
}

/* ── Signal rail (pipeline) ── */
.rail-row {
    display: flex;
    align-items: stretch;
    gap: 1rem;
}
.rail-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 20px;
    flex-shrink: 0;
}
.rail-node {
    width: 13px; height: 13px;
    border-radius: 50%;
    background: var(--slate);
    border: 2px solid var(--bg);
    box-shadow: 0 0 0 1px var(--border);
    margin-top: 1.4rem;
    flex-shrink: 0;
    transition: background 0.3s, box-shadow 0.3s;
}
.rail-node.running {
    background: var(--amber);
    box-shadow: 0 0 0 1px var(--amber), 0 0 10px 2px rgba(255,180,84,0.55);
    animation: pulse 1.4s ease-in-out infinite;
}
.rail-node.done {
    background: var(--teal);
    box-shadow: 0 0 0 1px var(--teal), 0 0 8px 1px rgba(94,234,212,0.5);
}
@keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.25); } }

.rail-line {
    width: 2px;
    flex-grow: 1;
    min-height: 14px;
    background: var(--border);
    margin: 2px 0;
}
.rail-line.done { background: var(--teal); }

.step-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.3s, background 0.3s;
}
.step-card.active { border-color: rgba(255,180,84,0.4); background: var(--amber-dim); }
.step-card.done    { border-color: rgba(94,234,212,0.35); background: var(--teal-dim); }

.step-header { display: flex; align-items: center; gap: 0.7rem; }
.step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    color: var(--text-faint);
}
.step-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.94rem;
    font-weight: 700;
    color: var(--text);
}
.step-status {
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
}
.status-waiting { color: var(--text-faint); }
.status-running  { color: var(--amber); }
.status-done     { color: var(--teal); }
.step-desc { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem; }

/* ── Result panels ── */
.result-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.7rem;
    margin-top: 0.5rem;
}
.result-panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.9rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.result-content {
    font-size: 0.9rem;
    line-height: 1.75;
    color: #C7CCD6;
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

.report-panel, .feedback-panel {
    background: rgba(255,255,255,0.02);
    border-radius: 14px;
    padding: 1.9rem 2.3rem;
    margin-top: 1rem;
}
.report-panel { border: 1px solid rgba(94,234,212,0.25); }
.feedback-panel { border: 1px solid rgba(255,180,84,0.25); }

.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    padding-bottom: 0.6rem;
}
.panel-label.teal  { color: var(--teal); border-bottom: 1px solid rgba(94,234,212,0.2); }
.panel-label.amber { color: var(--amber); border-bottom: 1px solid rgba(255,180,84,0.2); }

.stSpinner > div { color: var(--teal) !important; }

details summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.73rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em !important;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-heading::before {
    content: '';
    width: 3px; height: 1.1rem;
    background: var(--teal);
    border-radius: 2px;
}

.notice {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-faint);
    text-align: center;
    margin-top: 3rem;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: render a rail row (node + connector + card) ────────────────────
def rail_step(num: str, title: str, state: str, desc: str, is_last: bool):
    status_map = {
        "waiting": ("STANDBY", "status-waiting", ""),
        "running": ("● LIVE", "status-running", "running"),
        "done":    ("✓ COMPLETE", "status-done", "done"),
    }
    label, status_cls, node_cls = status_map.get(state, ("", "", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    line_cls = "done" if state == "done" else ""

    line_html = "" if is_last else f'<div class="rail-line {line_cls}"></div>'
    html = (
        f'<div class="rail-row">'
        f'<div class="rail-col"><div class="rail-node {node_cls}"></div>{line_html}</div>'
        f'<div class="step-card {card_cls}" style="width:100%;">'
        f'<div class="step-header">'
        f'<span class="step-num">{num}</span>'
        f'<span class="step-title">{title}</span>'
        f'<span class="step-status {status_cls}">{label}</span>'
        f'</div>'
        f'<div class="step-desc">{desc}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key in ("results", "running", "done"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else False
if "topic_input" not in st.session_state:
    st.session_state["topic_input"] = ""


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">System Online · Multi-Agent Research Pipeline</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">
        Four specialized agents work in sequence — searching, reading, writing,
        and critiquing — to turn any topic into a sourced, reviewed research report.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout: input left, pipeline right ───────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Quantum computing breakthroughs in 2025",
        key="topic_input",
        label_visibility="visible",
    )
    run_btn = st.button("Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    def _set_topic(value):
        st.session_state["topic_input"] = value

    st.markdown('<div class="examples-label">TRY AN EXAMPLE →</div>', unsafe_allow_html=True)
    examples = ["LLM agents 2025", "CRISPR gene editing", "Fusion energy progress"]
    chip_cols = st.columns(len(examples))
    for c, ex in zip(chip_cols, examples):
        with c:
            st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
            st.button(ex, key=f"chip_{ex}", on_click=_set_topic, args=(ex,))
            st.markdown('</div>', unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)

    r = st.session_state.results

    def s(step):
        steps = ["search", "reader", "writer", "critic"]
        if step in r:
            return "done"
        if st.session_state.running:
            for k in steps:
                if k not in r:
                    return "running" if k == step else "waiting"
        return "waiting"

    rail_step("01", "Search Agent",  s("search"), "Gathers recent web information", False)
    rail_step("02", "Reader Agent",  s("reader"), "Scrapes & extracts deep content", False)
    rail_step("03", "Writer Chain",  s("writer"), "Drafts the full research report", False)
    rail_step("04", "Critic Chain",  s("critic"), "Reviews & scores the report", True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.rerun()

if st.session_state.running and not st.session_state.done:
    results = {}
    topic_val = st.session_state.topic_input

    # ── Step 1: Search ──
    with st.spinner("Search Agent is working…"):
        search_agent = build_search_agent()
        sr = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}")]
        })
        results["search"] = sr["messages"][-1].content
        st.session_state.results = dict(results)

    # ── Step 2: Reader ──
    with st.spinner("Reader Agent is scraping top resources…"):
        reader_agent = build_scrape_agent()
        rr = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic_val}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{results['search'][:800]}"
            )]
        })
        results["reader"] = rr["messages"][-1].content
        st.session_state.results = dict(results)

    # ── Step 3: Writer ──
    with st.spinner("Writer is drafting the report…"):
        research_combined = (
            f"SEARCH RESULTS:\n{results['search']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
        )
        results["writer"] = writer_chain.invoke({
            "topic": topic_val,
            "research": research_combined
        })
        st.session_state.results = dict(results)

    # ── Step 4: Critic ──
    with st.spinner("Critic is reviewing the report…"):
        results["critic"] = critic_chain.invoke({
            "report": results["writer"]
        })
        st.session_state.results = dict(results)

    st.session_state.running = False
    st.session_state.done = True
    st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    if "writer" in r:
        st.markdown('<div class="report-panel"><div class="panel-label teal">Final Research Report</div>',
                    unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="Download Report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in r:
        st.markdown('<div class="feedback-panel"><div class="panel-label amber">Critic Feedback</div>',
                    unsafe_allow_html=True)
        st.markdown(r["critic"])
        st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)