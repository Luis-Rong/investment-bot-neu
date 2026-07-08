"""Product chrome: base CSS, app header, and sidebar.

Kept in one place so `app.py` stays about flow, not styling. Colors follow a
validated, colorblind-safe palette (see ui/recommendation.py for the series
colors used in the allocation chart).
"""

import streamlit as st

BASE_CSS = """
<style>
/* Hide Streamlit's default chrome so it reads as a product, not a script */
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

.block-container {padding-top: 2.2rem; max-width: 780px;}

/* Brand header */
.mer-header {
    display: flex; align-items: baseline; gap: 0.6rem;
    margin-bottom: 0.15rem;
}
.mer-mark {color: #2a78d6; font-size: 1.5rem; line-height: 1;}
.mer-wordmark {
    font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em;
    color: #0b0b0b;
}
.mer-tagline {color: #52514e; font-size: 0.95rem; margin: 0 0 0.6rem 0;}
.mer-badges {display: flex; gap: 0.4rem; margin-bottom: 1.2rem;}
.mer-badge {
    font-size: 0.72rem; font-weight: 600; color: #52514e;
    background: #f0efec; border: 1px solid rgba(11,11,11,0.08);
    border-radius: 999px; padding: 0.15rem 0.6rem;
}
.mer-badge.on {color: #006300; background: #eaf5ea; border-color: rgba(0,99,0,0.15);}

/* Generic card */
.mer-card {
    background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
    border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.75rem;
}

/* Stat tiles */
.mer-tiles {display: flex; gap: 0.75rem; margin-bottom: 0.75rem;}
.mer-tile {
    flex: 1; background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
    border-radius: 12px; padding: 0.8rem 1rem;
}
.mer-tile .label {font-size: 0.72rem; font-weight: 600; color: #898781;
    text-transform: uppercase; letter-spacing: 0.05em;}
.mer-tile .value {font-size: 1.35rem; font-weight: 700; color: #0b0b0b; margin-top: 0.1rem;}
.mer-tile .sub {font-size: 0.78rem; color: #52514e;}

/* Allocation stacked bar */
.mer-alloc-bar {
    display: flex; width: 100%; height: 34px; border-radius: 8px;
    overflow: hidden; gap: 2px; background: #fcfcfb; margin: 0.4rem 0 0.7rem 0;
}
.mer-alloc-seg {height: 100%;}
.mer-legend-row {
    display: flex; align-items: center; gap: 0.55rem;
    padding: 0.45rem 0.2rem; border-bottom: 1px solid #e1e0d9;
    font-size: 0.9rem; color: #0b0b0b;
}
.mer-legend-row:last-child {border-bottom: none;}
.mer-dot {width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0;}
.mer-legend-pct {font-weight: 700; min-width: 3rem;}
.mer-legend-name {flex: 1;}
.mer-legend-meta {color: #898781; font-size: 0.78rem; font-variant-numeric: tabular-nums;}

/* Section headings */
.mer-section {
    font-size: 0.78rem; font-weight: 700; color: #898781;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 1.3rem 0 0.5rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] .mer-side-title {
    font-weight: 700; font-size: 1.05rem; color: #0b0b0b; margin-bottom: 0.3rem;
}
.mer-disclaimer {
    font-size: 0.75rem; color: #52514e; background: #f0efec;
    border-radius: 8px; padding: 0.6rem 0.7rem; line-height: 1.45;
}
</style>
"""


def inject_base_css():
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def render_app_header():
    st.markdown(
        """
        <div class="mer-header">
            <span class="mer-mark">&#9670;</span>
            <span class="mer-wordmark">Meridian</span>
        </div>
        <p class="mer-tagline">AI portfolio advisor &mdash; live market data, cited fund facts.</p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(n_funds: int, grounded: bool):
    with st.sidebar:
        st.markdown('<div class="mer-side-title">&#9670; Meridian</div>', unsafe_allow_html=True)
        st.caption("Chat-based robo-advisor prototype")

        grounded_badge = (
            '<span class="mer-badge on">Citations: grounded</span>'
            if grounded
            else '<span class="mer-badge">Citations: off &mdash; run rag pipeline</span>'
        )
        st.markdown(
            f"""
            <div class="mer-badges" style="flex-direction: column; align-items: flex-start;">
                <span class="mer-badge on">Live data: {n_funds} ETFs</span>
                {grounded_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**How it works**")
        st.markdown(
            "1. Chat about your goal, horizon and risk comfort\n"
            "2. Real 5-year market data scores the fund universe\n"
            "3. You get a portfolio with cited fund facts\n"
            '4. Say *"make it less risky"* to adjust anytime'
        )

        if st.button("Start over", use_container_width=True):
            for key in ("messages", "profile", "recommendation"):
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown(
            '<div class="mer-disclaimer">Prototype for demonstration purposes. '
            "Not financial advice. Past performance does not predict future "
            "returns.</div>",
            unsafe_allow_html=True,
        )
