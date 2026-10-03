"""
TrainIQ Design System and UI Styles
Color Palette:
- Deep Navy: #101828
- Primary Blue: #3563E9
- Electric Blue: #5C83FF
- Pale Blue: #EDF4FF
- Off-White: #F8FAFC
- Card White: #FFFFFF
- Lavender Accent: #8B7CFF
- Success Green: #20A779
- Warning Amber: #F3AA38
- Slate Text: #475467
- Light Slate: #94A3B8
"""

import plotly.graph_objects as go
import plotly.io as pio

# Custom Streamlit CSS
CUSTOM_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* Global Reset and Typography */
html, body, [class*="css"], .stMarkdown, .stText, p, span, label, div {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Background */
.stApp {
    background-color: #F8FAFC;
    color: #101828;
}

/* Hide default streamlit header decoration */
header[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(226, 232, 240, 0.8);
    height: 3.5rem;
}

/* Hide default hamburger menu footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #F1F5F9;
}
::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}

/* SIDEBAR STYLING - Premium Dark Navy */
section[data-testid="stSidebar"] {
    background-color: #101828 !important;
    border-right: 1px solid #1E293B;
    width: 300px !important;
}

section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    background-color: #101828 !important;
}

section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] span, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
    color: #E2E8F0 !important;
}

/* Sidebar Radio Navigation Item Styling */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
    background: transparent;
    gap: 6px;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 4px;
    color: #94A3B8 !important;
    font-weight: 500;
    font-size: 0.92rem;
    transition: all 0.2s ease;
    cursor: pointer;
    display: flex;
    align-items: center;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
    background: rgba(53, 99, 233, 0.15);
    color: #FFFFFF !important;
    border-color: rgba(92, 131, 255, 0.3);
    transform: translateX(3px);
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[aria-checked="true"] {
    background: linear-gradient(135deg, rgba(53, 99, 233, 0.25) 0%, rgba(92, 131, 255, 0.15) 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #3563E9 !important;
    box-shadow: 0 4px 12px rgba(53, 99, 233, 0.25);
    font-weight: 600 !important;
}

/* Hide the circular radio input dot in sidebar */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none;
}

/* CARDS AND CONTAINERS */
.tiq-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px -2px rgba(16, 24, 40, 0.04), 0 2px 6px -1px rgba(16, 24, 40, 0.02);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.tiq-card:hover {
    box-shadow: 0 12px 28px -4px rgba(16, 24, 40, 0.08), 0 4px 12px -2px rgba(16, 24, 40, 0.03);
    border-color: #CBD5E1;
}

/* METRIC CARDS */
.tiq-metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(16, 24, 40, 0.03);
    transition: all 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.tiq-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(53, 99, 233, 0.08);
    border-color: #3563E9;
}

.tiq-metric-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.tiq-metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
}

.tiq-metric-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.tiq-metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #101828;
    line-height: 1.2;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.tiq-metric-subtext {
    font-size: 0.8rem;
    color: #64748B;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* HERO BANNER */
.tiq-hero {
    background: linear-gradient(135deg, #101828 0%, #1E293B 50%, #1E3A8A 100%);
    border-radius: 20px;
    padding: 36px 40px;
    color: #FFFFFF;
    margin-bottom: 28px;
    box-shadow: 0 20px 40px -10px rgba(16, 24, 40, 0.3);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.tiq-hero::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 450px;
    height: 450px;
    background: radial-gradient(circle, rgba(92, 131, 255, 0.25) 0%, rgba(53, 99, 233, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.tiq-hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
    color: #FFFFFF;
}

.tiq-hero-subtitle {
    font-size: 1.05rem;
    color: #CBD5E1;
    max-width: 650px;
    line-height: 1.6;
    margin-bottom: 20px;
}

.tiq-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.tiq-badge-blue {
    background: rgba(53, 99, 233, 0.15);
    color: #5C83FF;
    border: 1px solid rgba(92, 131, 255, 0.3);
}

.tiq-badge-green {
    background: rgba(32, 167, 121, 0.15);
    color: #20A779;
    border: 1px solid rgba(32, 167, 121, 0.3);
}

.tiq-badge-amber {
    background: rgba(243, 170, 56, 0.15);
    color: #F3AA38;
    border: 1px solid rgba(243, 170, 56, 0.3);
}

.tiq-badge-lavender {
    background: rgba(139, 124, 255, 0.15);
    color: #8B7CFF;
    border: 1px solid rgba(139, 124, 255, 0.3);
}

/* SECTION HEADERS */
.tiq-section-header {
    margin-top: 10px;
    margin-bottom: 20px;
}

.tiq-section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #101828;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
}

.tiq-section-subtitle {
    font-size: 0.88rem;
    color: #64748B;
    margin-top: 4px;
}

/* INSIGHT BOX */
.tiq-insight-card {
    background: #FFFFFF;
    border-left: 4px solid #3563E9;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 4px 16px rgba(16, 24, 40, 0.04);
    margin-top: 20px;
    border-top: 1px solid #F1F5F9;
    border-right: 1px solid #F1F5F9;
    border-bottom: 1px solid #F1F5F9;
}

.tiq-insight-title {
    font-size: 1rem;
    font-weight: 700;
    color: #101828;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.tiq-insight-text {
    font-size: 0.88rem;
    color: #475467;
    line-height: 1.6;
}

/* FORM ELEMENTS STYLING */
div[data-baseweb="input"] {
    border-radius: 10px !important;
    border-color: #E2E8F0 !important;
}

div[data-baseweb="select"] {
    border-radius: 10px !important;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 24px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3563E9 0%, #1E40AF 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(53, 99, 233, 0.35) !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(53, 99, 233, 0.5) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #475467 !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #F8FAFC !important;
    border-color: #CBD5E1 !important;
    color: #101828 !important;
}

/* RESULT CARD */
.tiq-result-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 30px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 30px -5px rgba(16, 24, 40, 0.06);
    text-align: center;
}

.tiq-result-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 9999px;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: 16px;
}

.tiq-result-badge.success {
    background: #DCFCE7;
    color: #15803D;
    border: 1px solid #86EFAC;
}

.tiq-result-badge.needs-improvement {
    background: #FEF3C7;
    color: #B45309;
    border: 1px solid #FCD34D;
}

/* TABLES */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
}

/* SIDEBAR FOOTER & BRANDING */
.tiq-sidebar-brand {
    padding: 0 8px 24px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 20px;
}

.tiq-sidebar-logo-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.tiq-sidebar-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #3563E9 0%, #5C83FF 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(53, 99, 233, 0.4);
}

.tiq-sidebar-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.tiq-sidebar-subtitle {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #5C83FF;
    text-transform: uppercase;
}

.tiq-sidebar-footer {
    margin-top: 30px;
    padding-top: 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.tiq-user-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.tiq-user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #8B7CFF 0%, #3563E9 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
    color: white;
}
</style>
"""

# Plotly Theme Settings
PLOTLY_LAYOUT_TEMPLATE = dict(
    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12, color="#475467"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#101828",
        font_size=12,
        font_family="Plus Jakarta Sans, sans-serif",
        font_color="#FFFFFF",
        bordercolor="#3563E9"
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11)
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11)
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#475467", size=11)
    )
)

COLOR_PALETTE = {
    "primary": "#3563E9",
    "electric": "#5C83FF",
    "success": "#20A779",
    "amber": "#F3AA38",
    "lavender": "#8B7CFF",
    "navy": "#101828",
    "pale_blue": "#EDF4FF",
    "slate": "#64748B",
    "outcome_colors": {
        "Successful": "#20A779",
        "Needs Improvement": "#3563E9"
    }
}
