import streamlit as st
import datetime

def render_html(html_str: str):
    """Safely renders HTML by removing leading indentation to prevent markdown verbatim code blocks."""
    clean_lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    clean_html = "".join(clean_lines)
    st.markdown(clean_html, unsafe_allow_html=True)

def render_sidebar():
    """Renders the custom TrainIQ dark navy sidebar and returns the active page."""
    with st.sidebar:
        # Custom Branding Header
        render_html("""
        <div class="tiq-sidebar-brand">
            <div class="tiq-sidebar-logo-row">
                <div class="tiq-sidebar-icon">⚡</div>
                <div>
                    <div class="tiq-sidebar-title">TrainIQ</div>
                    <div class="tiq-sidebar-subtitle">TRAINING INTELLIGENCE</div>
                </div>
            </div>
        </div>
        """)
        
        # Navigation Items with modern icons
        nav_options = [
            "📊 Overview",
            "📈 Analytics",
            "⚡ Employee Prediction",
            "🧠 Model Insights",
            "🗄️ Dataset Explorer",
            "ℹ️ About Project"
        ]
        
        # Initialize session state for navigation
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = nav_options[0]
            
        selected = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state.selected_page) if st.session_state.selected_page in nav_options else 0,
            key="main_nav_radio",
            label_visibility="collapsed"
        )
        
        st.session_state.selected_page = selected
        
        # Sidebar Footer: Model status and user profile
        render_html("""
        <div class="tiq-sidebar-footer">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">ML Pipeline</span>
                <span class="tiq-badge tiq-badge-green" style="font-size: 0.7rem; padding: 2px 8px;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: #20A779; display: inline-block;"></span> Active
                </span>
            </div>
            <div style="font-size: 0.78rem; color: #CBD5E1; margin-bottom: 16px; background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="font-weight: 600; color: #FFFFFF;">Random Forest v2.4</div>
                <div style="color: #64748B; font-size: 0.72rem;">Case Study 88 • 80:20 Stratified</div>
            </div>
            
            <div class="tiq-user-pill">
                <div class="tiq-user-avatar">DR</div>
                <div style="overflow: hidden;">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #FFFFFF; text-overflow: ellipsis; white-space: nowrap;">Dr. Ralliyah</div>
                    <div style="font-size: 0.7rem; color: #94A3B8;">People Analytics Lead</div>
                </div>
            </div>
        </div>
        """)
        
        return selected

def render_page_header(title: str, subtitle: str, badge_text: str = None, badge_type: str = "blue"):
    """Renders a clean, modern page header."""
    badge_html = f'<span class="tiq-badge tiq-badge-{badge_type}">{badge_text}</span>' if badge_text else ""
    render_html(f"""
    <div style="margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <h1 style="font-size: 1.85rem; font-weight: 800; color: #101828; margin: 0; letter-spacing: -0.03em;">{title}</h1>
                {badge_html}
            </div>
            <p style="font-size: 0.95rem; color: #64748B; margin: 6px 0 0 0; line-height: 1.5;">{subtitle}</p>
        </div>
        <div style="display: flex; align-items: center; gap: 10px; color: #64748B; font-size: 0.82rem; background: #FFFFFF; padding: 6px 14px; border-radius: 8px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <span>📅 {datetime.date.today().strftime("%B %d, %Y")}</span>
        </div>
    </div>
    """)

def render_metric_card(icon: str, label: str, value: str, subtext: str = "", accent_color: str = "#3563E9", icon_bg: str = "#EDF4FF"):
    """Renders an enterprise-grade metric card."""
    render_html(f"""
    <div class="tiq-metric-card" style="border-top: 3px solid {accent_color};">
        <div>
            <div class="tiq-metric-header">
                <span class="tiq-metric-label">{label}</span>
                <div class="tiq-metric-icon" style="background: {icon_bg}; color: {accent_color};">
                    {icon}
                </div>
            </div>
            <div class="tiq-metric-value">{value}</div>
        </div>
        <div class="tiq-metric-subtext">
            {subtext}
        </div>
    </div>
    """)

def render_hero_banner():
    """Renders the executive hero banner on the overview dashboard."""
    render_html("""
    <div class="tiq-hero">
        <div style="position: relative; z-index: 2;">
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: rgba(53, 99, 233, 0.3); border: 1px solid rgba(92, 131, 255, 0.4); border-radius: 9999px; font-size: 0.75rem; font-weight: 600; color: #93C5FD; margin-bottom: 12px;">
                <span>✨</span> Enterprise Training Intelligence
            </div>
            <div class="tiq-hero-title">Optimize Organizational Learning ROI</div>
            <div class="tiq-hero-subtitle">
                TrainIQ evaluates the complex relationship between employee characteristics, training investment, and post-training outcomes to drive smarter human capital decisions.
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 10px; font-size: 0.85rem; color: #FFFFFF;">
                    🎯 <strong>Problem No. 88</strong> • Training Effectiveness
                </div>
                <div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 10px; font-size: 0.85rem; color: #FFFFFF;">
                    🛡️ <strong>Zero Data Leakage</strong> • Verified Pipeline
                </div>
            </div>
        </div>
    </div>
    """)

def render_insight_card(title: str, points: list, icon: str = "💡"):
    """Renders a highlighted insight panel."""
    points_html = "".join([f"<li style='margin-bottom: 6px;'>{p}</li>" for p in points])
    render_html(f"""
    <div class="tiq-insight-card">
        <div class="tiq-insight-title">
            <span>{icon}</span> {title}
        </div>
        <ul class="tiq-insight-text" style="padding-left: 20px; margin-bottom: 0;">
            {points_html}
        </ul>
    </div>
    """)

def render_prediction_card(outcome: str, conf: float, p_needs: float, p_succ: float, model_name: str):
    """Renders the outcome prediction card using safe, perfectly formatted layout."""
    is_success = (outcome == "Successful")
    badge_bg = "#DCFCE7" if is_success else "#FEF3C7"
    badge_color = "#15803D" if is_success else "#B45309"
    badge_border = "#86EFAC" if is_success else "#FCD34D"
    icon = "✨" if is_success else "🎯"
    
    recommendation = (
        "The model projects that this employee is likely to achieve an Outstanding post-training evaluation rating (Rating 4)."
        if is_success else
        "The model projects a baseline performance outcome (Rating 3). Customized post-training mentoring or follow-up workshops are recommended to maximize learning ROI."
    )
    
    with st.container(border=True):
        # Top Header Badge
        render_html(f"""
        <div style="text-align: center; margin-bottom: 12px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Predicted Outcome</div>
            <div style="display: inline-block; padding: 8px 24px; background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; border-radius: 9999px; font-size: 1.15rem; font-weight: 800; letter-spacing: 0.02em;">
                {icon} {outcome}
            </div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #101828; margin-top: 10px;">
                Confidence: <span style="color: #3563E9;">{conf:.1f}%</span>
            </div>
            <div style="font-size: 0.88rem; color: #475467; line-height: 1.5; margin-top: 8px; margin-bottom: 16px;">
                {recommendation}
            </div>
        </div>
        """)
        
        # Probability Breakdown using Native Progress Bars
        st.markdown("**Predicted Probability Breakdown**")
        
        col_lbl1, col_val1 = st.columns([3, 1])
        col_lbl1.caption("Needs Improvement (Class 0)")
        col_val1.markdown(f"<div style='text-align: right; font-weight: 700; font-size: 0.85rem; color: #3563E9;'>{p_needs:.1f}%</div>", unsafe_allow_html=True)
        st.progress(min(max(p_needs / 100.0, 0.0), 1.0))
        
        col_lbl2, col_val2 = st.columns([3, 1])
        col_lbl2.caption("Successful (Class 1)")
        col_val2.markdown(f"<div style='text-align: right; font-weight: 700; font-size: 0.85rem; color: #20A779;'>{p_succ:.1f}%</div>", unsafe_allow_html=True)
        st.progress(min(max(p_succ / 100.0, 0.0), 1.0))
        
        # Card Footer
        render_html(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; color: #94A3B8; border-top: 1px solid #F1F5F9; padding-top: 12px; margin-top: 16px;">
            <span>Evaluator: <strong>{model_name}</strong></span>
            <span>Status: <strong style="color: #20A779;">Live Real-Time</strong></span>
        </div>
        """)
