"""
🔱 DRISHTI Dashboard — Premium Research Edition
A Hierarchical Skill Graph Pipeline with 3-Layer Robotic Intelligence Core
"""

import streamlit as st
import pandas as pd
import json
import networkx as nx
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# TEMP DEBUG
st.cache_data.clear()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="DRISHTI — Robotic Skill Intelligence",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PREMIUM DARK THEME CSS
# =============================================================================
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%);
        color: #E8E9ED;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1629 0%, #1A1F3A 100%);
        border-right: 1px solid rgba(123, 97, 255, 0.2);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3{
        color: #00D9FF !important;
        font-weight: 700;
    }
            
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #00D9FF !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(123, 97, 255, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(123, 97, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetric"] label {
        color: #B8BFCC !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    
    /* Headers */
    h1 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #00D9FF;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #7B61FF;
        font-weight: 600;
    }
    
    /* Hero title */
    .hero-title {
        font-size: 72px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #00D9FF 0%, #7B61FF 50%, #00D9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 30px 0;
        margin-bottom: 10px;
        animation: gradient-shift 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 20px;
        color: #B8BFCC;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Info boxes */
    .info-card {
        background: linear-gradient(135deg, rgba(123, 97, 255, 0.15) 0%, rgba(0, 217, 255, 0.05) 100%);
        padding: 30px;
        border-radius: 16px;
        border-left: 4px solid #7B61FF;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(123, 97, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(123, 97, 255, 0.3);
    }
    
    .info-card h3 {
        color: #00D9FF;
        margin-bottom: 15px;
        font-size: 22px;
    }
    
    .info-card p {
        color: #D4D7E0;
        line-height: 1.7;
        font-size: 15px;
    }
    
    /* Layer cards */
    .layer-card {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(123, 97, 255, 0.1) 100%);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(0, 217, 255, 0.3);
        margin: 10px 0;
        min-height: 250px;
    }
    
    .layer-card h3 {
        color: #00D9FF;
        font-size: 20px;
        margin-bottom: 10px;
    }
    
    .layer-card h4 {
        color: #7B61FF;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(15, 22, 41, 0.6) !important;
        border: 1px solid rgba(123, 97, 255, 0.3);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7B61FF 0%, #00D9FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(123, 97, 255, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(123, 97, 255, 0.6);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00D9FF 0%, #7B61FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(26, 31, 58, 0.8);
        border: 1px solid rgba(123, 97, 255, 0.3);
        border-radius: 8px;
        color: white;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(15, 22, 41, 0.6);
        border: 1px solid rgba(123, 97, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #B8BFCC !important;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        border-color: rgba(123, 97, 255, 0.2);
        margin: 30px 0;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess {
        background: rgba(0, 217, 255, 0.1);
        border-left: 4px solid #00D9FF;
        color: #E8E9ED;
    }
    
    .stInfo {
        background: rgba(123, 97, 255, 0.1);
        border-left: 4px solid #7B61FF;
        color: #E8E9ED;
    }
    
    .stWarning {
        background: rgba(255, 183, 77, 0.1);
        border-left: 4px solid #FFB74D;
        color: #E8E9ED;
    }
    
    /* Team member cards */
    .team-card {
        background: linear-gradient(135deg, rgba(123, 97, 255, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(123, 97, 255, 0.3);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .team-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 32px rgba(123, 97, 255, 0.3);
    }
    
    .team-card h3 {
        color: #00D9FF;
        margin-bottom: 10px;
    }
    
    /* LinkedIn button */
    .linkedin-btn {
        display: inline-block;
        background: #0077B5;
        color: white;
        padding: 8px 20px;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 10px;
        transition: all 0.3s ease;
    }
    
    .linkedin-btn:hover {
        background: #005885;
        transform: translateY(-2px);
    }

    div[role="alert"] p {
    color: #00B3D1 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# =============================================================================
# DATA LOADERS (cached for speed)
# =============================================================================
@st.cache_data
def load_master_graph():
    path = OUTPUT_DIR / "skill_graphs" / "DRISHTI_master_graph.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    G = nx.DiGraph()
    for node in data["nodes"]:
        nid = node.copy().pop("id")
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node["id"], **attrs)
    for edge in data["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}
        G.add_edge(src, tgt, **attrs)
    return G, data

@st.cache_data
def load_all_enriched():
    files = list(OUTPUT_DIR.rglob("*_enriched.csv"))
    dfs = []
    for f in files:
        try:
            if f.stat().st_size > 0:
                dfs.append(pd.read_csv(f))
        except:
            continue
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_video_metadata():
    path = OUTPUT_DIR / "video_metadata.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# =============================================================================
# PLOTLY DARK THEME TEMPLATE
# =============================================================================
def get_plotly_template():
    return {
        'layout': {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#E8E9ED', 'family': 'Inter'},
            'xaxis': {
                'gridcolor': 'rgba(123, 97, 255, 0.1)',
                'zerolinecolor': 'rgba(123, 97, 255, 0.2)',
            },
            'yaxis': {
                'gridcolor': 'rgba(123, 97, 255, 0.1)',
                'zerolinecolor': 'rgba(123, 97, 255, 0.2)',
            }
        }
    }

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("# 🔱 DRISHTI")
    st.markdown("*Skill Intelligence for Robotics*")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Pipeline Statistics",
            "🤲 3-Layer Intelligence",
            "🔗 Skill Graph Explorer",
            "🌐 Interactive Graph",
            "🔍 Skill Query Engine",
            "📄 Manual → Skill Graph",
            "💼 Business Vision",
            "🔮 Future Plans",
            "👥 Team & About"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    
    combined = load_all_enriched()
    G, _ = load_master_graph()
    
    if not combined.empty:
        st.metric("Videos Processed", combined["video_stem"].nunique())
        st.metric("Action Segments", len(combined))
        st.metric("Unique Robot Skills", combined["robot_skill"].nunique())
    
    if G:
        st.metric("Graph Nodes", G.number_of_nodes())
        st.metric("Graph Edges", G.number_of_edges())

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================
if page == "🏠 Overview":
    st.markdown('<div class="hero-title">🔱 DRISHTI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">A Hierarchical Skill Graph Pipeline with 3-Layer Robotic Intelligence Core</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>🌍 The Problem</h3>
        <p>Robots cannot learn from simulations alone. They need real-world, first-person human activity data — and India has zero structured pipeline for it.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>💡 Our Solution</h3>
        <p>We built an end-to-end engine that converts egocentric videos AND text manuals into a structured Hierarchical Skill Graph with a 3-Layer Intelligence Core.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
        <h3>🚀 The Impact</h3>
        <p>Robotics companies, research labs, and enterprises can directly query our graph to train robots faster, with cross-domain skill transfer built in.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🔁 Our 8-Stage Pipeline")
    
    st.code("""
    Raw Video → Ingest → Perception (Hands + Objects) → Contact Detection
                                                                ↓
    Action Segmentation → 3-Layer Intelligence Core → Skill Graph → JSON-LD Export
    """, language="text")
    
    st.markdown("---")
    st.markdown("## ⭐ The 3-Layer Intelligence Core")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="layer-card">
        <h3>🤲 KINEMATIC</h3>
        <h4>HOW the hand moves</h4>
        <p style="color: #D4D7E0;">Detects grip type (pinch, power, open), motion intensity, bimanual coordination</p>
        <p style="color: #00D9FF; font-weight: 600; margin-top: 20px;">Example: Pinch-grasp on necklace chain</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="layer-card">
        <h3>🎯 INTENT</h3>
        <h4>WHY the action is done</h4>
        <p style="color: #D4D7E0;">Combines domain + objects + kinematics to infer purpose</p>
        <p style="color: #7B61FF; font-weight: 600; margin-top: 20px;">Example: Product display arrangement</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="layer-card">
        <h3>🤖 ROBOT SKILL</h3>
        <h4>WHAT the robot must learn</h4>
        <p style="color: #D4D7E0;">Maps human action to a learnable robotic capability with DOF & complexity</p>
        <p style="color: #00D9FF; font-weight: 600; margin-top: 20px;">Example: Dexterous fine manipulation</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE: PIPELINE STATISTICS
# =============================================================================
elif page == "📊 Pipeline Statistics":
    st.title("📊 Pipeline Statistics")
    
    combined = load_all_enriched()
    
    if combined.empty:
        st.warning("No data found. Please run the pipeline first.")
    else:
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Videos", combined["video_stem"].nunique())
        with col2:
            st.metric("Total Segments", len(combined))
        with col3:
            st.metric("Unique Skills", combined["robot_skill"].nunique())
        with col4:
            st.metric("Unique Intents", combined["intent"].nunique())
        
        st.markdown("---")
        
        # Domain breakdown
        st.subheader("📂 Segments per Domain")
        dom_counts = combined.groupby("domain").size().reset_index(name="count")
        
        fig = px.bar(dom_counts, x="domain", y="count", 
                     color="domain",
                     color_discrete_map={
                         "artificial_jewellery": "#FFD700",
                         "home_cleaning": "#4CAF50",
                         "shop": "#2196F3"
                     },
                     template=get_plotly_template())
        fig.update_layout(
            showlegend=False, 
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E8E9ED')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Robot skill distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🤖 Top Robot Skills")
            skill_counts = combined["robot_skill"].value_counts().head(10).reset_index()
            skill_counts.columns = ["robot_skill", "count"]
            
            fig = px.bar(skill_counts, x="count", y="robot_skill", orientation='h',
                        color="count", 
                        color_continuous_scale=["#7B61FF", "#00D9FF"])
            fig.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E9ED')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Top Intents")
            intent_counts = combined["intent"].value_counts().head(10).reset_index()
            intent_counts.columns = ["intent", "count"]
            
            fig = px.bar(intent_counts, x="count", y="intent", orientation='h',
                        color="count", 
                        color_continuous_scale=["#00D9FF", "#7B61FF"])
            fig.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E9ED')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Skill complexity pie
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⚙️ Skill Complexity")
            comp_counts = combined["skill_complexity"].value_counts().reset_index()
            comp_counts.columns = ["complexity", "count"]
            
            fig = px.pie(comp_counts, names="complexity", values="count",
                        color_discrete_sequence=["#7B61FF", "#00D9FF", "#FFB74D"])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E9ED'),
                legend=dict(font=dict(color="white", size=14),
                bgcolor="rgba(0,0,0,0)"
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("👐 Bimanual vs Single-Hand")
            bi_counts = combined["bimanual"].value_counts().reset_index()
            bi_counts.columns = ["bimanual", "count"]
            
            fig = px.pie(bi_counts, names="bimanual", values="count",
                        color_discrete_sequence=["#00D9FF", "#7B61FF"])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E9ED'),
                legend=dict(font=dict(color="white", size=14),
                bgcolor="rgba(0,0,0,0)"
            ))
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: 3-LAYER INTELLIGENCE
# =============================================================================
elif page == "🤲 3-Layer Intelligence":
    st.title("🤲 3-Layer Intelligence Core — Live Inspector")
    st.markdown("*Pick any video and segment to see all 3 intelligence layers in action.*")
    
    combined = load_all_enriched()
    
    if combined.empty:
        st.warning("No data available.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox("Select Domain", combined["domain"].unique())
        with col2:
            videos = combined[combined["domain"] == domain]["video_stem"].unique()
            video = st.selectbox("Select Video", videos)
        
        video_segs = combined[
            (combined["domain"] == domain) & 
            (combined["video_stem"] == video)
        ].reset_index(drop=True)
        
        if not video_segs.empty:
            if len(video_segs) == 1:
                st.info("Only one segment is available for this video.")
                seg_idx = 0
            else:
                seg_idx = st.slider("Select Segment", 0, len(video_segs) - 1, 0)
            seg = video_segs.iloc[seg_idx]
            
            st.markdown("---")
            st.subheader(f"📹 {video} — Segment {seg_idx + 1}")
            st.markdown(f"⏱️ **Frames:** {seg['start_frame']} → {seg['end_frame']} (duration: {seg['duration_frames']} frames)")
            
            st.markdown("---")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="layer-card">
                <h3>🤲 KINEMATIC</h3>
                <p><b>Grasp:</b> {seg.get('grasp_used', 'N/A')}</p>
                <p><b>Motion:</b> {seg.get('motion_used', 'N/A')}</p>
                <p><b>Bimanual:</b> {seg.get('bimanual', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div class="layer-card">
                <h3>🎯 INTENT</h3>
                <p><b>Label:</b> {seg.get('intent', 'N/A')}</p>
                <p><b>Confidence:</b> {seg.get('intent_confidence', 0):.2f}</p>
                <p>{seg.get('intent_description', '')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"""
                <div class="layer-card">
                <h3>🤖 ROBOT SKILL</h3>
                <p><b>Skill:</b> {seg.get('robot_skill', 'N/A')}</p>
                <p><b>Complexity:</b> {seg.get('skill_complexity', 'N/A')}</p>
                <p><b>DOF Required:</b> {seg.get('dof_required', 'N/A')}</p>
                <p><b>Confidence:</b> {seg.get('robot_skill_confidence', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📋 Full Segment Data")
            st.dataframe(video_segs, use_container_width=True)

# =============================================================================
# PAGE: SKILL GRAPH EXPLORER
# =============================================================================
elif page == "🔗 Skill Graph Explorer":
    st.title("🔗 Hierarchical Skill Graph Explorer")
    
    G, raw_data = load_master_graph()
    
    if G is None:
        st.warning("Skill graph not found. Run the pipeline first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", G.number_of_nodes())
        with col2:
            st.metric("Total Edges", G.number_of_edges())
        with col3:
            video_roots = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "video_root")
            st.metric("Video Roots", video_roots)
        with col4:
            cross_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "cross_domain_skill_transfer")
            st.metric("Cross-Domain Links", cross_edges)
        
        st.markdown("---")
        
        # Edge type breakdown
        edge_types = {}
        for u, v, attrs in G.edges(data=True):
            et = attrs.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1
        
        st.subheader("📊 Edge Type Distribution")
        edge_df = pd.DataFrame(list(edge_types.items()), columns=["edge_type", "count"])
        
        fig = px.bar(edge_df, x="edge_type", y="count", color="edge_type",
                    color_discrete_sequence=["#7B61FF", "#00D9FF", "#FFB74D", "#4CAF50"])
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E8E9ED'),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-domain skills
        st.subheader("🔗 Cross-Domain Skill Transfer")
        skill_to_domains = {}
        for node, attrs in G.nodes(data=True):
            skill = attrs.get("robot_skill")
            domain = attrs.get("domain")
            if skill and domain:
                skill_to_domains.setdefault(skill, set()).add(domain)
        
        cross_skills = {s: list(d) for s, d in skill_to_domains.items() if len(d) > 1}
        
        if cross_skills:
            st.success(f"Found {len(cross_skills)} skills that transfer across multiple domains!")
            for skill, domains in cross_skills.items():
                st.markdown(f"✨ **{skill}** → appears in: `{', '.join(domains)}`")
        else:
            st.info("No cross-domain skills detected yet.")

# =============================================================================
# PAGE: INTERACTIVE GRAPH
# =============================================================================
elif page == "🌐 Interactive Graph":
    st.title("🌐 Interactive Skill Graph")
    st.markdown("*Live, draggable, hoverable visualization of the entire skill graph.*")
    
    html_path = OUTPUT_DIR / "visualizations" / "skill_graph_interactive.html"
    
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=850, scrolling=True)
        
        st.markdown("---")
        st.markdown("### 🎯 How to Interact")
        st.markdown("""
        - **Drag** any node to move it
        - **Hover** over any node to see all 3 intelligence layers
        - **Scroll** to zoom in/out
        - **Red edges** = cross-domain skill transfers
        - **Blue edges** = temporal sequence within a video
        """)
    else:
        st.warning("Interactive graph not found. Generate it from Notebook 09.")

# =============================================================================
# PAGE: SKILL QUERY ENGINE
# =============================================================================
elif page == "🔍 Skill Query Engine":
    st.title("🔍 Skill Query Engine")
    st.markdown("*Search the skill graph like a database. This is what robotics companies will use via our API.*")
    
    G, _ = load_master_graph()
    combined = load_all_enriched()
    
    if G is None or combined.empty:
        st.warning("No data available.")
    else:
        skills = sorted(combined["robot_skill"].unique())
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_skill = st.selectbox("🎯 Select Robot Skill to Query", skills)
        with col2:
            min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
        
        # Query
        results = []
        for node, attrs in G.nodes(data=True):
            if (attrs.get("robot_skill") == target_skill and 
                attrs.get("robot_skill_confidence", 0) >= min_conf):
                results.append({
                    "Domain": attrs.get("domain"),
                    "Video": attrs.get("video"),
                    "Frames": f"{attrs.get('start_frame')}-{attrs.get('end_frame')}",
                    "Intent": attrs.get("intent"),
                    "Kinematic": attrs.get("kinematic_grasp"),
                    "Confidence": attrs.get("robot_skill_confidence")
                })
        
        st.markdown(f"### 🎯 Found **{len(results)}** matching segments")
        
        if results:
            df_q = pd.DataFrame(results)
            st.dataframe(df_q, use_container_width=True)
            
            # Download as JSON (this is what API would return)
            json_output = df_q.to_json(orient='records', indent=2)
            st.download_button(
                "⬇️ Download as JSON (API-style response)",
                data=json_output,
                file_name=f"skill_query_{target_skill}.json",
                mime="application/json"
            )
            
            st.markdown("---")
            st.markdown("### 💡 What a Robotics Company Would Do With This")
            st.code(f"""
# Example API Usage (DRISHTI v1)

import requests

response = requests.get(
    "https://api.drishti.ai/v1/skills",
    params={{
        "robot_skill": "{target_skill}",
        "min_confidence": {min_conf}
    }},
    headers={{"Authorization": "Bearer YOUR_API_KEY"}}
)

skill_data = response.json()
# → Use skill_data to train robot policies
            """, language="python")
        else:
            st.info("No segments match your query.")

# =============================================================================
# PAGE: MANUAL → SKILL GRAPH
# =============================================================================
elif page == "📄 Manual → Skill Graph":
    st.title("📄 Manual → Skill Graph (Live Demo)")
    st.markdown("*Paste any procedural manual or SOP, and DRISHTI will instantly extract a robot skill graph.*")
    
    st.info("💡 This is the feature that proves DRISHTI works on text too — not just video. Construction SOPs, factory manuals, healthcare procedures — all become robot-ready skill graphs.")
    
    default_manual = """Step 1: Pick up the beading wire from the tray.
Step 2: Thread the first bead onto the wire carefully.
Step 3: Inspect the bead for any defects or cracks.
Step 4: Place the threaded bead at the end of the wire.
Step 5: Pick up the next bead and thread it onto the wire.
Step 6: Arrange the beads evenly along the wire.
Step 7: Fold the wire ends to secure the beads.
Step 8: Inspect the final necklace for quality.
Step 9: Place the finished necklace in the packaging box."""
    
    manual_text = st.text_area("📝 Paste your procedural manual here:", default_manual, height=300)
    
    domain_input = st.text_input("Domain name (e.g., construction, healthcare)", "manual_demo")
    
    if st.button("🚀 Extract Skill Graph from Manual"):
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from manual_to_skill_graph import build_manual_skill_graph
            
            with st.spinner("Parsing manual and extracting skills..."):
                manual_G, steps = build_manual_skill_graph(manual_text, domain=domain_input, title="User Manual")
            
            st.success(f"✅ Extracted {len(steps)} steps and built a skill graph!")
            
            st.markdown("### 📋 Extracted Skills")
            
            extracted = []
            for node, attrs in manual_G.nodes(data=True):
                if attrs.get("node_type") == "manual_step":
                    extracted.append({
                        "Step": attrs["step_number"],
                        "Action Verb": attrs["action_verb"],
                        "Robot Skill": attrs["robot_skill"],
                        "Complexity": attrs["skill_complexity"],
                        "DOF": attrs["dof_required"],
                        "Description": attrs["step_text"][:80] + "..."
                    })
            
            df_extracted = pd.DataFrame(extracted)
            st.dataframe(df_extracted, use_container_width=True)
            
            # Skill distribution
            st.markdown("### 📊 Skill Distribution in Manual")
            skill_counts = df_extracted["Robot Skill"].value_counts().reset_index()
            skill_counts.columns = ["skill", "count"]
            
            fig = px.pie(skill_counts, names="skill", values="count",
                        color_discrete_sequence=["#7B61FF", "#00D9FF", "#FFB74D", "#4CAF50"])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E8E9ED')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {e}")

# =============================================================================
# PAGE: BUSINESS VISION
# =============================================================================
elif page == "💼 Business Vision":
    st.title("💼 DRISHTI — From Research to Startup")
    
    st.markdown("## 🌐 Market Opportunity")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Humanoid Robotics Market by 2035", "$150B", "28% CAGR")
    with col2:
        st.metric("Indian Robotics Market by 2030", "$7.6B", "28% CAGR")
    with col3:
        st.metric("AI Training Data Market by 2028", "$5.3B", "23% CAGR")
    
    st.markdown("---")
    
    st.markdown("## 🎯 Our Positioning in the Robotics Stack")
    
    st.code("""
    ┌──────────────────────────────────────────┐
    │  LAYER 4: Robot Hardware                 │
    │  (Boston Dynamics, Figure, Agility)      │
    ├──────────────────────────────────────────┤
    │  LAYER 3: Control Models                 │
    │  (RT-2, Octo, OpenVLA)                   │
    ├──────────────────────────────────────────┤
    │  ⭐ LAYER 2: SKILL INTELLIGENCE          │
    │     ───── DRISHTI lives here ─────       │
    ├──────────────────────────────────────────┤
    │  LAYER 1: Raw Data                       │
    └──────────────────────────────────────────┘
    """, language="text")
    
    st.markdown("---")
    st.markdown("## 💰 Revenue Streams")
    
    revenue = pd.DataFrame({
        "Stream": ["Skill Graph API", "Custom Contracts", "Skill Diagnostics", "Marketplace"],
        "Target Customer": ["Robotics startups", "Enterprise clients", "Robotics consulting", "Developer ecosystem"],
        "Pricing": ["₹25K–₹15L/year", "₹5L–₹25L/project", "₹3L–₹10L/engagement", "Phase 2 revenue share"],
        "Year of Launch": ["Year 1", "Year 1", "Year 2", "Year 3+"]
    })
    st.dataframe(revenue, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## 🚀 Go-to-Market Strategy")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card">
        <h3>Year 1: Research</h3>
        <p>• IITs, IIITs, ARTPARK</p>
        <p>• DRDO labs</p>
        <p>• Build academic credibility</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
        <h3>Year 2: Indian Startups</h3>
        <p>• Addverb, Miko</p>
        <p>• TCS Robotics, GreyOrange</p>
        <p>• Enterprise case studies</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-card">
        <h3>Year 3+: Global</h3>
        <p>• Figure AI, Agility</p>
        <p>• Boston Dynamics, Toyota</p>
        <p>• Become global standard</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE: FUTURE PLANS
# =============================================================================
elif page == "🔮 Future Plans":
    st.title("🔮 Future Plans — What's Next for DRISHTI")
    st.markdown("*These are our 4 core research directions to scale DRISHTI into a production-ready robotics platform.*")
    
    st.markdown("---")
    
    # Plan 1
    st.markdown("""
    <div class="info-card">
    <h3>🤖 1. Closed-Loop Robot Execution with Adaptive Demonstration Refinement</h3>
    <p>We will integrate DRISHTI with robotic platforms like SO-101 and LeRobot to enable real robotic execution of skills extracted from human demonstrations. When the robot fails at a specific sub-skill, DRISHTI will identify the exact failing skill node and request a targeted human demonstration for only that skill — closing the loop between human teaching, structured understanding, and continuous robotic improvement.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Plan 2
    st.markdown("""
    <div class="info-card">
    <h3>🧠 2. Perception-Grounded Skill-to-Action Engine</h3>
    <p>DRISHTI will convert high-level skills into adaptive robot actions by combining its skill graph with real-time computer vision. Instead of fixed motion values, the robot will perceive each object's shape and dimensions on the fly — meaning the same <code>precision_pinch_grasp</code> can be reused for an earring, a coin, or a key, with parameters dynamically adjusted to the object in front of it.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Plan 3
    st.markdown("""
    <div class="info-card">
    <h3>📚 3. Universal Robot Skill Library — Cross-Domain Training at Scale</h3>
    <p>We will scale DRISHTI into a centralized, growing repository of reusable robot skills extracted from thousands of demonstrations across domains. Robotics companies will be able to query this library for specific skill types, complexity levels, and DOF requirements — turning DRISHTI into a foundational data infrastructure for cross-domain robot learning, much like ImageNet was for computer vision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Plan 4
    st.markdown("""
    <div class="info-card">
    <h3>🏭 4. SOP-to-Robot Deployment Pipeline for Industry</h3>
    <p>Once video-based learning is mature, DRISHTI will extend its existing manual-to-skill-graph capability into a complete industrial pipeline — directly converting written Standard Operating Procedures from factories and warehouses into robot-executable skill policies. This will eventually allow companies to deploy robots directly from existing documentation, without requiring fresh video data collection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 🎯 Why These 4 Plans Matter")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="layer-card">
        <h3>🔬 Research Impact</h3>
        <p style="color: #D4D7E0;">• Closes the sim-to-real gap through human-in-the-loop refinement</p>
        <p style="color: #D4D7E0;">• Creates reusable, cross-domain robot skill primitives</p>
        <p style="color: #D4D7E0;">• Bridges structured NLP and embodied AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="layer-card">
        <h3>🏢 Industry Impact</h3>
        <p style="color: #D4D7E0;">• Reduces robot training time from months to days</p>
        <p style="color: #D4D7E0;">• Enables SMEs to deploy robots without ML expertise</p>
        <p style="color: #D4D7E0;">• Unlocks automation for unstructured tasks (jewelry, healthcare, assembly)</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE: TEAM & ABOUT
# =============================================================================
elif page == "👥 Team & About":
    st.title("👥 Meet Team Focus")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="team-card">
        <h3>Suryansh Singh</h3>
        <p style="color: #B8BFCC; margin: 10px 0;">Roll No: 1024230047</p>
        <p style="color: #D4D7E0; margin: 15px 0;">Pipeline Architecture • Intelligence Core • Research</p>
        <a href="https://www.linkedin.com/in/suryansh-singh-here/" target="_blank" class="linkedin-btn">
        🔗 LinkedIn
        </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="team-card">
        <h3>Shraddha Sahni</h3>
        <p style="color: #B8BFCC; margin: 10px 0;">Roll No: 1024230044</p>
        <p style="color: #D4D7E0; margin: 15px 0;">Computer Vision • Data Analysis • Research</p>
        <a href="https://www.linkedin.com/in/shraddha-sahni-b56bbb370/" target="_blank" class="linkedin-btn">
        🔗 LinkedIn
        </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🛠️ Tech Stack Used")
    
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("""
        <div class="info-card">
        <h3>Computer Vision</h3>
        <p><strong>MediaPipe Hands</strong> — 21 keypoints per hand</p>
        <p><strong>YOLOv8 (Ultralytics)</strong> — Object detection</p>
        <p><strong>OpenCV</strong> — Frame processing</p>
        </div>
        """, unsafe_allow_html=True)
    with tech_col2:
        st.markdown("""
        <div class="info-card">
        <h3>Data & Graph</h3>
        <p><strong>NetworkX</strong> — Skill graph construction</p>
        <p><strong>JSON-LD</strong> — Robot-framework compatible export</p>
        <p><strong>Pandas</strong> — Batch data processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📂 Project Architecture")
    st.code("""
DRISHTI/
├── data/                      # Egocentric videos (3 domains)
├── src/                       # Reusable Python modules
│   ├── video_loader.py
│   ├── preprocessing.py
│   ├── hand_detector.py
│   ├── object_detector.py
│   ├── contact_estimator.py
│   ├── action_segmenter.py
│   ├── kinematic_analyzer.py
│   ├── intent_inference.py
│   ├── robot_skill_mapper.py
│   ├── skill_graph_builder.py
│   ├── manual_to_skill_graph.py
│   └── full_pipeline.py
├── notebooks/                 # Step-by-step Jupyter notebooks
│   ├── 01_setup_and_data_loading.ipynb
│   ├── ... (10 notebooks)
│   └── DEMO_DAY.ipynb
├── outputs/                   # All generated outputs
└── dashboard/                 # This Streamlit app
    """, language="text")
    
    st.markdown("---")
    st.markdown("## 📞 Contact")
    st.markdown("Built with ❤️ in India for the future of robotics.")
