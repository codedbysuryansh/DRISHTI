import json
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from pathlib import Path

# Color palette per domain
DOMAIN_COLORS = {
    "artificial_jewellery": "#FFD700",  # gold
    "home_cleaning": "#4CAF50",          # green
    "shop": "#2196F3",                   # blue
    "video_root": "#9C27B0",             # purple
    "default": "#808080"
}

# Color palette per robot skill
SKILL_COLORS = {
    "dexterous_fine_manipulation": "#E91E63",
    "bimanual_coordination": "#FF5722",
    "tool_manipulation_sweeping": "#4CAF50",
    "tool_manipulation_wiping": "#8BC34A",
    "object_placement_precision": "#2196F3",
    "precision_placement": "#03A9F4",
    "power_grip_manipulation": "#795548",
    "precision_sorting": "#FFC107",
    "handover_interaction": "#9C27B0",
    "visual_inspection": "#00BCD4",
    "general_manipulation": "#9E9E9E"
}

def get_node_color(attrs, color_by="domain"):
    if color_by == "domain":
        if attrs.get("node_type") == "video_root":
            return DOMAIN_COLORS["video_root"]
        return DOMAIN_COLORS.get(attrs.get("domain"), DOMAIN_COLORS["default"])
    elif color_by == "skill":
        return SKILL_COLORS.get(attrs.get("robot_skill"), SKILL_COLORS["general_manipulation"])
    return "#808080"

def visualize_static(G, output_path, color_by="domain", title="DRISHTI Skill Graph"):
    """Static matplotlib visualization (good for PPT)."""
    plt.figure(figsize=(20, 14))
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except:
        pos = nx.kamada_kawai_layout(G)
    
    # Node colors
    node_colors = [get_node_color(G.nodes[n], color_by) for n in G.nodes()]
    
    # Node sizes (root bigger)
    node_sizes = []
    for n in G.nodes():
        if G.nodes[n].get("node_type") == "video_root":
            node_sizes.append(800)
        else:
            node_sizes.append(300)
    
    # Edge styles
    edge_colors = []
    edge_widths = []
    for u, v, attrs in G.edges(data=True):
        if attrs.get("edge_type") == "cross_domain_skill_transfer":
            edge_colors.append("#FF1744")
            edge_widths.append(1.5)
        else:
            edge_colors.append("#999999")
            edge_widths.append(0.7)
    
    nx.draw(G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
            width=edge_widths,
            with_labels=False,
            arrows=True,
            arrowsize=8,
            alpha=0.85)
    
    plt.title(title, fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved static graph: {output_path}")

def visualize_interactive(G, output_path, color_by="skill"):
    """Interactive HTML graph using pyvis."""
    net = Network(
        height="850px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True
    )
    
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)
    
    # Add nodes
    for node, attrs in G.nodes(data=True):
        color = get_node_color(attrs, color_by)
        
        if attrs.get("node_type") == "video_root":
            label = f"📹 {attrs.get('video', node)}"
            size = 35
            title_html = f"""
            <b>VIDEO ROOT</b><br>
            Domain: {attrs.get('domain')}<br>
            Total segments: {attrs.get('total_segments')}
            """
        else:
            skill = attrs.get('robot_skill', 'unknown')
            intent = attrs.get('intent', 'unknown')
            label = f"{skill[:20]}"
            size = 18
            title_html = f"""
            <b>SEGMENT</b><br>
            <b>Frames:</b> {attrs.get('start_frame')}-{attrs.get('end_frame')}<br>
            <b>🤲 Kinematic:</b> {attrs.get('kinematic_grasp')}<br>
            <b>🎯 Intent:</b> {intent}<br>
            <b>🤖 Robot Skill:</b> {skill}<br>
            <b>Complexity:</b> {attrs.get('skill_complexity')}<br>
            <b>DOF:</b> {attrs.get('dof_required')}<br>
            <b>Bimanual:</b> {attrs.get('bimanual')}<br>
            <b>Confidence:</b> {attrs.get('robot_skill_confidence')}
            """
        
        net.add_node(node, label=label, title=title_html, color=color, size=size)
    
    # Add edges
    for u, v, attrs in G.edges(data=True):
        edge_type = attrs.get("edge_type", "temporal_next")
        if edge_type == "cross_domain_skill_transfer":
            net.add_edge(u, v, color="#FF1744", title=f"🔗 Shared Skill: {attrs.get('shared_skill')}", width=2)
        else:
            net.add_edge(u, v, color="#88aaff", title="temporal next", width=1)
    
    net.set_options('''
    var options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {"size": 11, "face": "arial"}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "stabilization": {"iterations": 150}
      }
    }
    ''')
    
    net.save_graph(str(output_path))
    print(f"✅ Saved interactive graph: {output_path}")

def query_graph_by_skill(G, target_skill):
    """Find all segments with a given robot skill."""
    matches = []
    for node, attrs in G.nodes(data=True):
        if attrs.get("robot_skill") == target_skill:
            matches.append({
                "node_id": node,
                "domain": attrs.get("domain"),
                "video": attrs.get("video"),
                "frames": f"{attrs.get('start_frame')}-{attrs.get('end_frame')}",
                "intent": attrs.get("intent"),
                "kinematic": attrs.get("kinematic_grasp"),
                "confidence": attrs.get("robot_skill_confidence")
            })
    return matches

def get_cross_domain_skills(G):
    """Find skills that appear across multiple domains."""
    skill_to_domains = {}
    for node, attrs in G.nodes(data=True):
        skill = attrs.get("robot_skill")
        domain = attrs.get("domain")
        if skill and domain:
            skill_to_domains.setdefault(skill, set()).add(domain)
    
    cross_domain = {s: list(d) for s, d in skill_to_domains.items() if len(d) > 1}
    return cross_domain