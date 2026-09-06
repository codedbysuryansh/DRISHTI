import networkx as nx
import json

def build_skill_graph_for_video(enriched_segments_df, video_stem, domain):
    """
    enriched_segments_df: DataFrame with all 3 layers (kinematic, intent, robot_skill)
    
    Returns: NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    # Add video-level root node
    root_id = f"{domain}::{video_stem}::root"
    G.add_node(root_id, 
               node_type="video_root",
               domain=domain,
               video=video_stem,
               total_segments=len(enriched_segments_df))
    
    prev_node = root_id
    
    for idx, row in enriched_segments_df.iterrows():
        node_id = f"{domain}::{video_stem}::seg_{row['start_frame']}_{row['end_frame']}"
        
        # Node attributes (all 3 layers)
        G.add_node(node_id,
                   node_type="segment",
                   domain=domain,
                   video=video_stem,
                   start_frame=int(row['start_frame']),
                   end_frame=int(row['end_frame']),
                   duration_frames=int(row['duration_frames']),
                   
                   # Layer 1: Kinematic
                   kinematic_grasp=row.get('grasp_used', 'unknown'),
                   kinematic_motion=row.get('motion_used', 'unknown'),
                   bimanual=bool(row.get('bimanual', False)),
                   
                   # Layer 2: Intent
                   intent=row.get('intent', 'unknown'),
                   intent_confidence=float(row.get('intent_confidence', 0)),
                   intent_description=row.get('intent_description', ''),
                   
                   # Layer 3: Robot Skill
                   robot_skill=row.get('robot_skill', 'unknown'),
                   robot_skill_confidence=float(row.get('robot_skill_confidence', 0)),
                   skill_complexity=row.get('skill_complexity', 'unknown'),
                   dof_required=row.get('dof_required', None),
                   safety_critical=bool(row.get('safety_critical', False))
        )
        
        # Temporal edge
        G.add_edge(prev_node, node_id, edge_type="temporal_next")
        prev_node = node_id
    
    return G

def add_cross_domain_skill_edges(graph_collection):
    """
    graph_collection: dict of {video_id: nx.DiGraph}
    
    Adds cross-domain edges linking nodes with the same robot_skill.
    Returns: merged graph
    """
    # Merge all graphs
    merged = nx.DiGraph()
    for g in graph_collection.values():
        merged = nx.compose(merged, g)
    
    # Build skill index
    skill_index = {}
    for node, attrs in merged.nodes(data=True):
        if attrs.get("node_type") == "segment":
            skill = attrs.get("robot_skill")
            if skill:
                if skill not in skill_index:
                    skill_index[skill] = []
                skill_index[skill].append(node)
    
    # Add cross-domain edges
    cross_edges = 0
    for skill, nodes in skill_index.items():
        if len(nodes) > 1:
            # Connect all nodes with same skill (creates a skill cluster)
            for i in range(len(nodes)):
                for j in range(i+1, min(i+6, len(nodes))):  # limit to avoid explosion
                    if nodes[i].split("::")[0] != nodes[j].split("::")[0]:  # different domains
                        merged.add_edge(nodes[i], nodes[j], 
                                      edge_type="cross_domain_skill_transfer",
                                      shared_skill=skill)
                        cross_edges += 1
    
    print(f"✅ Added {cross_edges} cross-domain skill transfer edges")
    return merged

def export_graph_to_json_ld(G, output_path):
    """
    Export NetworkX graph to JSON-LD format (robot-framework compatible).
    """
    data = {
        "@context": "https://drishti.robotics/skill-graph/v1",
        "@type": "RobotSkillGraph",
        "nodes": [],
        "edges": []
    }
    
    for node, attrs in G.nodes(data=True):
        data["nodes"].append({
            "id": node,
            **{k: (v if not isinstance(v, (bool, int, float)) else v) 
               for k, v in attrs.items()}
        })
    
    for u, v, attrs in G.edges(data=True):
        data["edges"].append({
            "source": u,
            "target": v,
            **attrs
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Exported skill graph: {output_path}")