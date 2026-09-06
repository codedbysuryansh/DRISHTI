# =============================================================================
# DRISHTI: Text Manual → Skill Graph Extension
# Reads a procedural text (manual/SOP) and converts it to skill graph nodes
# This extends DRISHTI beyond video — to text-based skill extraction
# =============================================================================

import re
import json
import networkx as nx

# Simple skill keyword mapper for manual parsing
ACTION_VERB_TO_SKILL = {
    # Fine manipulation
    "pick": "dexterous_fine_manipulation",
    "grasp": "dexterous_fine_manipulation",
    "pinch": "dexterous_fine_manipulation",
    "hold": "dexterous_fine_manipulation",
    "thread": "dexterous_fine_manipulation",
    "insert": "precision_placement",
    "place": "precision_placement",
    "position": "precision_placement",
    "align": "precision_placement",
    "arrange": "object_placement_precision",
    "sort": "precision_sorting",
    "separate": "precision_sorting",
    "inspect": "visual_inspection",
    "check": "visual_inspection",
    "examine": "visual_inspection",
    # Tool manipulation
    "sweep": "tool_manipulation_sweeping",
    "wipe": "tool_manipulation_wiping",
    "mop": "tool_manipulation_sweeping",
    "clean": "tool_manipulation_wiping",
    "dust": "tool_manipulation_wiping",
    "scrub": "tool_manipulation_wiping",
    # Power grip
    "grip": "power_grip_manipulation",
    "carry": "power_grip_manipulation",
    "lift": "power_grip_manipulation",
    "move": "power_grip_manipulation",
    "push": "power_grip_manipulation",
    "pull": "power_grip_manipulation",
    # Bimanual
    "fold": "bimanual_coordination",
    "wrap": "bimanual_coordination",
    "tie": "bimanual_coordination",
    "assemble": "bimanual_coordination",
    # Handover
    "give": "handover_interaction",
    "hand": "handover_interaction",
    "pass": "handover_interaction",
    "transfer": "handover_interaction"
}

SKILL_TO_COMPLEXITY = {
    "dexterous_fine_manipulation": "high",
    "precision_placement": "medium",
    "precision_sorting": "medium",
    "visual_inspection": "medium",
    "tool_manipulation_sweeping": "medium",
    "tool_manipulation_wiping": "low",
    "power_grip_manipulation": "low",
    "bimanual_coordination": "high",
    "handover_interaction": "medium",
    "object_placement_precision": "medium",
    "general_manipulation": "low"
}

SKILL_TO_DOF = {
    "dexterous_fine_manipulation": 6,
    "precision_placement": 5,
    "precision_sorting": 5,
    "visual_inspection": 4,
    "tool_manipulation_sweeping": 4,
    "tool_manipulation_wiping": 3,
    "power_grip_manipulation": 3,
    "bimanual_coordination": 12,
    "handover_interaction": 5,
    "object_placement_precision": 5,
    "general_manipulation": 4
}

def extract_steps_from_manual(text):
    """
    Extracts steps from a procedural manual/SOP text.
    Handles numbered lists, bullet points, and plain sentences.
    """
    steps = []

    # Try numbered list: "1. Do this" or "Step 1: Do this"
    numbered = re.findall(
        r'(?:step\s*\d+[:\.]?\s*|^\d+[:\.)]\s*)(.+)',
        text, re.IGNORECASE | re.MULTILINE
    )
    if numbered:
        steps = [s.strip() for s in numbered]
    else:
        # Try bullet points
        bulleted = re.findall(r'(?:^[-•*]\s*)(.+)', text, re.MULTILINE)
        if bulleted:
            steps = [s.strip() for s in bulleted]
        else:
            # Fall back to splitting by sentences
            steps = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 5]

    return steps

def classify_step(step_text):
    """
    Classifies a manual step into a robot skill.
    Returns: (action_verb, skill_name, intent_description)
    """
    words = step_text.lower().split()

    matched_skill = "general_manipulation"
    matched_verb = "perform"

    for word in words:
        word_clean = re.sub(r'[^a-z]', '', word)
        if word_clean in ACTION_VERB_TO_SKILL:
            matched_skill = ACTION_VERB_TO_SKILL[word_clean]
            matched_verb = word_clean
            break

    return matched_verb, matched_skill, step_text

def build_manual_skill_graph(manual_text, domain="manual", title="Untitled Manual"):
    """
    Converts a procedural text manual into a DRISHTI-compatible skill graph.

    Returns: NetworkX DiGraph with same structure as video-based graph
    """
    G = nx.DiGraph()

    root_id = f"{domain}::manual::root"
    G.add_node(root_id,
               node_type="manual_root",
               domain=domain,
               title=title,
               source="text_manual")

    steps = extract_steps_from_manual(manual_text)
    print(f"📄 Extracted {len(steps)} steps from manual")

    prev_node = root_id

    for idx, step in enumerate(steps):
        verb, skill, description = classify_step(step)
        node_id = f"{domain}::manual::step_{idx+1}"

        G.add_node(node_id,
                   node_type="manual_step",
                   domain=domain,
                   step_number=idx + 1,
                   step_text=step,

                   # Layer 1: Kinematic (inferred from verb)
                   kinematic_grasp="inferred_from_text",
                   action_verb=verb,

                   # Layer 2: Intent (the step itself)
                   intent=f"manual_{verb}_task",
                   intent_description=description,

                   # Layer 3: Robot Skill
                   robot_skill=skill,
                   skill_complexity=SKILL_TO_COMPLEXITY.get(skill, "medium"),
                   dof_required=SKILL_TO_DOF.get(skill, 4),
                   source="text_manual"
        )

        G.add_edge(prev_node, node_id, edge_type="manual_step_sequence")
        prev_node = node_id

    print(f"✅ Manual skill graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, steps

def merge_manual_graph_into_master(master_graph, manual_graph):
    """
    Merges a manual-derived graph into the master video-based graph.
    Also adds cross-domain links where skills match.
    """
    merged = nx.compose(master_graph, manual_graph)

    # Find cross-domain skill links
    manual_nodes = {n: d for n, d in manual_graph.nodes(data=True)
                   if d.get("node_type") == "manual_step"}
    video_nodes = {n: d for n, d in master_graph.nodes(data=True)
                  if d.get("node_type") == "segment"}

    cross_edges = 0
    for m_node, m_attrs in manual_nodes.items():
        m_skill = m_attrs.get("robot_skill")
        for v_node, v_attrs in video_nodes.items():
            if v_attrs.get("robot_skill") == m_skill:
                merged.add_edge(m_node, v_node,
                               edge_type="manual_to_video_skill_match",
                               shared_skill=m_skill)
                cross_edges += 1
                break  # one match per manual node is enough

    print(f"✅ Merged graphs: {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
    print(f"🔗 Manual-to-video skill links: {cross_edges}")
    return merged