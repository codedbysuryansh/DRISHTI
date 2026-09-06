# =============================================================================
# ROBOT SKILL MAPPER
# Maps (kinematic + intent) → Robot Skill Primitive
# This is Layer 3 of DRISHTI Intelligence Core
# =============================================================================

ROBOT_SKILL_LIBRARY = {
    # Fine manipulation skills
    "dexterous_fine_manipulation": {
        "description": "Precise control of small objects with fingertips",
        "required_grasp": ["pinch"],
        "typical_intents": ["jewellery_assembling", "jewellery_quality_check"],
        "complexity": "high",
        "dof_required": 6,  # degrees of freedom
        "force_control": "high_precision"
    },
    "precision_placement": {
        "description": "Accurate placement of objects in specific locations",
        "required_grasp": ["pinch", "partial_grip"],
        "typical_intents": ["jewellery_assembling", "shop_display_arrange"],
        "complexity": "medium",
        "dof_required": 5
    },
    
    # Bimanual coordination skills
    "bimanual_coordination": {
        "description": "Two-handed coordinated manipulation",
        "required_grasp": ["partial_grip", "open_hand"],
        "typical_intents": ["shop_folding", "cleaning_arranging", "jewellery_packaging"],
        "complexity": "high",
        "dof_required": 12,  # 6 per hand
        "requires_dual_arm": True
    },
    
    # Tool manipulation skills
    "tool_manipulation_sweeping": {
        "description": "Using elongated tools with sweeping motion patterns",
        "required_grasp": ["closed_power", "partial_grip"],
        "typical_intents": ["cleaning_sweeping"],
        "complexity": "medium",
        "dof_required": 4,
        "motion_pattern": "periodic_sweep"
    },
    "tool_manipulation_wiping": {
        "description": "Surface contact manipulation with cloth/wiper",
        "required_grasp": ["open_hand", "partial_grip"],
        "typical_intents": ["cleaning_wiping", "cleaning_dusting"],
        "complexity": "low",
        "dof_required": 3,
        "motion_pattern": "back_and_forth"
    },
    
    # Object handling skills
    "object_placement_precision": {
        "description": "Controlled placement of medium-sized objects",
        "required_grasp": ["open_hand", "partial_grip"],
        "typical_intents": ["shop_display_arrange", "cleaning_arranging"],
        "complexity": "medium",
        "dof_required": 5
    },
    "power_grip_manipulation": {
        "description": "Firm grasping and manipulation of rigid objects",
        "required_grasp": ["closed_power"],
        "typical_intents": ["shop_inventory", "cleaning_mopping"],
        "complexity": "low",
        "dof_required": 3
    },
    
    # Sorting and organization
    "precision_sorting": {
        "description": "Picking and categorizing small items",
        "required_grasp": ["pinch", "open_hand"],
        "typical_intents": ["jewellery_sorting"],
        "complexity": "medium",
        "dof_required": 5
    },
    
    # Customer interaction skills
    "handover_interaction": {
        "description": "Safe human-robot handover of objects",
        "required_grasp": ["open_hand", "partial_grip"],
        "typical_intents": ["shop_customer_service"],
        "complexity": "medium",
        "dof_required": 5,
        "safety_critical": True
    },
    
    # Inspection skills
    "visual_inspection": {
        "description": "Hold object for detailed visual examination",
        "required_grasp": ["pinch"],
        "typical_intents": ["jewellery_quality_check"],
        "complexity": "medium",
        "dof_required": 4,
        "requires_vision": True
    },
    
    # General fallback
    "general_manipulation": {
        "description": "General-purpose object manipulation",
        "required_grasp": ["partial_grip", "open_hand"],
        "typical_intents": [],
        "complexity": "low",
        "dof_required": 4
    }
}

def map_to_robot_skill(intent, kinematic_grasp, motion_label, bimanual):
    """
    Maps (intent + kinematics) → Robot Skill Primitive
    
    Returns: (skill_name, confidence, metadata)
    """
    scores = []
    
    for skill_name, skill_meta in ROBOT_SKILL_LIBRARY.items():
        score = 0
        
        # Intent match (strongest signal)
        if intent in skill_meta["typical_intents"]:
            score += 5
        
        # Grasp compatibility
        if kinematic_grasp in skill_meta["required_grasp"]:
            score += 3
        
        # Bimanual requirement
        if bimanual and skill_meta.get("requires_dual_arm", False):
            score += 2
        elif not bimanual and not skill_meta.get("requires_dual_arm", False):
            score += 1
        
        # Motion pattern alignment
        if motion_label == "fast" and skill_meta.get("motion_pattern") == "periodic_sweep":
            score += 2
        elif motion_label == "slow" and "precision" in skill_name:
            score += 2
        
        if score > 0:
            scores.append((skill_name, score, skill_meta))
    
    if not scores:
        # Fallback
        return "general_manipulation", 0.3, ROBOT_SKILL_LIBRARY["general_manipulation"]
    
    # Best match
    scores.sort(key=lambda x: x[1], reverse=True)
    best = scores[0]
    
    # Normalize confidence
    confidence = min(1.0, best[1] / 10.0)
    
    return best[0], round(confidence, 2), best[2]

def enrich_with_robot_skill(intent_row):
    """
    Takes one row from intent CSV and adds robot skill info.
    """
    intent = intent_row.get("intent", "unknown")
    grasp = intent_row.get("grasp_used", "unknown")
    motion = intent_row.get("motion_used", "unknown")
    bimanual = intent_row.get("bimanual", False)
    
    skill_name, skill_conf, skill_meta = map_to_robot_skill(intent, grasp, motion, bimanual)
    
    return {
        "robot_skill": skill_name,
        "robot_skill_confidence": skill_conf,
        "skill_complexity": skill_meta.get("complexity", "unknown"),
        "dof_required": skill_meta.get("dof_required", None),
        "safety_critical": skill_meta.get("safety_critical", False),
        "requires_dual_arm": skill_meta.get("requires_dual_arm", False)
    }