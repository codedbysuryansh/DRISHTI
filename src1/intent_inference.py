import json

# =============================================================================
# INTENT RULES ENGINE
# Combines domain + objects + kinematics → intent label + confidence
# =============================================================================

INTENT_LIBRARY = {
    # --- Artificial Jewellery Domain ---
    "jewellery_assembling": {
        "keywords": ["bead", "chain", "clasp", "wire", "thread"],
        "required_grasp": ["pinch", "partial_grip"],
        "required_motion": ["still", "slow"],
        "description": "Assembling jewellery components (beads, chains, clasps)"
    },
    "jewellery_sorting": {
        "keywords": ["bead", "stone", "piece", "tray", "box"],
        "required_grasp": ["pinch", "open_hand"],
        "required_motion": ["slow", "moderate"],
        "description": "Sorting and organizing jewellery pieces"
    },
    "jewellery_packaging": {
        "keywords": ["box", "packet", "cover", "wrap"],
        "required_grasp": ["closed_power", "partial_grip"],
        "required_motion": ["slow", "moderate"],
        "description": "Packaging finished jewellery items"
    },
    "jewellery_quality_check": {
        "keywords": ["magnifier", "light", "inspect"],
        "required_grasp": ["pinch"],
        "required_motion": ["still"],
        "description": "Quality inspection of jewellery pieces"
    },

    # --- Home Cleaning Domain ---
    "cleaning_sweeping": {
        "keywords": ["broom", "jhaadu", "floor", "dust"],
        "required_grasp": ["closed_power", "partial_grip"],
        "required_motion": ["moderate", "fast"],
        "description": "Sweeping floor with broom (jhaadu)"
    },
    "cleaning_wiping": {
        "keywords": ["cloth", "rag", "towel", "surface", "table"],
        "required_grasp": ["open_hand", "partial_grip"],
        "required_motion": ["moderate"],
        "description": "Wiping surfaces with cloth"
    },
    "cleaning_mopping": {
        "keywords": ["mop", "floor", "water", "bucket"],
        "required_grasp": ["closed_power"],
        "required_motion": ["moderate", "fast"],
        "description": "Mopping floor with wet mop"
    },
    "cleaning_arranging": {
        "keywords": ["item", "object", "shelf", "organize"],
        "required_grasp": ["pinch", "partial_grip"],
        "required_motion": ["slow"],
        "description": "Arranging and organizing household items"
    },
    "cleaning_dusting": {
        "keywords": ["duster", "cloth", "surface", "furniture"],
        "required_grasp": ["open_hand", "partial_grip"],
        "required_motion": ["slow", "moderate"],
        "description": "Dusting furniture and surfaces"
    },

    # --- Shop Operations Domain ---
    "shop_display_arrange": {
        "keywords": ["cloth", "shirt", "dress", "rack", "shelf", "display"],
        "required_grasp": ["open_hand", "partial_grip"],
        "required_motion": ["slow", "moderate"],
        "description": "Arranging products on display (clothes, items)"
    },
    "shop_billing": {
        "keywords": ["bill", "cash", "card", "machine", "counter"],
        "required_grasp": ["pinch", "partial_grip"],
        "required_motion": ["still", "slow"],
        "description": "Billing and payment handling"
    },
    "shop_inventory": {
        "keywords": ["box", "stock", "inventory", "shelf"],
        "required_grasp": ["closed_power", "partial_grip"],
        "required_motion": ["moderate"],
        "description": "Managing inventory and stock"
    },
    "shop_customer_service": {
        "keywords": ["person", "customer", "handover", "give"],
        "required_grasp": ["open_hand", "partial_grip"],
        "required_motion": ["slow", "moderate"],
        "description": "Interacting with customers, handing over items"
    },
    "shop_folding": {
        "keywords": ["cloth", "shirt", "fold", "stack"],
        "required_grasp": ["open_hand", "partial_grip"],
        "required_motion": ["slow"],
        "description": "Folding clothes for display or customer"
    },
}

# Domain to intent mapping (fallback if no object match)
DOMAIN_DEFAULT_INTENTS = {
    "artificial_jewellery": "jewellery_assembling",
    "home_cleaning": "cleaning_arranging",
    "shop": "shop_display_arrange"
}

def match_intent(domain, objects_list, grasp, motion, bimanual):
    """
    objects_list: list of object labels detected in this segment
    grasp: dominant_grasp from kinematics
    motion: motion_label from kinematics
    bimanual: bool

    Returns: (best_intent, confidence, description)
    """
    scores = []

    for intent_name, rules in INTENT_LIBRARY.items():
        # Check if intent belongs to this domain (by prefix)
        intent_domain = intent_name.split("_")[0]
        domain_match = (
            (domain == "artificial_jewellery" and intent_domain == "jewellery") or
            (domain == "home_cleaning" and intent_domain == "cleaning") or
            (domain == "shop" and intent_domain == "shop")
        )
        if not domain_match:
            continue

        score = 0

        # Object keyword match (most important)
        objects_lower = [str(o).lower() for o in objects_list]
        for kw in rules["keywords"]:
            for obj in objects_lower:
                if kw in obj:
                    score += 3
                    break

        # Grasp match
        if grasp in rules["required_grasp"]:
            score += 2

        # Motion match
        if motion in rules["required_motion"]:
            score += 2

        # Bimanual bonus for certain tasks
        if bimanual and intent_name in ["jewellery_assembling", "shop_folding", "cleaning_arranging"]:
            score += 1

        if score > 0:
            scores.append((intent_name, score, rules["description"]))

    if not scores:
        # Fallback to domain default
        default = DOMAIN_DEFAULT_INTENTS.get(domain, "general_task")
        return default, 0.3, "General task (no specific intent matched)"

    # Pick best match
    scores.sort(key=lambda x: x[1], reverse=True)
    best = scores[0]

    # Normalize confidence (0-1)
    confidence = min(1.0, best[1] / 10.0)

    return best[0], round(confidence, 2), best[2]

def infer_intent_for_segment(domain, objects_df_segment, kinematic_row):
    """
    objects_df_segment: object detections for this segment (can be empty)
    kinematic_row: dict with grasp, motion_label, bimanual

    Returns: dict with intent info
    """
    # Extract objects
    objects_list = []
    if objects_df_segment is not None and not objects_df_segment.empty:
        objects_list = objects_df_segment["label"].dropna().unique().tolist()

    grasp = kinematic_row.get("dominant_grasp", "unknown")
    motion = kinematic_row.get("motion_label", "unknown")
    bimanual = kinematic_row.get("bimanual", False)

    intent, conf, desc = match_intent(domain, objects_list, grasp, motion, bimanual)

    return {
        "intent": intent,
        "intent_confidence": conf,
        "intent_description": desc,
        "objects_used": objects_list[:5],  # top 5 objects
        "grasp_used": grasp,
        "motion_used": motion,
        "bimanual": bimanual
    }