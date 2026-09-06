import pandas as pd
from pathlib import Path
from tqdm import tqdm

from robot_skill_mapper import enrich_with_robot_skill
from skill_graph_builder import (
    build_skill_graph_for_video,
    add_cross_domain_skill_edges,
    export_graph_to_json_ld
)

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

def run_batch_robot_skill_and_graph(
    intent_root_dir,
    out_root_dir
):
    """
    Reads:
      - outputs/intent/<domain>/<video>_intent.csv
    
    Writes:
      - outputs/robot_skills/<domain>/<video>_enriched.csv (all 3 layers)
      - outputs/skill_graphs/<domain>/<video>_graph.json (per-video graph)
      - outputs/skill_graphs/DRISHTI_master_graph.json (merged + cross-domain)
    """
    intent_root_dir = Path(intent_root_dir)
    out_root_dir = Path(out_root_dir)
    
    skill_csv_root = out_root_dir / "robot_skills"
    graph_root = out_root_dir / "skill_graphs"
    skill_csv_root.mkdir(parents=True, exist_ok=True)
    graph_root.mkdir(parents=True, exist_ok=True)
    
    intent_csvs = list(intent_root_dir.rglob("*_intent.csv"))
    summary = []
    all_graphs = {}
    
    for intent_csv in tqdm(sorted(intent_csvs), desc="Robot skill mapping + graph building"):
        domain = intent_csv.parent.name
        video_stem = intent_csv.name.replace("_intent.csv", "")
        
        df = safe_read_csv(intent_csv)
        if df.empty:
            summary.append({"domain": domain, "video_stem": video_stem, "status": "skipped_empty"})
            continue
        
        # Enrich with Layer 3 (robot skill)
        enriched_rows = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            skill_info = enrich_with_robot_skill(row_dict)
            row_dict.update(skill_info)
            enriched_rows.append(row_dict)
        
        enriched_df = pd.DataFrame(enriched_rows)
        
        # Save enriched CSV
        (skill_csv_root / domain).mkdir(parents=True, exist_ok=True)
        csv_out = skill_csv_root / domain / f"{video_stem}_enriched.csv"
        enriched_df.to_csv(csv_out, index=False)
        
        # Build skill graph
        G = build_skill_graph_for_video(enriched_df, video_stem, domain)
        
        # Save per-video graph
        (graph_root / domain).mkdir(parents=True, exist_ok=True)
        graph_out = graph_root / domain / f"{video_stem}_graph.json"
        export_graph_to_json_ld(G, graph_out)
        
        all_graphs[f"{domain}::{video_stem}"] = G
        
        summary.append({
            "domain": domain,
            "video_stem": video_stem,
            "segments": len(enriched_df),
            "enriched_csv": str(csv_out),
            "graph_json": str(graph_out),
            "status": "success"
        })
    
    # Build master graph with cross-domain edges
    print("\n🔗 Building master skill graph with cross-domain links...")
    master_graph = add_cross_domain_skill_edges(all_graphs)
    master_out = graph_root / "DRISHTI_master_graph.json"
    export_graph_to_json_ld(master_graph, master_out)
    
    print(f"\n✅ Master graph nodes: {master_graph.number_of_nodes()}")
    print(f"✅ Master graph edges: {master_graph.number_of_edges()}")
    
    return pd.DataFrame(summary), master_graph