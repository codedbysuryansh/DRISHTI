import time
import pandas as pd
from pathlib import Path

from video_loader import get_all_video_paths
from preprocessing import process_video_batch
from batch_hand_pipeline import run_batch_hand_detection
from batch_object_pipeline import run_batch_object_detection
from batch_contact_and_segment import run_batch_contact_and_segmentation
from batch_kinematic import run_batch_kinematics
from batch_intent import run_batch_intent_inference
from batch_robot_skill_and_graph import run_batch_robot_skill_and_graph

class DRISHTIPipeline:
    """
    Full end-to-end DRISHTI pipeline orchestrator.
    """
    def __init__(self, data_dir, output_root, domains):
        self.data_dir = Path(data_dir)
        self.output_root = Path(output_root)
        self.domains = domains
        self.timings = {}
        self.results = {}
    
    def run(self, 
            sample_rate=30, 
            blur_threshold=100,
            dist_thresh_px=25,
            min_len_frames=5,
            gap_merge_frames=3,
            yolo_model="yolov8n.pt",
            enable_small_objects=True):
        
        print("=" * 70)
        print("🔱 DRISHTI — FULL PIPELINE EXECUTION")
        print("=" * 70)
        print(f"📂 Data directory: {self.data_dir}")
        print(f"📂 Output directory: {self.output_root}")
        print(f"🎯 Domains: {', '.join(self.domains)}")
        print("=" * 70)
        
        # STAGE 1: Video scanning
        print("\n[1/8] 🔍 Scanning videos...")
        t0 = time.time()
        video_records = get_all_video_paths(self.data_dir, self.domains)
        self.results['video_records'] = video_records
        self.timings['scan'] = time.time() - t0
        print(f"   ✅ Found {len(video_records)} videos | Time: {self.timings['scan']:.2f}s")
        
        if not video_records:
            print("❌ No videos found. Check data directory.")
            return
        
        # STAGE 2: Preprocessing
        print("\n[2/8] 🎬 Frame extraction & preprocessing...")
        t0 = time.time()
        from preprocessing import process_video_batch
        preprocess_summary = process_video_batch(
            video_records,
            output_dir=str(self.output_root / "sampled_frames"),
            sample_rate=sample_rate,
            blur_threshold=blur_threshold
        )
        self.results['preprocess'] = preprocess_summary
        self.timings['preprocess'] = time.time() - t0
        success = sum([1 for r in preprocess_summary if r['status'] == 'success'])
        print(f"   ✅ Processed {success}/{len(video_records)} videos | Time: {self.timings['preprocess']:.2f}s")
        
        # STAGE 3: Hand detection
        print("\n[3/8] 🤲 Hand landmark detection...")
        t0 = time.time()
        hand_summary = run_batch_hand_detection(
            frames_root_dir=str(self.output_root / "sampled_frames"),
            out_root_dir=str(self.output_root / "hand_landmarks"),
            save_annotated_examples=True,
            max_annotated_examples_per_video=5
        )
        self.results['hands'] = hand_summary
        self.timings['hands'] = time.time() - t0
        print(f"   ✅ Detected hands in {len(hand_summary)} videos | Time: {self.timings['hands']:.2f}s")
        
        # STAGE 4: Object detection
        print("\n[4/8] 📦 Object detection...")
        t0 = time.time()
        obj_summary = run_batch_object_detection(
            frames_root_dir=str(self.output_root / "sampled_frames"),
            out_root_dir=str(self.output_root / "object_detections"),
            model_name=yolo_model,
            save_annotated_examples=True,
            max_annotated_examples_per_video=5,
            enable_small_object_proposals=enable_small_objects
        )
        self.results['objects'] = obj_summary
        self.timings['objects'] = time.time() - t0
        print(f"   ✅ Detected objects in {len(obj_summary)} videos | Time: {self.timings['objects']:.2f}s")
        
        # STAGE 5: Contact & Segmentation
        print("\n[5/8] 🔗 Contact detection & action segmentation...")
        t0 = time.time()
        seg_summary = run_batch_contact_and_segmentation(
            hand_root_dir=str(self.output_root / "hand_landmarks"),
            obj_root_dir=str(self.output_root / "object_detections"),
            out_root_dir=str(self.output_root),
            dist_thresh_px=dist_thresh_px,
            min_len_frames=min_len_frames,
            gap_merge_frames=gap_merge_frames
        )
        self.results['segments'] = seg_summary
        self.timings['segments'] = time.time() - t0
        total_segs = seg_summary['segments'].sum() if 'segments' in seg_summary.columns else 0
        print(f"   ✅ Extracted {total_segs} action segments | Time: {self.timings['segments']:.2f}s")
        
        # STAGE 6: Kinematic Analysis (Layer 1)
        print("\n[6/8] 🤲 Kinematic analysis (Layer 1: HOW)...")
        t0 = time.time()
        kin_summary = run_batch_kinematics(
            hand_root_dir=str(self.output_root / "hand_landmarks"),
            segments_root_dir=str(self.output_root / "segments"),
            out_root_dir=str(self.output_root / "kinematics")
        )
        self.results['kinematics'] = kin_summary
        self.timings['kinematics'] = time.time() - t0
        print(f"   ✅ Kinematic features extracted | Time: {self.timings['kinematics']:.2f}s")
        
        # STAGE 7: Intent Inference (Layer 2)
        print("\n[7/8] 🎯 Intent inference (Layer 2: WHY)...")
        t0 = time.time()
        intent_summary = run_batch_intent_inference(
            kinematics_root_dir=str(self.output_root / "kinematics"),
            objects_root_dir=str(self.output_root / "object_detections"),
            segments_root_dir=str(self.output_root / "segments"),
            out_root_dir=str(self.output_root / "intent")
        )
        self.results['intent'] = intent_summary
        self.timings['intent'] = time.time() - t0
        print(f"   ✅ Intent labels generated | Time: {self.timings['intent']:.2f}s")
        
        # STAGE 8: Robot Skill Mapping & Graph (Layer 3)
        print("\n[8/8] 🤖 Robot skill mapping & skill graph generation (Layer 3: WHAT)...")
        t0 = time.time()
        skill_summary, master_graph = run_batch_robot_skill_and_graph(
            intent_root_dir=str(self.output_root / "intent"),
            out_root_dir=str(self.output_root)
        )
        self.results['robot_skills'] = skill_summary
        self.results['master_graph'] = master_graph
        self.timings['robot_skills'] = time.time() - t0
        print(f"   ✅ Hierarchical skill graph built | Time: {self.timings['robot_skills']:.2f}s")
        
        # Summary
        total_time = sum(self.timings.values())
        print("\n" + "=" * 70)
        print("✅ DRISHTI PIPELINE COMPLETE")
        print("=" * 70)
        print(f"⏱️  Total runtime: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"📊 Master graph: {master_graph.number_of_nodes()} nodes, {master_graph.number_of_edges()} edges")
        print(f"📁 All outputs saved to: {self.output_root}")
        print("=" * 70)
        
        return self.results, self.timings
    
    def generate_summary_report(self):
        """Generate a markdown summary report."""
        report_path = self.output_root / "DRISHTI_PIPELINE_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# 🔱 DRISHTI Pipeline Execution Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Summary\n\n")
            f.write(f"- **Videos processed:** {len(self.results.get('video_records', []))}\n")
            
            if 'segments' in self.results:
                total_segs = self.results['segments']['segments'].sum()
                f.write(f"- **Action segments extracted:** {total_segs}\n")
            
            if 'master_graph' in self.results:
                g = self.results['master_graph']
                f.write(f"- **Skill graph nodes:** {g.number_of_nodes()}\n")
                f.write(f"- **Skill graph edges:** {g.number_of_edges()}\n")
            
            f.write(f"\n## ⏱️ Stage Timings\n\n")
            f.write("| Stage | Time (s) |\n")
            f.write("|-------|----------|\n")
            for stage, t in self.timings.items():
                f.write(f"| {stage} | {t:.2f} |\n")
            
            total = sum(self.timings.values())
            f.write(f"| **TOTAL** | **{total:.2f}** |\n")
        
        print(f"📄 Summary report saved: {report_path}")