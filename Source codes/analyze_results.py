import os
import json
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FIXED = "experiment_results"
RESULTS_ORIG = "experiment_results_original"
OUTPUT_DIR = "analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

attacks = ["badnets", "capsulebd", "mirage"]
methods = ["fedavg", "fltrust", "rfout", "proposed"]

summary = {}

def get_metrics(path):
    if not os.path.exists(path): return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            # Handle different JSON structures
            if "results" in data:
                res = data["results"]
            else:
                res = data # proposed format is a list directly or has different keys
            
            if isinstance(res, list):
                # proposed might be a list of dicts
                accs = [r.get("acc", 0) for r in res]
                asrs = [r.get("asr", 0) for r in res]
                return accs, asrs
            elif isinstance(res, dict):
                # some formats might be {"acc": [], "asr": []}
                return res.get("acc", []), res.get("asr", [])
    except:
        return None
    return None

for atk in attacks:
    summary[atk] = {}
    for mtd in methods:
        # Find JSON files
        fixed_path = ""
        orig_path = ""
        
        # Search recursively for any json in the specific method folder
        def find_json(root):
            for r, d, files in os.walk(root):
                for f in files:
                    if f.endswith(".json"): return os.path.join(r, f)
            return ""

        f_json = find_json(os.path.join(RESULTS_FIXED, atk, "fixed", mtd))
        o_json = find_json(os.path.join(RESULTS_ORIG, atk, mtd))
        
        f_data = get_metrics(f_json)
        o_data = get_metrics(o_json)
        
        if f_data or o_data:
            summary[atk][mtd] = {"fixed": f_data, "orig": o_data}

# Generate plots
for atk, mtds in summary.items():
    if not mtds: continue
    
    fig, axes = plt.subplots(len(mtds), 2, figsize=(15, 5 * len(mtds)))
    if len(mtds) == 1: axes = [axes]
    
    for i, (mtd, data) in enumerate(mtds.items()):
        # Accuracy plot
        ax_acc = axes[i][0]
        if data["orig"]:
            ax_acc.plot(data["orig"][0], label="Original", linestyle='--', color='gray')
        if data["fixed"]:
            ax_acc.plot(data["fixed"][0], label="Fixed", color='blue')
        ax_acc.set_title(f"{atk} - {mtd} - Accuracy")
        ax_acc.set_xlabel("Round")
        ax_acc.set_ylabel("ACC (%)")
        ax_acc.legend()
        
        # ASR plot
        ax_asr = axes[i][1]
        if data["orig"]:
            ax_asr.plot(data["orig"][1], label="Original", linestyle='--', color='gray')
        if data["fixed"]:
            ax_asr.plot(data["fixed"][1], label="Fixed", color='red')
        ax_asr.set_title(f"{atk} - {mtd} - Attack Success Rate")
        ax_asr.set_xlabel("Round")
        ax_asr.set_ylabel("ASR (%)")
        ax_asr.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{atk}_comparison.png"))
    plt.close()

# Generate Summary Table for log.md
report = "# Experimental Analysis Summary\n\n"
report += "## 1. Performance Comparison (Original vs Fixed)\n\n"
report += "| Attack | Method | Version | Final ACC | Final ASR | Delta ACC | Delta ASR |\n"
report += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

for atk, mtds in summary.items():
    for mtd, data in mtds.items():
        f_acc = data["fixed"][0][-1] if data["fixed"] else 0
        f_asr = data["fixed"][1][-1] if data["fixed"] else 0
        o_acc = data["orig"][0][-1] if data["orig"] else 0
        o_asr = data["orig"][1][-1] if data["orig"] else 0
        
        report += f"| {atk} | {mtd} | Original | {o_acc:.2f}% | {o_asr:.2f}% | - | - |\n"
        report += f"| {atk} | {mtd} | Fixed | {f_acc:.2f}% | {f_asr:.2f}% | {f_acc-o_acc:+.2f}% | {f_asr-o_asr:+.2f}% |\n"

with open("summary_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
