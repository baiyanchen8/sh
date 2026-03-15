import os
import subprocess
import shutil
import sys
from datetime import datetime

# --- 配置 ---
RESULTS_ROOT = "experiment_results"
PYTHON_EXE = sys.executable  # 使用當前啟動的 venv python

# 攻擊與方法清單
ATTACKS = ["badnets", "capsulebd", "mirage"]
METHODS = ["fedavg", "fltrust", "rfout", "proposed"]

def run_task(attack, method, version):
    """
    執行單個實驗任務
    version: 'original' (使用備份檔) 或 'fixed' (使用當前檔案)
    """
    print(f"\n{'='*60}")
    print(f"啟動實驗: [攻擊: {attack}] [方法: {method}] [版本: {version}]")
    print(f"{'='*60}")

    # 1. 決定檔案路徑
    if version == "original":
        if attack == "mnist":
            script_path = os.path.join("backups_original", "mnist_defense.py")
        else:
            script_path = os.path.join("backups_original", attack, f"{attack}_cifar_{method}.py")
    else:
        if attack == "mnist":
            script_path = "mnist_defense.py"
        else:
            script_path = os.path.join(attack, f"{attack}_cifar_{method}.py")

    if not os.path.exists(script_path):
        print(f"跳過：找不到腳本 {script_path}")
        return

    # 2. 建立儲存目錄
    out_dir = os.path.join(RESULTS_ROOT, attack, version, method)
    os.makedirs(out_dir, exist_ok=True)

    # 3. 清理 outputs 目錄
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs")

    # 4. 執行命令
    log_file = os.path.join(out_dir, "execution.log")
    with open(log_file, "w", encoding="utf-8") as f:
        print(f"正在執行... 日誌記錄於 {log_file}")
        process = subprocess.Popen(
            [PYTHON_EXE, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 同時輸出到螢幕與檔案
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        
        process.wait()

    # 5. 移動結果
    if os.path.exists("outputs") and os.listdir("outputs"):
        for item in os.listdir("outputs"):
            src = os.path.join("outputs", item)
            dst = os.path.join(out_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst): shutil.rmtree(dst)
                else: os.remove(dst)
            shutil.move(src, dst)
        print(f"結果已歸檔至: {out_dir}")
    else:
        print(f"警告：{script_path} 執行結束但未產生 outputs 資料。")

def main():
    # 如果需要 MNIST，取消下面這行的註解
    # run_task("mnist", "defense", "fixed")

    # 執行 CIFAR-10 所有組合
    for attack in ATTACKS:
        for method in METHODS:
            # 1. 執行原始版本作為對照組
            run_task(attack, method, "original")
            
            # 2. 執行修正後的版本
            run_task(attack, method, "fixed")

if __name__ == "__main__":
    try:
        main()
        print("\n所有實驗任務已完成！")
    except KeyboardInterrupt:
        print("\n使用者中止實驗。")
