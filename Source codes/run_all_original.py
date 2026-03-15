import os
import subprocess
import shutil
import sys

# --- 配置 ---
# 這裡指向備份的原始碼路徑
ORIGINAL_ROOT = "backups_original"
RESULTS_ROOT = "experiment_results_original"
PYTHON_EXE = sys.executable

ATTACKS = ["badnets", "capsulebd", "mirage"]
METHODS = ["fedavg", "fltrust", "rfout", "proposed"]

def run_original_task(attack, method):
    print(f"\n{'!'*60}")
    print(f"啟動原始基準測試: [攻擊: {attack}] [方法: {method}]")
    print(f"{'!'*60}")

    # 1. 決定檔案路徑
    if attack == "mnist":
        script_path = os.path.join(ORIGINAL_ROOT, "mnist_defense.py")
    else:
        script_path = os.path.join(ORIGINAL_ROOT, attack, f"{attack}_cifar_{method}.py")

    if not os.path.exists(script_path):
        print(f"跳過：找不到腳本 {script_path}")
        return

    # 2. 建立儲存目錄
    out_dir = os.path.join(RESULTS_ROOT, attack, method)
    os.makedirs(out_dir, exist_ok=True)

    # 3. 清理 outputs 目錄 (重要：原始腳本會把結果存在這裡)
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs")

    # 4. 執行
    log_file = os.path.join(out_dir, "original_run.log")
    with open(log_file, "w", encoding="utf-8") as f:
        # 在 Windows 上，我們需要確保子進程不會崩潰
        # 由於原始碼沒加 if __name__ == '__main__':，我們直接執行它
        print(f"正在執行原始碼... 日誌記錄於 {log_file}")
        process = subprocess.Popen(
            [PYTHON_EXE, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        
        process.wait()

    # 5. 歸檔原始數據
    if os.path.exists("outputs") and os.listdir("outputs"):
        for item in os.listdir("outputs"):
            shutil.move(os.path.join("outputs", item), os.path.join(out_dir, item))
        print(f"原始數據已歸檔至: {out_dir}")
    else:
        print(f"警告：{script_path} 未產生任何 outputs。")

def main():
    if not os.path.exists(ORIGINAL_ROOT):
        print(f"錯誤：找不到備份目錄 {ORIGINAL_ROOT}，無法執行對照組。")
        return

    # 依序執行所有原始攻擊組合
    for attack in ATTACKS:
        for method in METHODS:
            run_original_task(attack, method)

if __name__ == "__main__":
    try:
        main()
        print("\n所有原始基準測試已完成！")
    except KeyboardInterrupt:
        print("\n使用者中止。")
