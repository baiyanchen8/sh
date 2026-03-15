#!/bin/bash

# Federated Learning Backdoor Attack & Defense Orchestrator (venv version)
# 確保您在執行前已 source venv/bin/activate

RESULTS_ROOT="experiment_results"
mkdir -p $RESULTS_ROOT

run_task() {
    local attack=$1
    local method=$2
    local version=$3

    echo "=========================================================="
    echo "STARTING: Attack=$attack, Method=$method, Version=$version"
    echo "=========================================================="

    local out_dir="$RESULTS_ROOT/$attack/$version/$method"
    mkdir -p "$out_dir"

    local script_path=""
    if [ "$version" == "original" ]; then
        if [ "$attack" == "mnist" ]; then
            script_path="backups_original/mnist_defense.py"
        else
            script_path="backups_original/$attack/${attack}_cifar_${method}.py"
        fi
    else
        if [ "$attack" == "mnist" ]; then
            script_path="mnist_defense.py"
        else
            script_path="$attack/${attack}_cifar_${method}.py"
        fi
    fi

    if [ ! -f "$script_path" ]; then
        echo "Error: Script $script_path not found. Skipping."
        return
    fi

    # 確保 outputs 目錄是空的，避免數據混淆
    rm -rf outputs/*
    mkdir -p outputs

    # 執行實驗
    python "$script_path" 2>&1 | tee "$out_dir/execution.log"

    # 歸檔結果
    if [ -d "outputs" ] && [ "$(ls -A outputs)" ]; then
        mv outputs/* "$out_dir/"
        echo "Success: Results moved to $out_dir"
    else
        echo "Warning: No output files found in outputs/ for $script_path"
    fi
}

# --- 執行序列 ---

# 1. MNIST (選擇性執行)
# run_task "mnist" "defense" "fixed"

# 2. CIFAR-10 所有組合
ATTACKS=("badnets" "capsulebd" "mirage")
METHODS=("fedavg" "fltrust" "rfout" "proposed")

for ATK in "${ATTACKS[@]}"; do
    for MTD in "${METHODS[@]}"; do
        # 如果需要跑原始碼對照，取消下面這行的註解:
        # run_task "$ATK" "$MTD" "original"
        
        # 跑修正後的版本
        run_task "$ATK" "$MTD" "fixed"
    done
done

echo "=========================================================="
echo "ALL EXPERIMENTS COMPLETED."
echo "Final results located in: $RESULTS_ROOT"
echo "=========================================================="
