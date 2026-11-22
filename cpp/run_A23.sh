#!/usr/bin/env bash
# A 组种子 2/3 补跑器：等主电池（B2/B3/D）全部完成后自动开跑。
# A 组是 H1 配对 t 检验的对照组，必须有 3 个种子。
set -u
cd "$(dirname "$0")"

wait_battery () {
    while true; do
        local all_done=1
        for arm in B2 B3 D; do
            for seed in 1 2 3; do
                local rf="results_${arm}_s${seed}.txt"
                if [ -f "$rf" ] && grep -q "after=D3" "$rf"; then
                    continue
                fi
                all_done=0
            done
        done
        [ "$all_done" = "1" ] && return 0
        sleep 120
    done
}

run_arm () {
    local arm="$1" seed="$2"
    local rf="results_${arm}_s${seed}.txt"
    while true; do
        if [ -f "$rf" ] && grep -q "after=D3" "$rf"; then
            echo "[A23] $arm s$seed DONE"
            return 0
        fi
        echo "[A23] $arm s$seed resume $(date +%H:%M:%S)"
        ./lesson5_continual.exe arm "$arm" "$seed" 1200 resume >> "battery_${arm}_s${seed}.log" 2>&1
        echo "[A23] $arm s$seed exit=$? $(date +%H:%M:%S)"
        sleep 2
    done
}

echo "[A23] waiting for main battery (B2/B3/D)..."
wait_battery
echo "[A23] battery complete, starting A seed 2/3"
run_arm A 2
run_arm A 3
echo "[A23] ALL DONE $(date)"
