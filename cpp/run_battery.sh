#!/usr/bin/env bash
# 持续学习实验电池全自动驱动器
# 顺序：先收尾 B1 s1（正在跑），然后 B1 s2/s3，再 C/B0/B2/B3，最后 D。
# 每个 run 反复 resume 直到 results 文件出现"全部完成"行。
# 任何中途崩溃（休眠 0x40010004 等）都会由外层 while 自动重跑，state 文件保证断点续训。
set -u
cd "$(dirname "$0")"

run_arm () {
    local arm="$1" seed="$2"
    local rf="results_${arm}_s${seed}.txt"
    while true; do
        if [ -f "$rf" ] && grep -q "全部完成" "$rf"; then
            echo "[battery] $arm s$seed DONE"
            return 0
        fi
        echo "[battery] $arm s$seed resume $(date +%H:%M:%S)"
        ./lesson5_continual.exe arm "$arm" "$seed" 1200 resume >> "battery_${arm}_s${seed}.log" 2>&1
        ec=$?
        echo "[battery] $arm s$seed exit=$ec $(date +%H:%M:%S)"
        sleep 2
    done
}

# 主序：B1 三个种子先行（论文核心组），再对照组，最后 D 手术组（最慢）
for seed in 1 2 3; do run_arm B1 "$seed"; done
for arm in C B0 B2 B3; do
    for seed in 1 2 3; do run_arm "$arm" "$seed"; done
done
for seed in 1 2 3; do run_arm D "$seed"; done
echo "[battery] ALL ARMS COMPLETE $(date)"
