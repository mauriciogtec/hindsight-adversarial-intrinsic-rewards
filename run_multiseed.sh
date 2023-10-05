init_seed=0
for reward in "aim" "none"; do
    for outer in {1..5}; do
        for inner in {1..5}; do
            seed=$((init_seed + 5 * (outer - 1) + inner - 1))
            echo "Running python scheduler_experiments/main_scheduler.py --reward $reward --seed $seed"
            (python scheduler_experiments/main_scheduler.py --reward $reward --seed $seed) &
        done
        wait
    done
done