
run() {
    local noise_scale=$1
    local limit_samples=$2
    echo "Perplexity for noise scale $noise_scale and limit samples $limit_samples: $perplexity" 
    python scripts/ppl-llama.py --noise $noise_scale --limit_samples $limit_samples > logs/ppl-llama-noise-$noise_scale-limit-$limit_samples.log 2>&1
    rg "loss" logs/ppl-llama-noise-$noise_scale-limit-$limit_samples.log > logs/ppl-llama-noise-$noise_scale-limit-$limit_samples-loss.log
    python scripts/plot_losses.py logs/ppl-llama-noise-$noise_scale-limit-$limit_samples-loss.log --output logs/ppl-llama-noise-$noise_scale-limit-$limit_samples-loss
}

run 0 200
run 1e-9 200
run 1e-8 200
run 1e-7 200
run 1e-5 200
run 1e-6 200
run 1e-4 200


