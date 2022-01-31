LM= #LM_DIR
#####################################################
for seed in 666 42 1024;
do
CUDA_VISIBLE_DEVICES=3 python experiment.py --env hopper --dataset medium --model_type dt -w --seed $seed --pretrained_lm $LM  --outdir "checkpoints/chibiv2_kmeans_medium_positions_hopper_$seed" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1 --dropout 0.2 --share_input_output_proj &
done
for seed in 666 42 1024;
do
CUDA_VISIBLE_DEVICES=4 python experiment.py --env walker2d --dataset medium --model_type dt -w --seed $seed  --pretrained_lm $LM  --outdir "checkpoints/chibiv2_kmeans_medium_positions_walker_$seed" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
done
for seed in 666 42 1024;
do
CUDA_VISIBLE_DEVICES=5 python experiment.py --env halfcheetah --dataset medium --model_type dt -w --seed $seed  --pretrained_lm $LM  --outdir "checkpoints/chibiv2_kmeans__medium_positions_halfcheetah_$seed"  --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
done
#####################################################
wait
