nvidia-smi

python informative_search.py \
    --base_model '../Llama3.2-3B-Instruct' \
    --run_log '3B_grounding_0.7_novelty_0.4_self' \
    --premise_info_thres 0.7 \
    --conc_threshold 0.4 \
    --is_selfselect \
