python  cli/eplb_generate_map_ds.py \
--json_folder ./ \
--num_replicas 512  \
--num_groups  16 \
--num_nodes  16 \
--num_gpus 256 \
--output_path mindspeed_rl/workers/eplb/expert_map.json \