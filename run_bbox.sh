export PYTHONPATH=$(pwd)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

python tools/train.py \
    --config_file configs/omnire_bbox.yaml \
    --output_root output \
    --project recon \
    --run_name recon_$1_bbox \
    dataset=waymo/3cams \
    data.scene_idx=$1 \
    data.start_timestep=0 \
    data.end_timestep=-1