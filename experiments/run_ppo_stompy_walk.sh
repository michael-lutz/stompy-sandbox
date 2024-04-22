#!/bin/bash

# Shell script to start training PPO Agent

TRAIN_SCRIPT_PATH="../learning/train_ppo.py"

# Setting default parameters
ENV="StompyWalk"
LOG_DIR="logs"
EXP_NAME="stompy_walk_test"
ITERS=1000
MAX_TIMESTEPS=500
LEARNING_RATE=0.0002
GAMMA=0.99
TAU=0.95
CLIP_PARAM=0.1
PPO_EPOCHS=10
BATCH_SIZE=64
UPDATE_INTERVAL=64
ACTION_STD_DECAY_RATE=0.05
MIN_ACTION_STD=0.1
ACTION_STD_DECAY_INTERVAL=250000
RECORD_VIDEO=true
VIDEO_INTERVAL=50
WANDB_PROJECT="stompy_walk_ppo"
DEVICE="cuda:4"

# Start training
python3 -u $TRAIN_SCRIPT_PATH \
    --env $ENV \
    --iters $ITERS \
    --max_timesteps $MAX_TIMESTEPS \
    --learning_rate $LEARNING_RATE \
    --gamma $GAMMA \
    --tau $TAU \
    --clip_param $CLIP_PARAM \
    --ppo_epochs $PPO_EPOCHS \
    --batch_size $BATCH_SIZE \
    --update_interval $UPDATE_INTERVAL \
    --action_std_decay_rate $ACTION_STD_DECAY_RATE \
    --min_action_std $MIN_ACTION_STD \
    --action_std_decay_interval $ACTION_STD_DECAY_INTERVAL \
    $(if [ "$RECORD_VIDEO" = true ]; then echo "--record_video"; fi) \
    --video_interval $VIDEO_INTERVAL \
    --wandb_project $WANDB_PROJECT \
    --log_dir $LOG_DIR \
    --device $DEVICE