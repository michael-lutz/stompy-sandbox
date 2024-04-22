# Stompy's Sandbox

This repository provides a simple way to train and test RL policies on Stompy, a bipedal robot created by K-Scale Labs. The simulation runs on Mujoco and implements the Gym interface.

## Installation
Clone the repository:
```zsh
git clone
```

I recommend creating a new conda enviroment to run the code. Run the following commands to create a new conda environment and activate it:
```zsh
conda create -n stompy python=3.8
conda activate stompy
```

Run the following commands to install the required packages in requirements.txt
```zsh
pip install -r requirements.txt
pip install -e
```

## Usage
To train a policy, navigate to the experiments folder and run the following command:
```zsh
zsh run_ppo_stompy_walk.sh
```

Adjust hyperparameters in the any of the .sh files in the experiments folder.

## Todo
- [ ] Implement MJX to improve scalability
- [ ] Speed up rendering! (Currently very slow)
- [ ] Collect high quality demonstrations of walking and recovering from falls
- [ ] Add AMP discriminator & training setup