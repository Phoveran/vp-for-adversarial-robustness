# Visual Prompting for Adversarial Robustness

[paper](https://arxiv.org/abs/2210.06284)

# Install 

`pip install -r requirements.txt`


# Usage

`python gen_prompt.py`


# Code structure


     ./

         attack/ # including all Fast-BAT & Auto Attack related attacks

         model/ # visual_prompt.py

         cfg.py # author style paths

         cifar10_pretrain.py # pretrain a CIFAR10 model

         gen_prompt.py # generate prompt

         evaluate_diff_pgd_steps.py # evaluate using different PGD steps

         losses.py # CW type loss
