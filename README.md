# Project 2: Defining and Solving Reinforcement Learning Environments

Author: Minuu Lakshmi Shanmugam, B01157604
Course: CS 4/565 Introduction to AI, Fall 2026
Due Oct 26, 2026 at 11:59 PM

## What this project does

This project defines a 5 by 4 grid navigation task, implements it in Gymnasium form, validates it with a random agent, and solves it with Q learning and with SARSA. The deterministic version always carries out the requested move. The stochastic version carries out the requested move with probability 0.70 and slips to each other move with probability 0.10.

The grid has 20 positions. Each position is combined with 8 collected states, one for each subset of the three special cells, so the observation space has 160 states. Actions are down, up, right, and left. Start is (0, 0). Goal is (4, 3). Special cells are (0, 3) with reward 1.0, (2, 3) with reward 2.0, and (3, 2) with reward 3.0. Each special reward is collected only on the first visit in an episode. An ordinary step costs 0.1. Reaching the goal gives 5.0 and ends the episode. An episode is cut off at 100 steps.

The best possible deterministic total is 10.5, in 9 steps, collecting all three specials and then reaching the goal.

## Files in this folder

1. Project2_Report.pdf: combined report for Part 1 and Part 2.
2. Project2.ipynb: executed Jupyter Notebook with all code, traces, plots, and results.
3. main.py: same code as the notebook, as a single program that regenerates all plots and pickle files.
4. README.md: this file.
5. outputs/deterministic_q_table.pkl: Q table for the deterministic environment, shape 160 by 4.
6. outputs/nondeterministic_q_table.pkl: Q table for the stochastic environment, shape 160 by 4.
7. outputs/sarsa_q_table.pkl: SARSA table for the deterministic environment, shape 160 by 4.
8. outputs/plots: all plots used in the report, including environment images, Q learning plots, SARSA plot, and tuning plots. There are 21 plots in total.
9. requirements.txt: pinned package versions for gymnasium, numpy, and matplotlib.

## Reproduction

1. Install the required packages with this command: pip install -r requirements.txt. That installs gymnasium, numpy, and matplotlib. If gymnasium is missing, the code cannot run, so install first and then run.
2. Run from this folder with this command: python main.py
3. Or open Project2.ipynb and run all cells from top to bottom.
4. Plots are written to outputs/plots. Pickle files are written to outputs.
5. There are no command line options. All settings are in the code: seed 2026, 500 episodes, learning rate 0.3, discount factor 0.9, epsilon start 1.0, epsilon minimum 0.05, epsilon decay 0.99 for the main runs.
6. A full run takes under one minute on a standard laptop, including all tuning runs. No separate long running experiment is needed.

Versions used for the reported results: Python 3.12.3, Gymnasium 1.4.0, NumPy 2.5.4, Matplotlib 3.11.2. Seed 2026 is used throughout.

## Results in brief

1. Q learning, deterministic: last 50 episode mean 10.47. Greedy rollout reaches the goal in 9 steps with total reward 10.5.
2. Q learning, stochastic: last 50 episode mean 8.89. Greedy rollout reaches the goal in 19 steps with total reward 9.5.
3. SARSA, deterministic: last 50 episode mean 10.36. Greedy rollout reaches the goal in 9 steps with total reward 10.5.
4. Tuning: four parameters, three values each, twelve runs, plus four comparison figures and twelve detail figures, for 16 tuning figures in total.
5. Recommended setting: learning rate 0.3, discount factor 0.9, epsilon decay 0.95, epsilon start 1.0, epsilon minimum 0.05, 500 episodes.
