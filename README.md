
## Installation: 
To install please create an environment

```bash
conda create -n qreps python=3.7
```

Then activate the environment
```bash
conda activate qreps
```

Install the library locally in edit mode
```bash
pip install -e.[experiments]
```

## Figure 1: 
To run the experiments move to each folder and runner the runner files. 
To reproduce the experiments of Figure 1 run 
```bash
python exps/logistic_vs_squared_bellman_error.py
```

## Figure 2: 
To run the experiments of Figure 2 move to the environments folder
```bash
cd exps/environments
```
and from there launch all experiments:
```bash
python launch_experiments.py 
```
To merge the results of the different random seeds run:
```bash
python merge_results.py $ENV_NAME
```
and replace on ENV_NAME the environment you want to merge. 
Finally, run the plotter 
```bash
python plot_all.py
```

### Note: 
If this takes too long and rather just run a single random seed per environment, please move to each environment folder. 
Within each folder run the runner and the plotter. 
For example, for the two state stochastic environment do:
```bash
cd exps/environments/two_state_stochastic
```
and then run 
```bash
python two_state_stochastic_run.py
```
to reproduce the experiment. To plot just run 
```bash
python two_state_stochastic_plot.py
```

## Figure 4.
To reproduce the action_gap experiments, go to:
```bash
cd exps/action_gap
```
and then run 
```bash
python action_gap_run.py
```
to reproduce the experiment. To plot just run 
```bash
python action_gap_plot.py
```


## Figure 5.
To reproduce the bias experiments, go to:
```bash
cd exps/bias
```
and then run 
```bash
python bias_run.py
```
to reproduce the experiment. To plot just run 
```bash
python bias_plot.py
```