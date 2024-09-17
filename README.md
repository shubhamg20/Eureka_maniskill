# Eureka-Maniskill: Applying Eureka on Maniskill Simulator

This repository implements the Eureka framework ([https://eureka-research.github.io/](https://eureka-research.github.io/)) for Maniskill simulator, replicating the functionality originally designed for Isaac Gym. It allows you to train robots on various manipulation tasks using Maniskill's rich environment and diverse task options.

## Installation

1. **Create a conda environment:**

   ```bash
   conda create -n eureka-maniskill python==3.9  

2. **Activate the conda environment:**
   ```bash
   conda activate eureka-maniskill

3. **Install Dependicies:**

   ```bash
   pip install mani_skill
   cd Eureka; pip install -e .

4. **Activate the conda environemt**
   ```bash
   conda activate eureka-maniskill

   ## Eureka-Maniskill: Applying Eureka on Maniskill Simulator

This repository implements the Eureka framework ([https://eureka-research.github.io/](https://eureka-research.github.io/)) for Maniskill simulator([https://maniskill.readthedocs.io/en/latest/index.html](https://maniskill.readthedocs.io/en/latest/index.html)), replicating the functionality originally designed for Isaac Gym. It allows you to train robots on various manipulation tasks using Maniskill's rich environment and diverse task options.

## Usage

### 1. Choose a Task:

Maniskill supports different manipulation tasks. Currently, you can only use one of the following tasks and replace it with the existing value in the `eureka/cfg/config.yaml` file under the `"env"` key:

* PushCube
* PullCube
* LiftPegUpright
* TwoRobotPickCube

### 2. Run the Training:

Once you've chosen your desired task, simply run the following command:

```bash
python eureka/eureka_maniskill.py

## Additional Notes

This repository provides a starting point for using Eureka with Maniskill. Feel free to modify the configuration and experiment with different hyperparameters.
For further details on the Eureka framework itself, please refer to the original project's documentation: https://eureka-research.github.io/
