## Python environment installation instructions
- Make sure you have conda installed in your system. [Instructions link here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
- Then, get the `conda_env.yml` file, and from the same directory, run `conda env create -f conda_env.yml`. 
- Activate the environment - `conda activate ddrl_a4`

## Running the code
- Command for training: `python run.py --alg <name> --seed <int value> --env <env name>`
