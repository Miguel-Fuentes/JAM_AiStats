# JAM-PGM Codebase

This codebase is a collection of Python scripts for replicating the results in the "Joint Selection: Adaptively Incorporating Public Information for Private Synthetic Data" paper.

**Importantly** this code is built on top of the [mbi](https://github.com/ryan112358/private-pgm/tree/master) library which includes the code for running Private-PGM. In order to perform the experiments with a version of AIM that uses bounded differential privacy, the mbi code was included in this codebase and modified to include the bounded bounded_aim.py script.

To be clear, the following scripts were written of the authors of the JAM-PGM paper:
- `jam_pgm.py`
- `run_bounded_aim.py`
- `utils/accounting.py`
- `utils/measurement.py`
- `utils/selection.py`
- `utils/workload.py`
- `mechanisms/bounded_aim.py` (very minor modification to the original aim.py script)

All other scripts were written by the authors of the mbi library and can be found in the [mbi](https://github.com/ryan112358/private-pgm/tree/master) repository.

## Installation
In order to install this version of the mbi library and the dependencies for the JAM-PGM codebase, run the following command:

```pip install -r requirements.txt```

## Running Experiments
In order to run JAM-PGM on one of the data sets in the data folder with a given privacy budget, run the following command:

```python jam.py <dataset_name> --epsilon <epsilon> --delta <delta> --rounds <number of rounds>```

Additional optional parameters are described in the ``jam.py`` script.

Similarly, in order to run AIM on one of the data sets in the data folder run:

```python run_bounded_aim.py <dataset_name> --epsilon <epsilon> --delta <delta>```

Again, optional parameters are described in the ``run_bounded_aim.py`` script.