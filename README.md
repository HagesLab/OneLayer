# OneLayer
Our internal GUI we use to analyze individual transient carrier simulations from projects such as MetroTRPL or Bayesian-Inference-TRPL. Or, SCAPS-1D but transient. Supports custom carrier models / device stacks.

Unfortunately, for lack of time, we no longer maintain this project actively. We nevertheless hope that this project is useful to anyone wanting to run transient carrier simulations.

### Installation

0. Recommended prerequisites: use Python 3.8+ and the following library versions:
  - Numpy 1.19+
  - Matplotlib 3.2.2+
  - Scipy 1.5.0+
  - tk 8.6.8+
  - PyTables 3.6.1+
  
Other versions may work and will not affect the accuracy of results if they do, but if errors occur when importing libraries, try these versions.

1. Download this repository.
2. Run **main.py** as you would normally run a Python script - through your favorite IDE or using a command prompt with something akin to `python main.py`.

Alternatively, [PyInstaller](https://pyinstaller.readthedocs.io/en/v3.6/usage.html) provides a straightforward method to package source files into an executable. This can be done with TEDs code after downloading this repository by...

0. Install PyInstaller with `pip install pyinstaller`, `conda install pyinstaller`, or similar command in the appropriate command prompt.
1. Navigate to this repository.
2. Run the command `pyinstaller main.py --onefile -n TEDs`. This creates several directories containing some intermediate files PyInstaller uses, but the executable will be generated in the **dist** directory. Replacing the argument "TEDs" changes the name of the generated executable.

### Tests
Add test scripts using unittest with similar structure to existing scripts in Tests/ directory. 
Run tests by:
```python -m unittest discover Tests```

One can also track the progress of code and its test coverage with:
```python -m coverage run --source='.' -m unittest discover Tests && python -m coverage report```
(need to install "coverage" by: ```pip install coverage```)
