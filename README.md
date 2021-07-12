# OneLayer
One-layer transient PL simulations, but in runnable application form

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
