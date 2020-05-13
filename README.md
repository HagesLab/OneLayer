# OneLayer
One-layer transient PL simulations, but in runnable application form

### Installation
Simply download the files **config.txt** and **TEDs.exe**; then run **TEDs.exe** , or...

0. Ensure you have installed Python version 3.7+.
1. Download the files **main.py**, **finite.py**, and **odefuncs.py** to the same directory as **config.txt**.
2. Run **main.py** as you would normally run a Python script - through your favorite IDE or using a command prompt with something akin to `python main.py`.

Alternatively, [PyInstaller] (https://pyinstaller.readthedocs.io/en/v3.6/usage.html) provides a straightforward method to package source files into an executable. This can be done with TEDs code by...

0. Install PyInstaller with `pip install pyinstaller`, `conda install pyinstaller`, or similar command in the appropriate command prompt.
1. Navigate to the directory containing the four files listed above.
2. Run the command `pyinstaller main.py --onefile -n TEDs`. This creates several directories containing some intermediate files PyInstaller uses, but the executable will be generated in the **dist** directory. Replacing the argument "TEDs" changes the name of the generated executable.
3. Move the executable out of the **dist** directory and into the same directory as **config.txt**.
