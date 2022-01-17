#################################################
# Transient Electron Dynamics Simulator
# Model a variety of custom one-dimensional time-variant problems
# Last modified: Nov 25, 2020
# Author: Calvin Fai, Charles Hages
# Contact:
################################################# 
        
from Notebook.notebook import Notebook

from Modules.Nanowire import Nanowire
from Modules.Std_SingleLayer import Std_SingleLayer
from Modules.MAPI_Rubrene_DBP import MAPI_Rubrene


# Telling TED what modules are available.
# {"Display name of module": OneD_Model derived module class}.
MODULE_LIST = {
    "Standard One-Layer": Std_SingleLayer,
    "Nanowire": Nanowire,
    "MAPI-Rubrene/DBP": MAPI_Rubrene
}

if __name__ == "__main__":
    nb = Notebook("ted", MODULE_LIST)
    nb.run()
