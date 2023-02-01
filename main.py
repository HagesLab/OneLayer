#################################################
# Transient Electron Dynamics Simulator
# Model a variety of custom one-dimensional time-variant problems
# Last modified: Nov 25, 2020
# Author: Calvin Fai, Charles Hages
# Contact:
################################################# 

"""
Usage:
  ./main [--module=<module>] [--tab=<tab_index>]

Options:
--module:
    "Standard One-Layer"
    "Nanowire"
    "MAPI-Rubrene/DBP"
    "PN-Junction"
    "Si-DualBand"

--tab:
    "0": Inputs
    "1": Simulate
    "2": Analyze
"""

from docopt import docopt
from config import init_logging

from Notebook.notebook import Notebook
from Modules.Nanowire import Nanowire
from Modules.Std_SingleLayer import Std_SingleLayer
from Modules.module_MAPI_Rubrene_DBP.central import MAPI_Rubrene
from Modules.module_pnJunction.central import PN_Junction
from Modules.module_Si_dualband.central import Si_DualBand

logger = init_logging(__name__)
# {"Display name of module": OneD_Model derived module class}.
MODULE_LIST = {
    "0": Std_SingleLayer,
    "1": Nanowire,
    "2": MAPI_Rubrene,
    "3": PN_Junction,
    "4": Si_DualBand
}

logger = init_logging(__name__)


def get_cli_args():
    """Parses the CLI arguments, verifies their
    validity in the context and returns a dict"""
    raw_args = docopt(__doc__, version='ingest 0.1.0')
    args = {}
    logger.info(raw_args)
    try:
        module = raw_args.get("--module")
        if module is not None:
            module = str(module)
            assert module in MODULE_LIST.keys()
            args["module"] = module
    except AssertionError:
        logger.info("Invalid module \"{}\"".format(module))

    try:
        tab_id = raw_args.get("--tab")
        if tab_id is not None:
            tab_id = int(tab_id)
            assert 0 <= tab_id <= 2
            args["tab_id"] = tab_id
    except AssertionError:
        logger.info("Invalid tab_id \"{}\"".format(tab_id))

    return args


if __name__ == "__main__":
    nb = Notebook("ted", MODULE_LIST, get_cli_args())
    nb.run()
