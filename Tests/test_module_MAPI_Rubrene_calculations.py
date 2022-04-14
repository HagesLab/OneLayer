import unittest
import json
import numpy as np
from Modules.module_MAPI_Rubrene_DBP.calculations import CalculatedOutputs


class TestModuleMapiRubreneCalculations(unittest.TestCase):
    
    # print("===============================================")
    # print("mapi_sim_outputs: ", mapi_sim_outputs)
    # print("ru_sim_outputs: ", ru_sim_outputs)
    # print("mapi_params: ", mapi_params)
    # print("rubrene_params: ", rubrene_params)

    mapi_sim_outputs = json.load(open("./Tests/data/MAPI_Rubrene/mapi_sim_outputs.json"))
    mapi_sim_outputs["N"] = np.array(mapi_sim_outputs["N"])
    mapi_sim_outputs["P"] = np.array(mapi_sim_outputs["P"])

    ru_sim_outputs = json.load(open("./Tests/data/MAPI_Rubrene/ru_sim_outputs.json"))
    ru_sim_outputs["P_up"] = np.array(ru_sim_outputs["P_up"])
    ru_sim_outputs["T"] = np.array(ru_sim_outputs["T"])
    ru_sim_outputs["delta_S"] = np.array(ru_sim_outputs["delta_S"])
    ru_sim_outputs["delta_D"] = np.array(ru_sim_outputs["delta_D"])

    mapi_params = json.load(open("./Tests/data/MAPI_Rubrene/mapi_params.json"))
    mapi_params["delta_N"] = np.array(mapi_params["delta_N"])

    rubrene_params = json.load(open("./Tests/data/MAPI_Rubrene/rubrene_params.json"))

    calculated_outputs = CalculatedOutputs(
        mapi_sim_outputs,
        ru_sim_outputs,
        mapi_params,
        rubrene_params
    )

    def test_calculated_outputs_E_field(self):
        """E_field."""
        E_field = self.calculated_outputs.E_field()
        E_field_expected = np.array(json.load(open("./Tests/data/MAPI_Rubrene/E_field.json")))
        self.assertEqual(E_field.shape, E_field_expected.shape)
        np.testing.assert_allclose(E_field, E_field_expected, 1e-30, 1e-16)

    def test_calculated_outputs_E_field_r(self):
        """E_field_r."""
        E_field_r = self.calculated_outputs.E_field_r()
        E_field_r_expected = np.array(json.load(open("./Tests/data/MAPI_Rubrene/E_field_r.json")))
        self.assertEqual(E_field_r.shape, E_field_r_expected.shape)
        np.testing.assert_allclose(E_field_r, E_field_r_expected, 1e-30, 1e-09)

    def test_calculated_outputs_delta_n(self):
        """delta_n"""
        delta_n = self.calculated_outputs.delta_n()
        delta_n_expected = np.array(json.load(open("./Tests/data/MAPI_Rubrene/delta_n.json")))
        self.assertEqual(delta_n.shape, delta_n_expected.shape)
        np.testing.assert_allclose(delta_n, delta_n_expected, 1e-30, 1e-16)
