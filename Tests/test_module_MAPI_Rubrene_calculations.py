import unittest
import json
import tables
import os
import numpy as np
from Modules.module_MAPI_Rubrene_DBP.calculations import CalculatedOutputs

data_path = "./Tests/data/MAPI_Rubrene/"
class TestModuleMapiRubreneCalculations(unittest.TestCase):
    
    # print("===============================================")
    # print("mapi_sim_outputs: ", mapi_sim_outputs)
    # print("ru_sim_outputs: ", ru_sim_outputs)
    # print("mapi_params: ", mapi_params)
    # print("rubrene_params: ", rubrene_params)

    mapi_sim_outputs = json.load(
        open(os.path.join(data_path, "mapi_sim_outputs.json")))
    mapi_sim_outputs["N"] = np.array(mapi_sim_outputs["N"])
    mapi_sim_outputs["P"] = np.array(mapi_sim_outputs["P"])

    ru_sim_outputs = json.load(
        open(os.path.join(data_path, "ru_sim_outputs.json")))
    ru_sim_outputs["P_up"] = np.array(ru_sim_outputs["P_up"])
    ru_sim_outputs["T"] = np.array(ru_sim_outputs["T"])
    ru_sim_outputs["delta_S"] = np.array(ru_sim_outputs["delta_S"])
    ru_sim_outputs["delta_D"] = np.array(ru_sim_outputs["delta_D"])

    mapi_params = json.load(
        open(os.path.join(data_path, "mapi_params.json")))
    mapi_params["delta_N"] = np.array(mapi_params["delta_N"])

    rubrene_params = json.load(
        open(os.path.join(data_path, "rubrene_params.json")))

    calculated_outputs = CalculatedOutputs(
        mapi_sim_outputs,
        ru_sim_outputs,
        mapi_params,
        rubrene_params
    )

    def test_calculated_outputs_E_field(self):
        """E_field."""
        E_field = self.calculated_outputs.E_field()
        E_field_expected = np.array(
            json.load(open(os.path.join(data_path, "E_field.json"))))
        np.testing.assert_allclose(E_field, E_field_expected, 1e-30, 1e-16)

    def test_calculated_outputs_E_field_r(self):
        """E_field_r."""
        E_field_r = self.calculated_outputs.E_field_r()
        E_field_r_expected = np.array(
            json.load(open(os.path.join(data_path, "E_field_r.json"))))
        np.testing.assert_allclose(E_field_r, E_field_r_expected, 1e-30, 1e-09)

    def test_calculated_outputs_delta_n(self):
        """delta_n"""
        delta_n = self.calculated_outputs.delta_n()
        delta_n_expected = np.array(
            json.load(open(os.path.join(data_path, "delta_n.json"))))
        np.testing.assert_allclose(delta_n, delta_n_expected, 1e-30, 1e-16)

    def test_calculated_outputs_delta_p(self):
        """delta_p"""
        delta_p = self.calculated_outputs.delta_p()
        delta_p_expected = np.array(
            json.load(open(os.path.join(data_path, "delta_p.json"))))
        np.testing.assert_allclose(delta_p, delta_p_expected, 1e-30, 1e-16)
    
    def test_calculated_outputs_radiative_recombination(self):
        """radiative_recombination"""
        radiative_recombination = \
            self.calculated_outputs.radiative_recombination()
        radiative_recombination_expected = np.array(
            json.load(
                open(os.path.join(data_path, "radiative_recombination.json"))))
        np.testing.assert_allclose(
            radiative_recombination,
            radiative_recombination_expected, 1e-30, 1e-15)

    def test_calculated_outputs_nonradiative_recombination(self):
        """radiative_recombination"""
        nonradiative_recombination = \
            self.calculated_outputs.nonradiative_recombination()
        nonradiative_recombination_expected = np.array(
            json.load(
                open(os.path.join(data_path, "nonradiative_recombination.json"))))
        np.testing.assert_allclose(
            nonradiative_recombination,
            nonradiative_recombination_expected, 1e-30, 1e-15)

    # def test_calculated_outputs_delta_T(self):
    #     """delta_T"""
    #     e = self.calculated_outputs.delta_T()
    #     print(e)

    def test_calculated_outputs_mapi_PL(self):
        """mapi_PL"""
        with \
            tables.open_file(
                os.path.join(
                    data_path, "generated-N.h5"), mode='r') as ifstream_N, \
            tables.open_file(
                os.path.join(
                    data_path, "generated-P.h5"), mode='r') as ifstream_P:
            temp_N = np.array(ifstream_N.root.data)
            temp_P = np.array(ifstream_P.root.data)

        mapi_PL = self.calculated_outputs.mapi_PL(temp_N, temp_P)
        mapi_PL_expected = np.array(
            json.load(
                open(
                    os.path.join(
                        data_path, "mapi_PL.json"))))
        
        np.testing.assert_allclose(
            mapi_PL,
            mapi_PL_expected, 1e-30, 1e-13)
    
    def test_calculated_outputs_dbp_PL(self):
        """dbp_PL"""
        with \
            tables.open_file(
                os.path.join(
                    data_path, "generated-delta_D.h5"), mode='r') as ifstream_D:
            temp_D = np.array(ifstream_D.root.data)

        dbp_PL = self.calculated_outputs.dbp_PL(temp_D)
        dbp_PL_expected = np.array(
            json.load(
                open(
                    os.path.join(
                        data_path, "dbp_PL.json"))))
        
        np.testing.assert_allclose(
            dbp_PL,
            dbp_PL_expected, 1e-30, 1e-13)
    
    def test_calculated_outputs_TTA(self):
        """TTA"""
        with \
            tables.open_file(
                os.path.join(
                    data_path, "generated-T.h5"), mode='r') as ifstream_T:
            temp_T = np.array(ifstream_T.root.data)

        TTA = self.calculated_outputs.TTA(temp_T)
        TTA_expected = np.array(
            json.load(
                open(
                    os.path.join(
                        data_path, "TTA.json"))))
        
        np.testing.assert_allclose(
            TTA,
            TTA_expected, 1e-30, 1e-13)
    
