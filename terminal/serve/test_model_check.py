"""#6621 (the Strata project's 0.2.20): a request naming a catalog model that is not the loaded one is answered 404, except
the loaded model's own family name ("hy4-flash" for "hy4-flash-sq-64") and, with switch_on_request (#38), an installed model
the server loads for it.

    python -m serve.test_model_check
"""
import unittest
from types import SimpleNamespace

import serve.server as server


def service(model="hy4-flash-sq-64", installed=None):
    installed = installed or {}
    return SimpleNamespace(model=model, aliases=[], strict_model=False,
                           _STRATA_MODEL_PREFIXES=server.Service._STRATA_MODEL_PREFIXES,
                           model_names=lambda: [model], requested_model=lambda name: installed.get(name))


def check(svc, name):
    return server.Service.check_model(svc, {"model": name})


class ModelCheck(unittest.TestCase):
    def test_the_loaded_model_and_its_family_name_are_served(self):
        self.assertIsNone(check(service(), "hy4-flash-sq-64"))
        self.assertIsNone(check(service(), "hy4-flash"))

    def test_another_catalog_model_is_404(self):
        err = check(service(), "qwen3.6-35b-a3b-ud-q3_k_m")
        self.assertEqual(err["code"], "model_not_found")
        self.assertIsNotNone(check(service(), "hy4-flash-sq-96"))

    def test_an_installed_model_that_switch_on_request_loads_is_served(self):
        svc = service(installed={"qwen3.6-35b-a3b-ud-q3_k_m": {"config": "strata-qwen36-ud-q3_k_m.json"}})
        self.assertIsNone(check(svc, "qwen3.6-35b-a3b-ud-q3_k_m"))

    def test_other_names_are_left_to_the_loaded_model(self):
        self.assertIsNone(check(service(), "gpt-4"))


if __name__ == "__main__":
    unittest.main()
