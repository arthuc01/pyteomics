import os
import unittest

import pyteomics

pyteomics.__path__ = [os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, 'pyteomics'))]

from pyteomics import proforma
from pyteomics.annotation import annotate_spectrum


class SpectrumAnnotationTest(unittest.TestCase):
    def setUp(self):
        peptide = proforma.ProForma.parse('PEPTIDE')
        self.b1 = peptide.fragments('b', charge=1)[0]
        self.y1 = peptide.fragments('y', charge=1)[0]

    def test_matches_b_and_y_ions_with_ppm_tolerance(self):
        spectrum = {
            'm/z array': [self.b1 * (1 + 5e-6), self.y1 * (1 - 4e-6), 999.0],
            'intensity array': [100.0, 80.0, 5.0],
        }

        result = annotate_spectrum(spectrum, 'PEPTIDE', tolerance=10)

        self.assertEqual([match.label for match in result.matches], ['b1^1', 'y1^1'])
        self.assertEqual([match.peak_index for match in result.matches], [0, 1])
        self.assertAlmostEqual(result.matches[0].mass_error, 5.0, places=6)
        self.assertEqual(result.matches[0].mass_error_unit, 'ppm')
        self.assertEqual(result.unmatched_peak_indices, [2])
        self.assertAlmostEqual(result.coverage, 2 / 12)
        self.assertTrue(result.diagnostics.is_valid)

    def test_supports_absolute_tolerance(self):
        spectrum = {
            'm/z array': [self.b1 + 0.01],
            'intensity array': [100.0],
        }

        result = annotate_spectrum(
            spectrum, 'PEPTIDE', ion_types=('b',), charges=(1,),
            tolerance=0.02, tolerance_unit='Da')

        self.assertEqual(len(result.matches), 1)
        self.assertAlmostEqual(result.matches[0].mass_error, 0.01)
        self.assertEqual(result.matches[0].mass_error_unit, 'Da')

    def test_invalid_spectrum_returns_diagnostics_instead_of_crashing(self):
        result = annotate_spectrum(
            {'m/z array': [self.b1]}, 'PEPTIDE')

        self.assertEqual(result.matches, [])
        self.assertEqual(result.diagnostics.errors[0].code, 'spectrum.array.missing')

    def test_rejects_unknown_tolerance_unit(self):
        spectrum = {'m/z array': [], 'intensity array': []}

        with self.assertRaisesRegex(ValueError, "tolerance_unit"):
            annotate_spectrum(spectrum, 'PEPTIDE', tolerance_unit='furlong')

    def test_result_can_be_serialized(self):
        spectrum = {
            'm/z array': [self.b1],
            'intensity array': [100.0],
        }

        payload = annotate_spectrum(
            spectrum, 'PEPTIDE', ion_types=('b',), charges=(1,)).to_dict()

        self.assertEqual(payload['matches'][0]['label'], 'b1^1')
        self.assertIn('diagnostics', payload)
        self.assertIn('coverage', payload)


if __name__ == '__main__':
    unittest.main()
