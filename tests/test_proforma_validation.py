import os
import unittest

import pyteomics

pyteomics.__path__ = [os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, 'pyteomics'))]

from pyteomics import proforma


class ProFormaValidationTest(unittest.TestCase):
    def test_valid_sequence_has_empty_report(self):
        report = proforma.validate('PEPT[+79.966]IDE/2')

        self.assertTrue(report.is_valid)
        self.assertEqual(len(report), 0)

    def test_syntax_error_is_positioned(self):
        report = proforma.validate('PEPT[+80')

        self.assertFalse(report.is_valid)
        self.assertEqual(report.errors[0].code, 'proforma.syntax')
        self.assertEqual(report.errors[0].location, 'character 8')

    def test_zero_charge_is_invalid(self):
        report = proforma.validate('PEPTIDE/0')

        self.assertFalse(report.is_valid)
        self.assertEqual(report.errors[0].code, 'proforma.charge.zero')

    def test_explanation_describes_sequence_and_modifications(self):
        explanation = proforma.explain('PEPT[+79.966]IDE/2')

        self.assertEqual(explanation['canonical'], 'PEPT[+79.966]IDE/2')
        self.assertEqual(explanation['sequence'], 'PEPTIDE')
        self.assertEqual(explanation['length'], 7)
        self.assertEqual(explanation['charge'], 2)
        self.assertEqual(explanation['modifications'][0]['position'], 4)
        self.assertEqual(explanation['modifications'][0]['residue'], 'T')
        self.assertAlmostEqual(explanation['modifications'][0]['mass'], 79.966)

    def test_explanation_rejects_invalid_sequence(self):
        with self.assertRaises(proforma.ProFormaError):
            proforma.explain('PEPT[+80')

    def test_explanation_keeps_distinct_modification_categories(self):
        sequence = '<13C><[+57]@C>[+1]?{+2}[+3]-PEP(TI)[+4]DE-[+5]/2'

        explanation = proforma.explain(sequence)
        by_category = {
            item['category']: item for item in explanation['modifications']}

        self.assertEqual(explanation['isotopes'], ['13C'])
        self.assertEqual(by_category['fixed']['targets'], ['C'])
        self.assertEqual(by_category['interval']['start'], 4)
        self.assertEqual(by_category['interval']['end'], 5)
        self.assertEqual(
            set(by_category),
            {'fixed', 'unlocalized', 'labile', 'n_term', 'interval', 'c_term'},
        )


if __name__ == '__main__':
    unittest.main()
