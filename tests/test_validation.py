import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pyteomics

pyteomics.__path__ = [os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, 'pyteomics'))]

from pyteomics.validation import (
    Diagnostic, Severity, ValidationReport, validate_file, validate_spectra,
    validate_spectrum)


class DiagnosticModelTest(unittest.TestCase):
    def test_diagnostic_serialization(self):
        diagnostic = Diagnostic(
            code='spectrum.array_length_mismatch',
            severity=Severity.ERROR,
            message='m/z and intensity arrays have different lengths',
            location='spectrum[42]',
            suggestion='Ensure every peak has both values',
            context={'mz_length': 2, 'intensity_length': 1},
        )

        self.assertEqual(diagnostic.to_dict(), {
            'code': 'spectrum.array_length_mismatch',
            'severity': 'error',
            'message': 'm/z and intensity arrays have different lengths',
            'location': 'spectrum[42]',
            'suggestion': 'Ensure every peak has both values',
            'context': {'mz_length': 2, 'intensity_length': 1},
        })

    def test_report_groups_diagnostics_by_severity(self):
        report = ValidationReport([
            Diagnostic('test.warning', 'Take care', Severity.WARNING),
            Diagnostic('test.error', 'Broken', Severity.ERROR),
            Diagnostic('test.info', 'For reference', Severity.INFO),
        ])

        self.assertFalse(report.is_valid)
        self.assertEqual([item.code for item in report.errors], ['test.error'])
        self.assertEqual([item.code for item in report.warnings], ['test.warning'])
        self.assertEqual([item.code for item in report.infos], ['test.info'])
        self.assertEqual(len(report), 3)
        self.assertEqual(report.to_dict()['is_valid'], False)

    def test_report_can_be_extended(self):
        report = ValidationReport()
        report.add('test.warning', 'Take care', Severity.WARNING)
        report.extend(ValidationReport([
            Diagnostic('test.info', 'For reference', Severity.INFO),
        ]))

        self.assertTrue(report.is_valid)
        self.assertEqual([item.code for item in report], ['test.warning', 'test.info'])


class SpectrumValidationTest(unittest.TestCase):
    def test_valid_spectrum_has_no_diagnostics(self):
        report = validate_spectrum({
            'm/z array': [100.0, 200.0],
            'intensity array': [10.0, 20.0],
        })

        self.assertTrue(report.is_valid)
        self.assertEqual(len(report), 0)

    def test_missing_peak_array_is_an_error(self):
        report = validate_spectrum({'m/z array': [100.0]})

        self.assertEqual(report.errors[0].code, 'spectrum.array.missing')
        self.assertEqual(report.errors[0].location, 'intensity array')

    def test_different_array_lengths_are_reported(self):
        report = validate_spectrum({
            'm/z array': [100.0, 200.0],
            'intensity array': [10.0],
        }, location='scan=42')

        error = report.errors[0]
        self.assertEqual(error.code, 'spectrum.array.length_mismatch')
        self.assertEqual(error.location, 'scan=42')
        self.assertEqual(error.context, {'mz_length': 2, 'intensity_length': 1})

    def test_peak_value_problems_have_useful_locations(self):
        report = validate_spectrum({
            'm/z array': [200.0, float('nan'), 100.0],
            'intensity array': [10.0, 20.0, -1.0],
        }, location='scan=42')

        self.assertEqual(
            [item.code for item in report],
            ['spectrum.mz.non_finite', 'spectrum.intensity.negative',
             'spectrum.mz.unsorted'],
        )
        self.assertEqual(report[0].location, 'scan=42 peak 1')
        self.assertTrue(report.warnings)

    def test_negative_mz_is_an_error(self):
        report = validate_spectrum({
            'm/z array': [-1.0],
            'intensity array': [10.0],
        })

        self.assertEqual(report.errors[0].code, 'spectrum.mz.negative')
        self.assertEqual(report.errors[0].location, 'peak 0')

    def test_non_numeric_peak_values_are_reported(self):
        report = validate_spectrum({
            'm/z array': ['unknown'],
            'intensity array': ['high'],
        })

        self.assertEqual(
            [item.code for item in report.errors],
            ['spectrum.mz.not_numeric', 'spectrum.intensity.not_numeric'],
        )

    def test_peak_arrays_must_be_sized_iterables(self):
        report = validate_spectrum({
            'm/z array': 100.0,
            'intensity array': 10.0,
        })

        self.assertEqual(report.errors[0].code, 'spectrum.array.invalid')

    def test_iterable_validation_keeps_spectrum_locations(self):
        spectra = [
            {'params': {'title': 'first'},
             'm/z array': [200.0, 100.0], 'intensity array': [2.0, 1.0]},
            {'id': 'scan=2', 'm/z array': [100.0]},
        ]

        report = validate_spectra(spectra)

        self.assertEqual(
            [item.location for item in report],
            ['first', 'intensity array'],
        )

    def test_mgf_file_is_dispatched_and_validated(self):
        text = 'BEGIN IONS\nTITLE=scan one\n200 20\n100 10\nEND IONS\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.mgf'
            path.write_text(text)

            report = validate_file(path)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.warnings[0].code, 'spectrum.mz.unsorted')
        self.assertEqual(report.warnings[0].location, 'scan one')

    def test_mzml_file_uses_the_mzml_reader(self):
        spectrum = {
            'id': 'scan=1',
            'm/z array': [100.0],
            'intensity array': [10.0],
        }
        with patch('pyteomics.mzml.read',
                   return_value=nullcontext([spectrum])) as read:
            report = validate_file('example.mzML')

        read.assert_called_once_with('example.mzML')
        self.assertTrue(report.is_valid)

    def test_file_validator_rejects_unknown_formats(self):
        with self.assertRaisesRegex(ValueError, 'format'):
            validate_file('example.txt')

    def test_file_read_errors_become_diagnostics(self):
        with patch('pyteomics.mgf.read', side_effect=ValueError('bad file')):
            report = validate_file('broken.mgf')

        self.assertFalse(report.is_valid)
        self.assertEqual(report.errors[0].code, 'file.read')
        self.assertEqual(report.errors[0].location, 'broken.mgf')


if __name__ == '__main__':
    unittest.main()
