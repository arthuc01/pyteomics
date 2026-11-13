"""Structured validation diagnostics shared across Pyteomics modules."""

import math
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(str, Enum):
    """Severity level for a validation diagnostic."""

    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'


@dataclass(frozen=True)
class Diagnostic:
    """A single machine-readable validation finding."""

    code: str
    message: str
    severity: Severity = Severity.ERROR
    location: Optional[str] = None
    suggestion: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'severity': self.severity.value,
            'message': self.message,
            'location': self.location,
            'suggestion': self.suggestion,
            'context': dict(self.context),
        }


class ValidationReport(Sequence[Diagnostic]):
    """An ordered collection of validation diagnostics."""

    def __init__(self, diagnostics: Optional[Iterable[Diagnostic]] = None):
        self._diagnostics = list(diagnostics or ())

    def __len__(self) -> int:
        return len(self._diagnostics)

    def __getitem__(self, index):
        return self._diagnostics[index]

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._diagnostics)

    @property
    def errors(self) -> list[Diagnostic]:
        return [item for item in self if item.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [item for item in self if item.severity is Severity.WARNING]

    @property
    def infos(self) -> list[Diagnostic]:
        return [item for item in self if item.severity is Severity.INFO]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(
        self,
        code: str,
        message: str,
        severity: Severity = Severity.ERROR,
        *,
        location: Optional[str] = None,
        suggestion: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> Diagnostic:
        diagnostic = Diagnostic(
            code=code,
            message=message,
            severity=severity,
            location=location,
            suggestion=suggestion,
            context=context or {},
        )
        self._diagnostics.append(diagnostic)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self._diagnostics.extend(diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'diagnostics': [item.to_dict() for item in self],
        }


def _peak_location(spectrum_location: Optional[str], peak_index: int) -> str:
    if spectrum_location:
        return '{} peak {}'.format(spectrum_location, peak_index)
    return 'peak {}'.format(peak_index)


def validate_spectrum(spectrum, *, location: Optional[str] = None) -> ValidationReport:
    """Check the two peak arrays shared by MGF and mzML spectra.

    The validator reports all independent peak problems it can find. It only
    stops early when an array is missing or their lengths differ, because
    pairing peaks after that point would give misleading locations.
    """
    report = ValidationReport()
    required_arrays = ('m/z array', 'intensity array')

    for array_name in required_arrays:
        if array_name not in spectrum:
            report.add(
                'spectrum.array.missing',
                'The spectrum is missing its {}.'.format(array_name),
                location=array_name,
                suggestion='Provide both m/z and intensity values for every peak.',
            )
    if report.errors:
        return report

    mz_values = spectrum['m/z array']
    intensity_values = spectrum['intensity array']
    array_lengths = {}
    for array_name, values in (
            ('m/z array', mz_values), ('intensity array', intensity_values)):
        try:
            array_lengths[array_name] = len(values)
            iter(values)
        except TypeError:
            report.add(
                'spectrum.array.invalid',
                'The {} is not a sized iterable.'.format(array_name),
                location=array_name,
            )
    if report.errors:
        return report

    if array_lengths['m/z array'] != array_lengths['intensity array']:
        report.add(
            'spectrum.array.length_mismatch',
            'The m/z and intensity arrays have different lengths.',
            location=location,
            suggestion='Ensure every peak has both an m/z and an intensity.',
            context={
                'mz_length': array_lengths['m/z array'],
                'intensity_length': array_lengths['intensity array'],
            },
        )
        return report

    finite_mz_values = []
    for peak_index, (mz, intensity) in enumerate(zip(mz_values, intensity_values)):
        peak_location = _peak_location(location, peak_index)
        try:
            mz_is_finite = math.isfinite(mz)
        except TypeError:
            report.add(
                'spectrum.mz.not_numeric',
                'The m/z value is not numeric.',
                location=peak_location,
            )
        else:
            if not mz_is_finite:
                report.add(
                    'spectrum.mz.non_finite',
                    'The m/z value is not finite.',
                    location=peak_location,
                )
            else:
                finite_mz_values.append(mz)
                if mz < 0:
                    report.add(
                        'spectrum.mz.negative',
                        'The m/z value is negative.',
                        location=peak_location,
                    )

        try:
            intensity_is_finite = math.isfinite(intensity)
        except TypeError:
            report.add(
                'spectrum.intensity.not_numeric',
                'The intensity value is not numeric.',
                location=peak_location,
            )
        else:
            if not intensity_is_finite:
                report.add(
                    'spectrum.intensity.non_finite',
                    'The intensity value is not finite.',
                    location=peak_location,
                )
            elif intensity < 0:
                report.add(
                    'spectrum.intensity.negative',
                    'The intensity value is negative.',
                    Severity.WARNING,
                    location=peak_location,
                    suggestion='Confirm whether negative intensities are intentional.',
                )

    # Ignore invalid values here so they do not hide a separate ordering issue.
    if any(left > right for left, right in zip(
            finite_mz_values, finite_mz_values[1:])):
        report.add(
            'spectrum.mz.unsorted',
            'The m/z array is not sorted in ascending order.',
            Severity.WARNING,
            location=location,
            suggestion='Sort both peak arrays together before binary-search matching.',
        )
    return report


def _spectrum_name(spectrum, index: int) -> str:
    params = spectrum.get('params', {})
    return str(params.get('title') or spectrum.get('id') or
               'spectrum[{}]'.format(index))


def validate_spectra(spectra) -> ValidationReport:
    """Validate spectra from any Pyteomics reader.

    MGF keeps names in ``params['title']`` while mzML uses ``id``. Keeping
    that small format difference here lets the actual peak checks stay shared.
    """
    report = ValidationReport()
    for index, spectrum in enumerate(spectra):
        report.extend(validate_spectrum(
            spectrum, location=_spectrum_name(spectrum, index)))
    return report


def validate_file(source, *, format: Optional[str] = None) -> ValidationReport:
    """Read and validate an MGF or mzML file.

    Reader failures are returned as diagnostics so callers can handle bad
    files and bad spectra through the same interface.
    """
    if format is None:
        _, extension = os.path.splitext(os.fspath(source))
        format = extension.lstrip('.')
    normalised_format = format.lower()

    # Imports stay local so validating one format does not load the other
    # parser or its optional dependencies.
    if normalised_format == 'mgf':
        from . import mgf as reader_module
    elif normalised_format == 'mzml':
        from . import mzml as reader_module
    else:
        raise ValueError("format must be either 'mgf' or 'mzML'")

    try:
        with reader_module.read(source) as reader:
            return validate_spectra(reader)
    except Exception as error:
        report = ValidationReport()
        report.add(
            'file.read',
            'The file could not be read: {}'.format(error),
            location=os.fspath(source),
            suggestion='Check the file format and whether the file is complete.',
        )
        return report
