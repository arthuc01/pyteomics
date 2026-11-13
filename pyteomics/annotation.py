"""Match experimental spectrum peaks to ProForma fragment ions."""

from dataclasses import asdict, dataclass
from typing import Sequence

from . import proforma
from .validation import ValidationReport, validate_spectrum


@dataclass(frozen=True)
class TheoreticalIon:
    """A fragment ion calculated from a peptide."""

    label: str
    ion_type: str
    ordinal: int
    charge: int
    mz: float


@dataclass(frozen=True)
class PeakMatch:
    """The closest accepted experimental peak for one theoretical ion."""

    label: str
    ion_type: str
    ordinal: int
    charge: int
    theoretical_mz: float
    observed_mz: float
    intensity: float
    peak_index: int
    mass_error: float
    mass_error_unit: str

    def to_dict(self):
        return asdict(self)


@dataclass
class AnnotationResult:
    """Structured output from :func:`annotate_spectrum`."""

    peptide: str
    theoretical_ions: list[TheoreticalIon]
    matches: list[PeakMatch]
    unmatched_peak_indices: list[int]
    diagnostics: ValidationReport

    @property
    def coverage(self) -> float:
        if not self.theoretical_ions:
            return 0.0
        return len(self.matches) / len(self.theoretical_ions)

    def to_dict(self):
        return {
            'peptide': self.peptide,
            'coverage': self.coverage,
            'theoretical_ions': [asdict(ion) for ion in self.theoretical_ions],
            'matches': [match.to_dict() for match in self.matches],
            'unmatched_peak_indices': list(self.unmatched_peak_indices),
            'diagnostics': self.diagnostics.to_dict(),
        }


def _normalise_tolerance_unit(tolerance_unit: str) -> str:
    unit = tolerance_unit.lower()
    if unit == 'ppm':
        return 'ppm'
    if unit == 'da':
        return 'Da'
    raise ValueError("tolerance_unit must be either 'ppm' or 'Da'")


def _mass_error(observed_mz: float, theoretical_mz: float, unit: str) -> float:
    difference = observed_mz - theoretical_mz
    if unit == 'ppm':
        return difference / theoretical_mz * 1e6
    return difference


def _theoretical_ions(peptide, ion_types: Sequence[str], charges: Sequence[int]):
    ions = []
    for ion_type in ion_types:
        for charge in charges:
            for ordinal, mz in enumerate(
                    peptide.fragments(ion_type, charge=charge), 1):
                ions.append(TheoreticalIon(
                    label='{}{}^{}'.format(ion_type, ordinal, charge),
                    ion_type=ion_type,
                    ordinal=ordinal,
                    charge=charge,
                    mz=float(mz),
                ))
    return ions


def annotate_spectrum(
    spectrum,
    peptide,
    *,
    ion_types=('b', 'y'),
    charges=(1,),
    tolerance=20.0,
    tolerance_unit='ppm',
) -> AnnotationResult:
    """Annotate a spectrum using the nearest b/y fragment within tolerance.

    A peak and a theoretical ion are each used at most once. When several
    assignments are possible, the smallest relative tolerance error wins;
    this makes crowded spectra deterministic without hiding the raw values.
    """
    unit = _normalise_tolerance_unit(tolerance_unit)
    if tolerance < 0:
        raise ValueError('tolerance must be non-negative')

    parsed_peptide = (
        peptide if isinstance(peptide, proforma.ProForma)
        else proforma.ProForma.parse(peptide)
    )
    diagnostics = validate_spectrum(spectrum)
    theoretical_ions = _theoretical_ions(parsed_peptide, ion_types, charges)
    peak_count = len(spectrum.get('m/z array', ()))

    if not diagnostics.is_valid:
        return AnnotationResult(
            peptide=str(parsed_peptide),
            theoretical_ions=theoretical_ions,
            matches=[],
            unmatched_peak_indices=list(range(peak_count)),
            diagnostics=diagnostics,
        )

    mz_values = spectrum['m/z array']
    intensity_values = spectrum['intensity array']
    candidates = []
    for ion_index, ion in enumerate(theoretical_ions):
        for peak_index, observed_mz in enumerate(mz_values):
            error = _mass_error(observed_mz, ion.mz, unit)
            if abs(error) <= tolerance:
                # Dividing by tolerance lets Da and ppm candidates use the
                # same stable tie-breaking rule. Zero tolerance is exact-only.
                relative_error = abs(error) / tolerance if tolerance else 0.0
                candidates.append((relative_error, ion_index, peak_index, error))

    used_ions = set()
    used_peaks = set()
    accepted = []
    for _, ion_index, peak_index, error in sorted(candidates):
        if ion_index in used_ions or peak_index in used_peaks:
            continue
        used_ions.add(ion_index)
        used_peaks.add(peak_index)
        accepted.append((ion_index, peak_index, error))

    # Present matches in theoretical-ion order so reports remain easy to scan.
    matches = []
    for ion_index, peak_index, error in sorted(accepted):
        ion = theoretical_ions[ion_index]
        matches.append(PeakMatch(
            label=ion.label,
            ion_type=ion.ion_type,
            ordinal=ion.ordinal,
            charge=ion.charge,
            theoretical_mz=ion.mz,
            observed_mz=float(mz_values[peak_index]),
            intensity=float(intensity_values[peak_index]),
            peak_index=peak_index,
            mass_error=error,
            mass_error_unit=unit,
        ))

    return AnnotationResult(
        peptide=str(parsed_peptide),
        theoretical_ions=theoretical_ions,
        matches=matches,
        unmatched_peak_indices=[
            index for index in range(peak_count) if index not in used_peaks],
        diagnostics=diagnostics,
    )
