"""FHIR DiagnosticReport templating for generated pathology reports.

Maps generated text → HL7 FHIR DiagnosticReport JSON.
Uses fhir.resources library (Python, modern, supports R5).

To be implemented in Hafta 5 (4-6 hour task).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def generated_text_to_fhir_diagnostic_report(
    generated_text: str,
    slide_id: str,
    case_id: str | None = None,
    patient_id: str | None = None,
) -> dict:
    """Convert generated pathology text into HL7 FHIR DiagnosticReport JSON.

    Args:
        generated_text: model output (multi-sentence pathology report)
        slide_id: WSI identifier
        case_id: TCGA case ID (optional)
        patient_id: anonymized patient identifier (use slide_id-derived UUID)

    Returns:
        FHIR DiagnosticReport as dict (JSON-serializable)
    """
    from fhir.resources.diagnosticreport import DiagnosticReport
    from fhir.resources.codeableconcept import CodeableConcept
    from fhir.resources.coding import Coding
    from fhir.resources.reference import Reference

    if patient_id is None:
        # Deterministic UUID from slide_id (research-only mock patient)
        patient_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"patholens.{slide_id}"))

    report = DiagnosticReport(
        status="final",
        category=[
            CodeableConcept(
                coding=[
                    Coding(
                        system="http://terminology.hl7.org/CodeSystem/v2-0074",
                        code="PAT",
                        display="Pathology",
                    )
                ]
            )
        ],
        code=CodeableConcept(
            coding=[
                Coding(
                    system="http://loinc.org",
                    code="11526-1",
                    display="Pathology study",
                )
            ]
        ),
        subject=Reference(reference=f"Patient/{patient_id}"),
        effectiveDateTime=datetime.now(timezone.utc).isoformat(),
        issued=datetime.now(timezone.utc).isoformat(),
        conclusion=generated_text,
        identifier=[
            {
                "system": "http://patholens-vlm.itu.edu.tr/slide-id",
                "value": slide_id,
            }
        ],
    )

    return report.model_dump()


# CLI for quick testing
if __name__ == "__main__":
    import json

    sample = generated_text_to_fhir_diagnostic_report(
        generated_text="Invasive ductal carcinoma, Nottingham grade 2. Tumor measures approximately 2.5 cm. Lymphovascular invasion is identified. Margins appear negative.",
        slide_id="TCGA-AR-A1AH-01Z-00-DX1",
    )
    print(json.dumps(sample, indent=2, default=str))
