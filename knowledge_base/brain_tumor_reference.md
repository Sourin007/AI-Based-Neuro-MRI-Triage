# Brain Tumor Triage Reference

This local corpus is a compact medical grounding set for the multi-agent workflow. It is not a replacement for formal clinical guidelines, but it helps the system produce evidence-linked explanations instead of unsupported free text.

## MRI Interpretation Notes

Brain MRI interpretation should consider lesion location, enhancement pattern, edema, mass effect, diffusion, perfusion, and comparison with prior studies. AI classifier confidence reflects model certainty on training-like inputs, not a definitive diagnosis.

Glioma assessments usually require correlation with tumor location, infiltrative pattern, contrast enhancement, and the patient symptom profile. Histopathology remains the reference standard for definitive subtype and grade determination.

Meningioma assessments often consider extra-axial appearance, dural attachment, enhancement behavior, and associated mass effect. Some lesions can mimic meningioma on limited image review, so full-sequence radiology interpretation is important.

Pituitary-region lesions should be correlated with endocrine symptoms, visual symptoms, lesion size, and dedicated sellar imaging when appropriate. Clinical urgency can rise when there are visual field changes, pituitary apoplexy concerns, or significant mass effect.

No-tumor predictions do not rule out subtle lesions, inflammatory changes, vascular abnormalities, or image quality limitations. If symptoms remain concerning, standard neuroradiology review and follow-up imaging may still be warranted.

## Safety and Escalation Guidance

High-confidence tumor-positive AI output should trigger specialist review rather than direct autonomous diagnosis. Recommended next steps may include contrast-enhanced MRI review, comparison with prior scans, multidisciplinary discussion, and biopsy planning when clinically indicated.

Low-confidence or conflicting outputs should be treated as indeterminate. The workflow should request more evidence, retrieve more context, and emphasize uncertainty in the final report.

All AI-generated reports must include a clinician-facing disclaimer that the system is advisory, explainable, and intended to support rather than replace radiology or oncology expertise.
