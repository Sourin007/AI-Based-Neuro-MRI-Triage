import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BadgeCheck,
  Brain,
  ChevronRight,
  FlaskConical,
  Gauge,
  Layers3,
  Moon,
  Orbit,
  ScanLine,
  ShieldCheck,
  Sun,
  UploadCloud,
  Waves,
} from "lucide-react";
import { cn } from "../lib/cn";

type TumorType = "Glioma" | "Meningioma" | "Pituitary" | "None";
type Severity = "Low" | "Medium" | "High" | "Critical";
type WorkflowStage = "intake" | "processing" | "results";
type ThemeMode = "dark" | "light";
type ImagingViewMode = "mri" | "gradcam" | "tumorMask";

type AnalysisResult = {
  tumorType: TumorType;
  confidence: number;
  volumetricEstimate: string;
  volumeCm3: number;
  suggestedPathway: string[];
  severity: Severity;
  explanation: string;
  uploadedImage: string;
  gradcamOverlay: string;
  tumorMaskOverlay: string;
  gradcamPeak: { x: number; y: number };
  tumorAreaRatio: number;
  activationIntensity: number;
  spreadScore: number;
};

type ApiReport = {
  prediction: {
    binary_label: "tumor" | "no_tumor";
    subtype_label: "pituitary" | "glioma" | "notumor" | "meningioma";
    confidence: number;
    class_probabilities: Record<string, number>;
  };
  explanation: {
    summary: string;
    reasoning: string[];
    caveats: string[];
  };
  supporting_evidence: Array<{
    source: string;
    content: string;
    relevance_score: number;
  }>;
  risk_assessment: {
    level: "low" | "medium" | "high" | "critical";
    rationale: string;
    recommended_next_steps: string[];
    urgent_flags: string[];
  };
  validation: {
    is_valid: boolean;
    needs_reprocess: boolean;
    issues: string[];
    critic_notes: string;
    attempt_count: number;
  };
  consensus: {
    summary: string;
    final_decision: "tumor" | "no_tumor" | "indeterminate";
    strategy: string;
    opinions: Array<{
      role: "radiologist" | "oncologist" | "general_physician";
      opinion: string;
      decision: "tumor" | "no_tumor" | "indeterminate";
      confidence: number;
    }>;
  };
  explainability: {
    uploaded_image: string;
    gradcam_overlay: string;
    tumor_mask_overlay: string;
    lime_overlay: string | null;
    gradcam_peak: { x: number; y: number };
    lime_available: boolean;
    notes: string[];
    tumor_area_px: number;
    tumor_area_ratio: number;
    estimated_volume_cm3: number;
    activation_intensity: number;
    spread_score: number;
    overlap_iou: number | null;
    pixel_spacing_mm: number;
    slice_thickness_mm: number;
  };
  report_text: string;
  disclaimer: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const severityStyles: Record<Severity, { label: string; bar: string; pill: string }> = {
  Low: {
    label: "Low escalation pressure",
    bar: "from-cyan-400 via-emerald-400 to-emerald-300",
    pill: "border-emerald-400/30 bg-emerald-400/10 text-emerald-200",
  },
  Medium: {
    label: "Moderate escalation pressure",
    bar: "from-cyan-400 via-indigo-400 to-amber-300",
    pill: "border-amber-400/30 bg-amber-400/10 text-amber-200",
  },
  High: {
    label: "Urgent specialist escalation",
    bar: "from-indigo-400 via-fuchsia-400 to-rose-400",
    pill: "border-rose-400/30 bg-rose-400/10 text-rose-200",
  },
  Critical: {
    label: "Immediate escalation required",
    bar: "from-rose-500 via-red-500 to-orange-400",
    pill: "border-red-400/40 bg-red-500/15 text-red-100",
  },
};

const acceptedTypes = ["DICOM", "NIfTI", "PNG", "JPG"];

function formatTumorType(subtypeLabel: ApiReport["prediction"]["subtype_label"]): TumorType {
  switch (subtypeLabel) {
    case "glioma":
      return "Glioma";
    case "meningioma":
      return "Meningioma";
    case "pituitary":
      return "Pituitary";
    default:
      return "None";
  }
}

function mapSeverity(level: ApiReport["risk_assessment"]["level"]): Severity {
  switch (level) {
    case "critical":
      return "Critical";
    case "high":
      return "High";
    case "medium":
      return "Medium";
    default:
      return "Low";
  }
}

function mapReportToUi(report: ApiReport): AnalysisResult {
  const tumorType = formatTumorType(report.prediction.subtype_label);

  return {
    tumorType,
    confidence: Math.round(report.prediction.confidence * 100),
    volumetricEstimate: `${report.explainability.estimated_volume_cm3.toFixed(2)} cm3`,
    volumeCm3: report.explainability.estimated_volume_cm3,
    suggestedPathway: report.risk_assessment.recommended_next_steps,
    severity: mapSeverity(report.risk_assessment.level),
    explanation: report.explanation.summary,
    uploadedImage: report.explainability.uploaded_image,
    gradcamOverlay: report.explainability.gradcam_overlay,
    tumorMaskOverlay: report.explainability.tumor_mask_overlay,
    gradcamPeak: report.explainability.gradcam_peak,
    tumorAreaRatio: report.explainability.tumor_area_ratio,
    activationIntensity: report.explainability.activation_intensity,
    spreadScore: report.explainability.spread_score,
  };
}

function simplifyPathwayStep(step: string, severity: Severity, tumorType: TumorType): string {
  const normalized = step.toLowerCase();
  const isLowRisk = severity === "Low";

  if (isLowRisk) {
    if (normalized.includes("oncology") || normalized.includes("cancer care") || normalized.includes("tumor board")) {
      return "Review the scan with a radiologist or relevant specialist to confirm whether this finding needs observation, follow-up imaging, or treatment.";
    }
    if (normalized.includes("neurosurgery") || normalized.includes("surgeon")) {
      return "A surgical opinion is usually only needed if a specialist confirms growth, symptoms, pressure effects, or a higher-risk appearance.";
    }
    if (normalized.includes("biopsy") || normalized.includes("histopath")) {
      return "Biopsy is not the first step for many low-risk findings; the next decision usually depends on specialist review, symptoms, and follow-up imaging.";
    }
    if (normalized.includes("follow-up") || normalized.includes("repeat")) {
      return "A follow-up MRI may be enough to monitor for any change over time, especially if symptoms are mild or stable.";
    }
    if (normalized.includes("radiology") || normalized.includes("mri")) {
      return "Have a radiologist review the full MRI study before making treatment decisions.";
    }
  }

  if (normalized.includes("radiology") || normalized.includes("mri")) {
    return "Show this scan to a radiologist for a closer review.";
  }
  if (normalized.includes("neurosurgery") || normalized.includes("surgeon")) {
    return "Talk with a brain surgeon about possible treatment options.";
  }
  if (normalized.includes("oncology") || normalized.includes("tumor board")) {
    return "Meet the cancer care team to decide the next treatment step.";
  }
  if (normalized.includes("biopsy") || normalized.includes("histopath")) {
    return "A biopsy or lab test may be needed to confirm the diagnosis.";
  }
  if (normalized.includes("follow-up") || normalized.includes("repeat")) {
    return "A follow-up scan may be needed to watch for changes.";
  }
  if (normalized.includes("urgent") || normalized.includes("emergency")) {
    return "Please get urgent medical help as soon as possible.";
  }

  return step
    .replace(/^[-*\d.\s]+/, "")
    .replace(/\bpatient\b/gi, "you")
    .replace(/\bclinical correlation\b/gi, "doctor review")
    .replace(/\bneoplasm\b/gi, "tumor")
    .replace(/\btreatment\b/gi, isLowRisk && tumorType !== "None" ? "management" : "treatment");
}

function useProcessingPipeline(active: boolean, onDone: () => void) {
  const onDoneRef = useRef(onDone);

  useEffect(() => {
    onDoneRef.current = onDone;
  }, [onDone]);

  useEffect(() => {
    if (!active) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      onDoneRef.current();
    }, 2200);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [active]);
}

function ConfidenceGauge({ value, isDark }: { value: number; isDark: boolean }) {
  const normalizedValue = Math.max(0, Math.min(100, value));
  const radius = 78;
  const strokeWidth = 12;
  const viewBoxWidth = 220;
  const viewBoxHeight = 140;
  const centerX = viewBoxWidth / 2;
  const centerY = 112;
  const startX = centerX - radius;
  const startY = centerY;
  const endX = centerX + radius;
  const endY = centerY;
  const arcPath = `M ${startX} ${startY} A ${radius} ${radius} 0 0 1 ${endX} ${endY}`;
  const arcLength = Math.PI * radius;

  const gaugeColor =
    normalizedValue >= 90 ? "#10b981" : normalizedValue >= 70 ? "#f59e0b" : "#ef4444";

  return (
    <div
      className={cn(
        "relative flex h-48 w-full items-end justify-center overflow-hidden rounded-[2rem] border p-6",
        isDark ? "border-white/10 bg-slate-950" : "border-slate-200 bg-slate-950"
      )}
    >
      <div
        className={cn(
          "absolute inset-x-8 bottom-8 top-8 rounded-full",
          isDark
            ? "border border-white/5 bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.14),_transparent_55%)]"
            : "border border-slate-800/60 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.12),_transparent_60%)]"
        )}
      />
      <div className="relative h-36 w-full max-w-[280px]">
        <svg viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`} className="h-full w-full overflow-visible">
          <path
            d={arcPath}
            fill="none"
            stroke={isDark ? "rgba(148,163,184,0.22)" : "rgba(148,163,184,0.28)"}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          <motion.path
            d={arcPath}
            fill="none"
            stroke={gaugeColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            initial={{ pathLength: 0, strokeOpacity: 0.8 }}
            animate={{
              pathLength: normalizedValue / 100,
              strokeOpacity: [0.82, 1, 0.88],
            }}
            transition={{
              pathLength: { duration: 1.1, ease: "easeOut" },
              strokeOpacity: { duration: 2.4, repeat: Infinity, ease: "easeInOut" },
            }}
            style={{
              pathLength: normalizedValue / 100,
              filter: `drop-shadow(0 0 8px ${gaugeColor}) drop-shadow(0 0 16px ${gaugeColor}66)`,
              strokeDasharray: arcLength,
            }}
          />
        </svg>
        <div className="absolute inset-x-0 bottom-2 flex justify-center">
          <div className="text-center">
            <p className="text-[10px] uppercase tracking-[0.35em] text-slate-400">Confidence</p>
            <p className="mt-1 text-3xl font-semibold text-white">{normalizedValue}%</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function HeatmapViewer({
  result,
  viewMode,
  isDark,
}: {
  result: AnalysisResult;
  viewMode: ImagingViewMode;
  isDark: boolean;
}) {
  const showingGradcam = viewMode === "gradcam";
  const showingTumorMask = viewMode === "tumorMask";
  const overlaySrc = showingTumorMask ? result.tumorMaskOverlay : showingGradcam ? result.gradcamOverlay : null;
  const peakLeft = `${12 + result.gradcamPeak.x * 76}%`;
  const peakTop = `${12 + result.gradcamPeak.y * 62}%`;

  return (
    <div className="space-y-4">
      <div
        className={cn(
          "relative min-h-[420px] overflow-hidden rounded-[2rem] border shadow-[0_30px_80px_rgba(2,6,23,0.18)]",
          isDark ? "border-white/10 bg-slate-950" : "border-slate-200 bg-white"
        )}
      >
        <div
          className={cn(
            "absolute inset-0",
            isDark
              ? "bg-[radial-gradient(circle_at_30%_35%,rgba(99,102,241,0.18),transparent_30%),radial-gradient(circle_at_72%_48%,rgba(34,211,238,0.18),transparent_22%),linear-gradient(180deg,rgba(15,23,42,0.8),rgba(2,6,23,0.96))]"
              : "bg-[radial-gradient(circle_at_30%_35%,rgba(56,189,248,0.12),transparent_30%),radial-gradient(circle_at_72%_48%,rgba(99,102,241,0.1),transparent_22%),linear-gradient(180deg,rgba(255,255,255,0.9),rgba(241,245,249,0.95))]"
          )}
        />
        <div
          className={cn(
            "absolute left-8 top-8 z-20 flex items-center gap-2 rounded-full border px-3 py-1 text-xs backdrop-blur-xl",
            isDark ? "border-white/10 bg-white/5 text-slate-300" : "border-slate-200 bg-white/90 text-slate-700"
          )}
        >
          <ScanLine className={cn("h-4 w-4", isDark ? "text-cyan-300" : "text-sky-600")} />
          Axial View | {showingTumorMask ? "Tumor Localization Mask" : showingGradcam ? "Grad-CAM Heatmap Enabled" : "Base MRI"}
        </div>
        <div
          className={cn(
            "absolute right-8 top-8 z-20 rounded-full border px-3 py-1 text-xs font-medium backdrop-blur-xl",
            isDark ? "border-cyan-400/20 bg-cyan-400/10 text-cyan-200" : "border-sky-200 bg-sky-50 text-sky-700"
          )}
        >
          {showingTumorMask ? "Tumor Region" : showingGradcam ? "Heatmap Focus" : "Base Slice"}
        </div>
        <div
          className={cn(
            "absolute inset-8 overflow-hidden rounded-[1.6rem] border",
            isDark ? "border-white/5 bg-slate-950/60" : "border-slate-200 bg-slate-100"
          )}
        >
          <img src={result.uploadedImage} alt="Uploaded MRI" className="h-full w-full object-cover opacity-95" />
          {overlaySrc ? (
            <motion.img
              key={`${result.tumorType}-${viewMode}`}
              src={overlaySrc}
              alt={showingTumorMask ? "Tumor localization overlay" : "Grad-CAM tumor heatmap"}
              className="absolute inset-0 h-full w-full object-cover"
              initial={{ opacity: 0.2 }}
              animate={{ opacity: showingTumorMask ? 0.98 : 0.92 }}
              transition={{ duration: 0.35 }}
            />
          ) : null}
          {overlaySrc ? (
            <div
              className="absolute h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-200/80 bg-cyan-300/20 shadow-[0_0_25px_rgba(34,211,238,0.55)]"
              style={{ left: peakLeft, top: peakTop }}
            />
          ) : null}
        </div>
      </div>
      <div className="mx-auto w-full max-w-[220px] space-y-3">
        <div
          className={cn(
            "rounded-xl border px-4 py-3 text-center backdrop-blur-xl",
            isDark ? "border-white/10 bg-white/5" : "border-slate-200 bg-white"
          )}
        >
          <p className="text-[10px] uppercase tracking-[0.3em] text-slate-500">Current Class</p>
          <p className={cn("mt-1 text-sm font-medium", isDark ? "text-slate-100" : "text-slate-900")}>{result.tumorType}</p>
        </div>
      </div>
    </div>
  );
}

function PredictingDots() {
  return (
    <span className="inline-flex">
      {[0, 1, 2].map((index) => (
        <motion.span
          key={index}
          className="inline-block"
          animate={{ opacity: [0.25, 1, 0.25], y: [0, -1, 0] }}
          transition={{ duration: 1, repeat: Infinity, ease: "easeInOut", delay: index * 0.16 }}
        >
          .
        </motion.span>
      ))}
    </span>
  );
}

function PredictionWaveform({ isDark }: { isDark: boolean }) {
  const stroke = isDark ? "#67e8f9" : "#0284c7";

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-[1.75rem] border p-5",
        isDark ? "border-cyan-400/15 bg-slate-950/80" : "border-sky-200 bg-sky-50/80"
      )}
    >
      <div
        className={cn(
          "absolute inset-0",
          isDark
            ? "bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.12),transparent_55%)]"
            : "bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.18),transparent_55%)]"
        )}
      />
      <div
        className={cn(
          "pointer-events-none absolute inset-0 rounded-[1.75rem]",
          isDark
            ? "shadow-[inset_0_0_0_1px_rgba(34,211,238,0.06),0_0_40px_rgba(34,211,238,0.08)]"
            : "shadow-[inset_0_0_0_1px_rgba(14,165,233,0.08),0_0_30px_rgba(14,165,233,0.08)]"
        )}
      />
      <div className="relative">
        <svg viewBox="0 0 420 110" className="h-24 w-full">
          <motion.path
            d="M0 55 C20 55 26 55 34 55 C42 55 46 30 54 30 C62 30 66 80 74 80 C82 80 86 55 96 55 C118 55 126 55 144 55 C156 55 162 16 170 16 C178 16 184 90 194 90 C204 90 210 45 220 45 C232 45 240 55 252 55 C264 55 270 40 280 40 C290 40 300 70 310 70 C320 70 330 55 342 55 C360 55 378 55 420 55"
            fill="none"
            stroke={stroke}
            strokeWidth="4"
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0.65, opacity: 0.75 }}
            animate={{
              pathLength: [0.68, 0.92, 0.86, 1],
              opacity: [0.7, 1, 0.82],
              x: [0, -4, 2, -8],
            }}
            transition={{
              duration: 2.6,
              repeat: Infinity,
              ease: "easeInOut",
              times: [0, 0.4, 0.72, 1],
            }}
            style={{ filter: `drop-shadow(0 0 10px ${stroke}88) drop-shadow(0 0 20px ${stroke}44)` }}
          />
          <motion.path
            d="M0 55 C20 55 26 55 34 55 C42 55 46 30 54 30 C62 30 66 80 74 80 C82 80 86 55 96 55 C118 55 126 55 144 55 C156 55 162 16 170 16 C178 16 184 90 194 90 C204 90 210 45 220 45 C232 45 240 55 252 55 C264 55 270 40 280 40 C290 40 300 70 310 70 C320 70 330 55 342 55 C360 55 378 55 420 55"
            fill="none"
            stroke={stroke}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeDasharray="7 10"
            initial={{ strokeDashoffset: 28 }}
            animate={{ strokeDashoffset: [28, 0, -28, -56] }}
            transition={{ duration: 2.1, repeat: Infinity, ease: "linear" }}
            opacity={0.45}
          />
        </svg>
        <div className="mt-3 text-center">
          <p className={cn("text-base font-semibold", isDark ? "text-cyan-50" : "text-sky-900")}>
            Predicting class
            <PredictingDots />
          </p>
          <p className={cn("mt-1 text-xs uppercase tracking-[0.32em]", isDark ? "text-slate-400" : "text-sky-700/70")}>
            Neural inference in progress
          </p>
        </div>
      </div>
    </div>
  );
}

function PredictionSkeleton({ isDark }: { isDark: boolean }) {
  const shimmerTone = isDark
    ? "bg-[linear-gradient(110deg,rgba(15,23,42,0.92)_8%,rgba(30,41,59,0.96)_18%,rgba(15,23,42,0.92)_33%)]"
    : "bg-[linear-gradient(110deg,rgba(226,232,240,0.9)_8%,rgba(255,255,255,0.95)_18%,rgba(226,232,240,0.9)_33%)]";

  return (
    <div className="space-y-4">
      <div
        className={cn(
          "relative h-52 overflow-hidden rounded-[1.5rem] border",
          isDark ? "border-white/8 bg-slate-900/90" : "border-slate-200 bg-slate-100",
          shimmerTone
        )}
      >
        <motion.div
          className="absolute inset-y-0 -left-1/3 w-1/3 bg-gradient-to-r from-transparent via-white/10 to-transparent"
          animate={{ x: ["0%", "420%"] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "linear" }}
        />
      </div>
      <div className="space-y-3">
        <div
          className={cn(
            "relative h-4 w-3/4 overflow-hidden rounded-full",
            isDark ? "bg-slate-800" : "bg-slate-200",
            shimmerTone
          )}
        >
          <motion.div
            className="absolute inset-y-0 -left-1/3 w-1/3 bg-gradient-to-r from-transparent via-white/12 to-transparent"
            animate={{ x: ["0%", "420%"] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "linear", delay: 0.12 }}
          />
        </div>
        <div
          className={cn(
            "relative h-4 w-1/2 overflow-hidden rounded-full",
            isDark ? "bg-slate-800" : "bg-slate-200",
            shimmerTone
          )}
        >
          <motion.div
            className="absolute inset-y-0 -left-1/3 w-1/3 bg-gradient-to-r from-transparent via-white/12 to-transparent"
            animate={{ x: ["0%", "420%"] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "linear", delay: 0.24 }}
          />
        </div>
      </div>
    </div>
  );
}

function ProcessingState({ isDark }: { isDark: boolean }) {
  return (
    <motion.div
      key="processing"
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -18 }}
      className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]"
    >
      <div className={cn("rounded-[2rem] border p-6 shadow-[0_30px_90px_rgba(2,6,23,0.18)]", isDark ? "border-white/10 bg-slate-950/80" : "border-slate-200 bg-white")}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">Neural Scan Pipeline</p>
            <h3 className={cn("mt-2 text-2xl font-semibold", isDark ? "text-white" : "text-slate-900")}>Processing Imaging Stack</h3>
          </div>
          <div className={cn("rounded-full border px-3 py-1 text-sm", isDark ? "border-indigo-400/30 bg-indigo-400/10 text-indigo-200" : "border-indigo-200 bg-indigo-50 text-indigo-700")}>
            Live Inference
          </div>
        </div>
        <div className={cn("relative mt-8 overflow-hidden rounded-[1.75rem] border p-8", isDark ? "border-white/10 bg-slate-900" : "border-slate-200 bg-slate-50")}>
          <motion.div
            className="absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-transparent via-cyan-300/20 to-transparent"
            animate={{ x: ["-10%", "450%"] }}
            transition={{ repeat: Infinity, duration: 2.2, ease: "linear" }}
          />
          <div className="grid gap-4">
            {Array.from({ length: 5 }).map((_, index) => (
              <div
                key={index}
                className={cn(
                  "h-20 rounded-2xl border",
                  isDark
                    ? "border-white/5 bg-[linear-gradient(90deg,rgba(15,23,42,0.9),rgba(30,41,59,0.9),rgba(15,23,42,0.9))]"
                    : "border-slate-200 bg-[linear-gradient(90deg,rgba(255,255,255,0.95),rgba(241,245,249,0.95),rgba(255,255,255,0.95))]"
                )}
              />
            ))}
          </div>
        </div>
      </div>
      <div
        className={cn(
          "rounded-[2rem] border p-6 backdrop-blur-2xl",
          isDark
            ? "border-cyan-400/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.95),rgba(15,23,42,0.82))] shadow-[0_0_0_1px_rgba(34,211,238,0.04),0_25px_60px_rgba(8,47,73,0.2)]"
            : "border-sky-200 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(248,250,252,0.95))] shadow-[0_20px_45px_rgba(14,165,233,0.08)]"
        )}
      >
        <div className="flex items-center gap-3">
          <Orbit className={cn("h-5 w-5", isDark ? "text-cyan-300" : "text-sky-600")} />
          <h4 className={cn("text-lg font-semibold", isDark ? "text-white" : "text-slate-900")}>Prediction Status</h4>
        </div>
        <div className="mt-6 space-y-5">
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.35, ease: "easeOut" }}
          >
            <PredictionWaveform isDark={isDark} />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 14, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.45, ease: "easeOut", delay: 0.08 }}
          >
            <PredictionSkeleton isDark={isDark} />
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
}

function SeverityMatrix({ severity, isDark }: { severity: Severity; isDark: boolean }) {
  const style = severityStyles[severity];

  return (
    <div className={cn("rounded-[1.5rem] border p-5", isDark ? "border-white/10 bg-slate-950/50" : "border-slate-200 bg-white")}>
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Risk Matrix</p>
          <p className={cn("mt-2 text-lg font-semibold", isDark ? "text-white" : "text-slate-900")}>{severity}</p>
        </div>
        <span
          className={cn(
            "rounded-full border px-3 py-1 text-xs font-medium",
            isDark
              ? style.pill
              : severity === "Low"
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : severity === "Medium"
                  ? "border-amber-200 bg-amber-50 text-amber-700"
                  : severity === "High"
                    ? "border-rose-200 bg-rose-50 text-rose-700"
                    : "border-red-200 bg-red-50 text-red-700"
          )}
        >
          {style.label}
        </span>
      </div>
      <div className={cn("mt-5 h-3 overflow-hidden rounded-full", isDark ? "bg-slate-800" : "bg-slate-200")}>
        <div className={cn("h-full w-full rounded-full bg-gradient-to-r", style.bar)} />
      </div>
      <div className={cn("mt-3 flex justify-between text-xs", isDark ? "text-slate-500" : "text-slate-600")}>
        <span>Low</span>
        <span>Medium</span>
        <span>High / Critical</span>
      </div>
    </div>
  );
}

function InsightCard({ result, isDark }: { result: AnalysisResult; isDark: boolean }) {
  const pathwayTitle =
    result.severity === "Low" ? "Suggested Clinical Next Steps" : "Suggested Treatment Pathway";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
      className={cn(
        "relative overflow-hidden rounded-[2rem] border p-6 backdrop-blur-2xl shadow-[0_20px_80px_rgba(34,211,238,0.08)]",
        isDark ? "border-white/10 bg-white/8" : "border-slate-200 bg-white/90"
      )}
    >
      <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(99,102,241,0.18),rgba(34,211,238,0.08),transparent_55%)]" />
      <div className="relative">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-cyan-300">AI Insight & Pathway</p>
            <h3 className={cn("mt-2 text-2xl font-semibold", isDark ? "text-white" : "text-slate-900")}>Clinical Intelligence Panel</h3>
          </div>
          <div className={cn("flex items-center gap-2 rounded-full border px-3 py-1 text-sm", isDark ? "border-cyan-400/20 bg-cyan-400/10 text-cyan-100" : "border-sky-200 bg-sky-50 text-sky-700")}>
            <BadgeCheck className="h-4 w-4" /> Verified by AI
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className={cn("rounded-[1.5rem] border p-5", isDark ? "border-white/10 bg-slate-950/55" : "border-slate-200 bg-slate-50/90")}>
            <div className="flex items-center gap-3">
              <FlaskConical className="h-5 w-5 text-cyan-300" />
              <p className={cn("font-medium", isDark ? "text-white" : "text-slate-900")}>{pathwayTitle}</p>
            </div>
            <ul className={cn("mt-4 space-y-3 text-sm", isDark ? "text-slate-300" : "text-slate-700")}>
              {result.suggestedPathway.map((item) => (
                <li key={item} className="flex gap-3">
                  <ChevronRight className="mt-0.5 h-4 w-4 flex-none text-indigo-300" />
                  <span>{simplifyPathwayStep(item, result.severity, result.tumorType)}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="grid gap-4">
            <div className={cn("rounded-[1.5rem] border p-5", isDark ? "border-white/10 bg-slate-950/55" : "border-slate-200 bg-slate-50/90")}>
              <div className="flex items-center gap-3">
                <Layers3 className="h-5 w-5 text-indigo-300" />
                <p className={cn("font-medium", isDark ? "text-white" : "text-slate-900")}>Tumor Volumetric Estimate</p>
              </div>
              <p className={cn("mt-4 text-3xl font-semibold", isDark ? "text-white" : "text-slate-900")}>{result.volumetricEstimate}</p>
              <p className={cn("mt-2 text-sm", isDark ? "text-slate-400" : "text-slate-600")}>Derived from the thresholded Integrated Grad-CAM lesion envelope using configurable pixel spacing assumptions.</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div className={cn("rounded-xl border px-3 py-3", isDark ? "border-white/10 bg-white/5" : "border-slate-200 bg-white")}>
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">Area Ratio</p>
                  <p className={cn("mt-2 text-sm font-medium", isDark ? "text-white" : "text-slate-900")}>{(result.tumorAreaRatio * 100).toFixed(1)}%</p>
                </div>
                <div className={cn("rounded-xl border px-3 py-3", isDark ? "border-white/10 bg-white/5" : "border-slate-200 bg-white")}>
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">Spread</p>
                  <p className={cn("mt-2 text-sm font-medium", isDark ? "text-white" : "text-slate-900")}>{result.spreadScore.toFixed(2)}</p>
                </div>
                <div className={cn("rounded-xl border px-3 py-3", isDark ? "border-white/10 bg-white/5" : "border-slate-200 bg-white")}>
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">Activation</p>
                  <p className={cn("mt-2 text-sm font-medium", isDark ? "text-white" : "text-slate-900")}>{result.activationIntensity.toFixed(2)}</p>
                </div>
              </div>
            </div>
            <SeverityMatrix severity={result.severity} isDark={isDark} />
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function ResultsWorkspace({
  result,
  viewMode,
  onChangeViewMode,
  isDark,
}: {
  result: AnalysisResult;
  viewMode: ImagingViewMode;
  onChangeViewMode: (mode: ImagingViewMode) => void;
  isDark: boolean;
}) {
  const viewOptions: Array<{ mode: ImagingViewMode; label: string }> = [
    { mode: "mri", label: "MRI" },
    { mode: "gradcam", label: "Grad-CAM" },
    { mode: "tumorMask", label: "Tumor Mask" },
  ];

  return (
    <motion.div
      key="results"
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -24 }}
      className="space-y-6"
    >
      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-4">
          <div className={cn("flex flex-wrap items-center justify-between gap-3 rounded-[1.5rem] border px-5 py-4", isDark ? "border-white/10 bg-slate-950/70" : "border-slate-200 bg-white/90")}>
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Radiology Workspace</p>
              <p className={cn("mt-1 text-lg font-semibold", isDark ? "text-white" : "text-slate-900")}>MRI Diagnostic Canvas</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {viewOptions.map(({ mode, label }) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => onChangeViewMode(mode)}
                  className={cn(
                    "inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm transition",
                    viewMode === mode
                      ? isDark
                        ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-100"
                        : "border-sky-200 bg-sky-50 text-sky-700"
                      : isDark
                        ? "border-white/10 bg-white/5 text-slate-300"
                        : "border-slate-200 bg-white text-slate-700"
                  )}
                >
                  <Waves className="h-4 w-4" />
                  {label}
                </button>
              ))}
            </div>
          </div>
          <HeatmapViewer result={result} viewMode={viewMode} isDark={isDark} />
        </div>

        <div className="space-y-4">
          <div className={cn("rounded-[2rem] border p-6 shadow-[0_20px_70px_rgba(15,23,42,0.18)]", isDark ? "border-white/10 bg-slate-950/80" : "border-slate-200 bg-white")}>
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-indigo-300">Detection Verdict</p>
                <h2 className={cn("mt-2 text-4xl font-semibold tracking-tight", isDark ? "text-white" : "text-slate-900")}>{result.tumorType}</h2>
              </div>
              <div className={cn("rounded-2xl border p-3", isDark ? "border-white/10 bg-white/5 text-slate-300" : "border-slate-200 bg-slate-50 text-slate-700")}>
                <Gauge className={cn("h-6 w-6", isDark ? "text-cyan-300" : "text-sky-600")} />
              </div>
            </div>
            <p className={cn("mt-4 text-sm leading-7", isDark ? "text-slate-300" : "text-slate-700")}>{result.explanation}</p>
            <div className="mt-6">
              <ConfidenceGauge value={result.confidence} isDark={isDark} />
            </div>
          </div>
        </div>
      </div>

      <InsightCard result={result} isDark={isDark} />
    </motion.div>
  );
}

export default function NeuroAiDiagnosticSuite() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [stage, setStage] = useState<WorkflowStage>("intake");
  const [isDragging, setIsDragging] = useState(false);
  const [viewMode, setViewMode] = useState<ImagingViewMode>("gradcam");
  const [fileName, setFileName] = useState<string>("No scan selected");
  const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null);
  const [processingComplete, setProcessingComplete] = useState(false);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const isDark = theme === "dark";

  const handleProcessingComplete = useCallback(() => {
    setProcessingComplete(true);
  }, []);

  useProcessingPipeline(stage === "processing", handleProcessingComplete);

  useEffect(() => {
    if (stage === "processing" && processingComplete && analysisReady && currentResult) {
      setStage("results");
    }
  }, [analysisReady, currentResult, processingComplete, stage]);

  async function beginAnalysis(file: File) {
    setFileName(file.name);
    setCurrentResult(null);
    setErrorMessage(null);
    setAnalysisReady(false);
    setProcessingComplete(false);
    setViewMode("gradcam");
    setStage("processing");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis request failed with status ${response.status}`);
      }

      const report = (await response.json()) as ApiReport;
      setCurrentResult(mapReportToUi(report));
      setAnalysisReady(true);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to reach the diagnostic API. Start the backend and try again.";
      setErrorMessage(`${message}. Make sure FastAPI is running on ${API_BASE_URL}.`);
      setStage("intake");
    }
  }

  function handleFile(file: File | null) {
    if (!file) return;
    void beginAnalysis(file);
  }

  return (
    <div className={cn("min-h-screen transition-colors duration-300", isDark ? "bg-slate-950 text-slate-100" : "bg-slate-100 text-slate-900")}>
      <div className="relative overflow-hidden">
        <div
          className={cn(
            "absolute inset-0",
            isDark
              ? "bg-[radial-gradient(circle_at_top_left,rgba(99,102,241,0.18),transparent_28%),radial-gradient(circle_at_top_right,rgba(34,211,238,0.16),transparent_22%),linear-gradient(180deg,#020617_0%,#020617_40%,#0f172a_100%)]"
              : "bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.18),transparent_28%),radial-gradient(circle_at_top_right,rgba(99,102,241,0.12),transparent_20%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_45%,#e0f2fe_100%)]"
          )}
        />
        <div className={cn("absolute inset-0 [background-image:linear-gradient(rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px)] [background-size:88px_88px]", isDark ? "opacity-30" : "opacity-50")} />

        <main className="relative mx-auto max-w-7xl px-6 py-10 lg:px-10 lg:py-14">
          <header className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className={cn("inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm backdrop-blur-xl", isDark ? "border-indigo-400/20 bg-indigo-400/10 text-indigo-100" : "border-sky-300 bg-white/80 text-sky-900")}>
                <Brain className="h-4 w-4 text-cyan-300" />
                Neuro-AI Diagnostic Suite
              </div>
              <h1 className={cn("mt-6 font-sans text-4xl font-semibold tracking-tight sm:text-5xl lg:text-6xl", isDark ? "text-white" : "text-slate-900")}>
                Precision MRI triage with explainable clinical intelligence.
              </h1>
              <p className={cn("mt-5 max-w-2xl text-base leading-8 sm:text-lg", isDark ? "text-slate-300" : "text-slate-700")}>
                Upload a brain MRI, surface tumor classification instantly, and route clinicians into an AI-assisted pathway for interpretation and next-step guidance.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => setTheme((value) => (value === "dark" ? "light" : "dark"))}
                className={cn("rounded-2xl border px-4 py-3 text-sm backdrop-blur-xl transition", isDark ? "border-white/10 bg-white/5 text-slate-200" : "border-slate-300 bg-white/80 text-slate-800")}
              >
                <div className="flex items-center gap-2">
                  {isDark ? <Sun className="h-4 w-4 text-amber-300" /> : <Moon className="h-4 w-4 text-indigo-500" />}
                  {isDark ? "Light Mode" : "Dark Mode"}
                </div>
              </button>
              {[
                { label: "Verified by AI", Icon: BadgeCheck },
                { label: "Theme Toggle", Icon: ShieldCheck },
                { label: "Heatmap Viewer", Icon: ScanLine },
              ].map(({ label, Icon }) => (
                <div key={label} className={cn("rounded-2xl border px-4 py-3 text-sm backdrop-blur-xl", isDark ? "border-white/10 bg-white/5 text-slate-200" : "border-slate-300 bg-white/80 text-slate-800")}>
                  <div className="flex items-center gap-2">
                    <Icon className="h-4 w-4 text-cyan-300" />
                    {label}
                  </div>
                </div>
              ))}
            </div>
          </header>

          <section className="mt-10 grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            <AnimatePresence mode="wait">
              {stage === "intake" ? (
                <motion.div
                  key="intake"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={cn("rounded-[2rem] border p-6 shadow-[0_30px_80px_rgba(2,6,23,0.25)]", isDark ? "border-white/10 bg-slate-950/75" : "border-slate-300 bg-white/80")}
                >
                  <div
                    onDragOver={(event) => {
                      event.preventDefault();
                      setIsDragging(true);
                    }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={(event) => {
                      event.preventDefault();
                      setIsDragging(false);
                      handleFile(event.dataTransfer.files?.[0] ?? null);
                    }}
                    onClick={() => inputRef.current?.click()}
                    className={cn(
                      "group relative flex min-h-[420px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[1.8rem] border border-dashed p-10 text-center transition-all duration-300",
                      isDragging
                        ? "border-cyan-300/70 bg-cyan-400/10 shadow-[0_0_80px_rgba(34,211,238,0.18)]"
                        : isDark
                          ? "border-white/15 bg-[linear-gradient(180deg,rgba(15,23,42,0.82),rgba(15,23,42,0.5))] hover:border-indigo-400/50 hover:bg-indigo-400/5 hover:shadow-[0_0_90px_rgba(99,102,241,0.2)]"
                          : "border-slate-300 bg-[linear-gradient(180deg,rgba(255,255,255,0.95),rgba(241,245,249,0.85))] hover:border-sky-400 hover:bg-sky-50"
                    )}
                  >
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(34,211,238,0.12),transparent_42%)] opacity-0 transition duration-300 group-hover:opacity-100" />
                    <motion.div
                      animate={{ y: [0, -8, 0] }}
                      transition={{ repeat: Infinity, duration: 2.6, ease: "easeInOut" }}
                      className="relative rounded-full border border-cyan-400/30 bg-cyan-400/10 p-5"
                    >
                      <UploadCloud className="h-10 w-10 text-cyan-300" />
                    </motion.div>
                    <h2 className={cn("relative mt-6 text-3xl font-semibold", isDark ? "text-white" : "text-slate-900")}>Drop MRI Study Into Intake Hub</h2>
                    <p className={cn("relative mt-4 max-w-xl text-base leading-7", isDark ? "text-slate-300" : "text-slate-700")}>
                      Drag and drop DICOM or NIfTI files to start prediction, or click to choose a scan manually.
                    </p>
                    <div className={cn("relative mt-6 flex flex-wrap items-center justify-center gap-3 text-xs uppercase tracking-[0.28em]", isDark ? "text-slate-400" : "text-slate-600")}>
                      {acceptedTypes.map((type) => (
                        <span key={type} className={cn("rounded-full border px-3 py-2", isDark ? "border-white/10 bg-white/5" : "border-slate-300 bg-white/90")}>
                          {type}
                        </span>
                      ))}
                    </div>
                    <div className={cn("relative mt-8 rounded-full border px-5 py-3 text-sm", isDark ? "border-indigo-400/20 bg-indigo-400/10 text-indigo-100" : "border-sky-300 bg-sky-100 text-sky-900")}>
                      Minimal friction. High-confidence triage.
                    </div>
                    <input
                      ref={inputRef}
                      type="file"
                      className="hidden"
                      accept=".dcm,.nii,.nii.gz,image/*"
                      onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
                    />
                  </div>
                </motion.div>
              ) : stage === "processing" ? (
                <div className="xl:col-span-2">
                  <ProcessingState isDark={isDark} />
                </div>
              ) : currentResult ? (
                <div className="xl:col-span-2">
                  <ResultsWorkspace
                    result={currentResult}
                    viewMode={viewMode}
                    onChangeViewMode={setViewMode}
                    isDark={isDark}
                  />
                </div>
              ) : null}
            </AnimatePresence>
          </section>

          {errorMessage ? (
            <section className="mt-6">
              <div className={cn("rounded-[1.5rem] border px-5 py-4 text-sm", isDark ? "border-rose-400/20 bg-rose-400/10 text-rose-100" : "border-rose-300 bg-rose-50 text-rose-700")}>
                {errorMessage}
              </div>
            </section>
          ) : null}

          <section className="mt-6">
            <div className={cn("rounded-[2rem] border p-6 backdrop-blur-2xl", isDark ? "border-white/10 bg-white/5" : "border-slate-300 bg-white/80")}>
              <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Intake Session</p>
              <div className="mt-5 grid gap-4 md:grid-cols-3">
                {[
                  ["Current file", fileName],
                  ["Inference state", stage === "intake" ? "Awaiting Upload" : stage === "processing" ? "Neural Processing" : "Report Ready"],
                  ["Modality support", "MRI | DICOM | NIfTI"],
                ].map(([label, value]) => (
                  <div key={label} className={cn("rounded-2xl border p-4", isDark ? "border-white/10 bg-slate-950/55" : "border-slate-300 bg-slate-50")}>
                    <p className="text-[10px] uppercase tracking-[0.3em] text-slate-500">{label}</p>
                    <p className={cn("mt-2 text-sm font-medium", isDark ? "text-white" : "text-slate-900")}>{value}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
