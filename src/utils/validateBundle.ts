import { ResultsBundle } from '../types/bundle';

export interface ValidationResult {
  valid: boolean;
  error?: string;
  bundle?: ResultsBundle;
}

function check(condition: boolean, path: string, expectedType: string): void {
  if (!condition) {
    throw new Error(`Schema validation error at '${path}': expected ${expectedType}`);
  }
}

export function validateResultsBundle(data: unknown): ValidationResult {
  try {
    if (!data || typeof data !== 'object') {
      return { valid: false, error: "Root bundle must be an object." };
    }

    const b = data as Record<string, unknown>;

    check(typeof b.is_sample_data === 'boolean', 'is_sample_data', 'boolean');
    check(typeof b.version === 'string', 'version', 'string');

    // Metadata
    check(!!b.metadata && typeof b.metadata === 'object', 'metadata', 'object');
    const meta = b.metadata as Record<string, unknown>;
    check(typeof meta.project_name === 'string', 'metadata.project_name', 'string');
    check(typeof meta.run_date === 'string', 'metadata.run_date', 'string');
    check(typeof meta.seed === 'number', 'metadata.seed', 'number');
    check(typeof meta.gemini_model === 'string', 'metadata.gemini_model', 'string');

    // Hardware
    check(!!meta.hardware && typeof meta.hardware === 'object', 'metadata.hardware', 'object');
    const hw = meta.hardware as Record<string, unknown>;
    check(typeof hw.gpu_name === 'string', 'metadata.hardware.gpu_name', 'string');
    check(typeof hw.cpu_name === 'string', 'metadata.hardware.cpu_name', 'string');

    // Libraries
    check(!!meta.libraries && typeof meta.libraries === 'object', 'metadata.libraries', 'object');
    const lib = meta.libraries as Record<string, unknown>;
    check(typeof lib.python === 'string', 'metadata.libraries.python', 'string');
    check(typeof lib.torch === 'string', 'metadata.libraries.torch', 'string');
    check(typeof lib.transformers === 'string', 'metadata.libraries.transformers', 'string');

    // Dataset
    check(!!meta.dataset && typeof meta.dataset === 'object', 'metadata.dataset', 'object');
    const ds = meta.dataset as Record<string, unknown>;
    check(typeof ds.dataset_name === 'string', 'metadata.dataset.dataset_name', 'string');
    check(typeof ds.num_classes === 'number' && ds.num_classes === 77, 'metadata.dataset.num_classes', '77');
    check(typeof ds.train_samples === 'number', 'metadata.dataset.train_samples', 'number');
    check(typeof ds.val_samples === 'number', 'metadata.dataset.val_samples', 'number');
    check(typeof ds.test_full_samples === 'number', 'metadata.dataset.test_full_samples', 'number');
    check(typeof ds.test_subset_samples === 'number', 'metadata.dataset.test_subset_samples', 'number');

    // Models
    check(Array.isArray(b.models) && b.models.length >= 4, 'models', 'array with at least 4 models');
    const models = b.models as Record<string, unknown>[];
    models.forEach((m, idx) => {
      const p = `models[${idx}]`;
      check(typeof m.id === 'string', `${p}.id`, 'string');
      check(typeof m.name === 'string', `${p}.name`, 'string');
      check(typeof m.category === 'string', `${p}.category`, 'string');
      check(typeof m.is_baseline === 'boolean', `${p}.is_baseline`, 'boolean');
      check(typeof m.is_local === 'boolean', `${p}.is_local`, 'boolean');
      check(typeof m.accuracy_full === 'number', `${p}.accuracy_full`, 'number');
      check(typeof m.macro_f1_full === 'number', `${p}.macro_f1_full`, 'number');
      check(typeof m.weighted_f1_full === 'number', `${p}.weighted_f1_full`, 'number');
      check(typeof m.accuracy_subset_308 === 'number', `${p}.accuracy_subset_308`, 'number');
      check(typeof m.macro_f1_subset_308 === 'number', `${p}.macro_f1_subset_308`, 'number');
      check(typeof m.weighted_f1_subset_308 === 'number', `${p}.weighted_f1_subset_308`, 'number');
      check(typeof m.latency_cpu_ms === 'number', `${p}.latency_cpu_ms`, 'number');
      check(typeof m.model_size_mb === 'number', `${p}.model_size_mb`, 'number');
      check(typeof m.estimated_cost_per_1k === 'number', `${p}.estimated_cost_per_1k`, 'number');
      check(typeof m.cost_type === 'string', `${p}.cost_type`, 'compute_estimate | api_pricing');
      check(typeof m.notes === 'string', `${p}.notes`, 'string');
    });

    // Training curves
    check(!!b.training_curves && typeof b.training_curves === 'object', 'training_curves', 'object');
    const curves = b.training_curves as Record<string, unknown>;
    check(Array.isArray(curves.distilbert_full) && curves.distilbert_full.length >= 3, 'training_curves.distilbert_full', 'array of >= 3 epochs');
    check(Array.isArray(curves.distilbert_lora) && curves.distilbert_lora.length >= 3, 'training_curves.distilbert_lora', 'array of >= 3 epochs');

    // Error analysis
    check(!!b.error_analysis && typeof b.error_analysis === 'object', 'error_analysis', 'object');
    const ea = b.error_analysis as Record<string, unknown>;
    check(typeof ea.best_model_id === 'string', 'error_analysis.best_model_id', 'string');
    check(Array.isArray(ea.confused_pairs) && ea.confused_pairs.length >= 15, 'error_analysis.confused_pairs', 'array with at least 15 pairs');
    (ea.confused_pairs as Record<string, unknown>[]).slice(0, 15).forEach((cp, idx) => {
      const p = `error_analysis.confused_pairs[${idx}]`;
      check(typeof cp.true_intent === 'string', `${p}.true_intent`, 'string');
      check(typeof cp.predicted_intent === 'string', `${p}.predicted_intent`, 'string');
      check(typeof cp.count === 'number', `${p}.count`, 'number');
    });

    check(Array.isArray(ea.misclassified_examples) && ea.misclassified_examples.length >= 20, 'error_analysis.misclassified_examples', 'array with at least 20 examples');
    (ea.misclassified_examples as Record<string, unknown>[]).slice(0, 20).forEach((me, idx) => {
      const p = `error_analysis.misclassified_examples[${idx}]`;
      check(typeof me.text === 'string', `${p}.text`, 'string');
      check(typeof me.true_label === 'string', `${p}.true_label`, 'string');
      check(typeof me.predicted_label === 'string', `${p}.predicted_label`, 'string');
    });

    check(Array.isArray(ea.per_class_f1) && ea.per_class_f1.length === 77, 'error_analysis.per_class_f1', 'array of exactly 77 banking intent classes');

    // Robustness
    check(!!b.robustness && typeof b.robustness === 'object', 'robustness', 'object');
    const rob = b.robustness as Record<string, unknown>;
    check(typeof rob.note === 'string', 'robustness.note', 'string');
    check(Array.isArray(rob.tests) && rob.tests.length >= 3, 'robustness.tests', 'array of at least 3 perturbation tests');

    // Quantization
    check(!!b.quantization && typeof b.quantization === 'object', 'quantization', 'object');
    const q = b.quantization as Record<string, unknown>;
    check(typeof q.accuracy_fp32 === 'number', 'quantization.accuracy_fp32', 'number');
    check(typeof q.accuracy_int8 === 'number', 'quantization.accuracy_int8', 'number');
    check(typeof q.size_fp32_mb === 'number', 'quantization.size_fp32_mb', 'number');
    check(typeof q.size_int8_mb === 'number', 'quantization.size_int8_mb', 'number');
    check(typeof q.speedup_factor === 'number', 'quantization.speedup_factor', 'number');

    // Cascade Analysis
    check(!!b.cascade && typeof b.cascade === 'object', 'cascade', 'object');
    const cas = b.cascade as Record<string, unknown>;
    check(typeof cas.selected_threshold === 'number', 'cascade.selected_threshold', 'number');
    check(typeof cas.expected_calibration_error === 'number', 'cascade.expected_calibration_error', 'number');
    check(Array.isArray(cas.threshold_steps) && cas.threshold_steps.length >= 10, 'cascade.threshold_steps', 'array with at least 10 threshold steps');
    check(Array.isArray(cas.reliability_bins) && cas.reliability_bins.length >= 5, 'cascade.reliability_bins', 'array with reliability bins');

    return {
      valid: true,
      bundle: data as ResultsBundle,
    };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      valid: false,
      error: message,
    };
  }
}
