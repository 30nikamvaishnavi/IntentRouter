export interface DatasetStats {
  dataset_name: string;
  source_url: string;
  num_classes: number;
  train_samples: number;
  val_samples: number;
  test_full_samples: number;
  test_subset_samples: number;
  subset_samples_per_class: number;
  data_source_used?: string;
}

export interface HardwareInfo {
  gpu_name: string;
  gpu_count: number;
  gpu_memory_gb: number;
  cpu_name: string;
  cpu_cores: number;
  ram_gb: number;
}

export interface LibraryVersions {
  python: string;
  torch: string;
  transformers: string;
  peft: string;
  datasets: string;
  scikit_learn: string;
  google_genai: string;
}

export interface BundleMetadata {
  project_name: string;
  run_date: string;
  seed: number;
  gemini_model: string;
  hardware: HardwareInfo;
  libraries: LibraryVersions;
  dataset: DatasetStats;
  author: string;
}

export interface ModelMetrics {
  id: string;
  name: string;
  category: 'baseline' | 'fine_tuned' | 'peft' | 'api' | 'quantized';
  is_baseline: boolean;
  is_local: boolean;
  accuracy_full: number;
  macro_f1_full: number;
  weighted_f1_full: number;
  top3_accuracy_full: number | null;
  accuracy_subset_308: number;
  macro_f1_subset_308: number;
  weighted_f1_subset_308: number;
  top3_accuracy_subset_308: number | null;
  latency_cpu_ms: number;
  latency_gpu_ms: number | null;
  model_size_mb: number;
  estimated_cost_per_1k: number;
  cost_type: 'compute_estimate' | 'api_pricing';
  trainable_params?: number;
  total_params?: number;
  trainable_params_percent?: number;
  training_time_sec?: number;
  gpu_vram_peak_mb?: number;
  notes: string;
}

export interface EpochMetric {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_accuracy: number;
  val_macro_f1: number;
  learning_rate: number;
}

export interface TrainingCurves {
  distilbert_full: EpochMetric[];
  distilbert_lora: EpochMetric[];
}

export interface ConfusedPair {
  rank: number;
  true_intent: string;
  predicted_intent: string;
  count: number;
  percentage_of_class_errors: number;
  explanation: string;
}

export interface MisclassifiedExample {
  id: number;
  text: string;
  true_label: string;
  predicted_label: string;
  confidence: number;
  error_type: 'subtle_boundary' | 'lexical_overlap' | 'multiple_intents' | 'entity_ambiguity';
  explanation: string;
}

export interface PerClassF1 {
  class_id: number;
  intent_name: string;
  category: string;
  test_count: number;
  precision: number;
  recall: number;
  f1_score: number;
  rank?: number;
}

export interface RobustnessTest {
  perturbation_type: string;
  description: string;
  samples_count: number;
  clean_accuracy_distilbert: number;
  perturbed_accuracy_distilbert: number;
  delta_accuracy_distilbert: number;
  clean_accuracy_gemini: number;
  perturbed_accuracy_gemini: number;
  delta_accuracy_gemini: number;
  example_original: string;
  example_perturbed: string;
}

export interface QuantizationComparison {
  base_model: string;
  quantized_model: string;
  quantization_method: string;
  accuracy_fp32: number;
  accuracy_int8: number;
  accuracy_drop: number;
  size_fp32_mb: number;
  size_int8_mb: number;
  size_reduction_ratio: number;
  latency_cpu_fp32_ms: number;
  latency_cpu_int8_ms: number;
  speedup_factor: number;
}

export interface CascadeStep {
  threshold: number;
  coverage: number;
  local_accuracy: number;
  cascade_accuracy_subset_308: number;
  local_count_subset_308: number;
  gemini_count_subset_308: number;
  estimated_cost_per_1k: number;
  average_latency_ms: number;
}

export interface ReliabilityBin {
  bin_index: number;
  bin_lower: number;
  bin_upper: number;
  mean_confidence: number;
  accuracy: number;
  sample_count: number;
}

export interface CascadeAnalysis {
  selected_threshold: number;
  expected_calibration_error: number;
  reliability_bins: ReliabilityBin[];
  threshold_steps: CascadeStep[];
  note: string;
}

export interface ResultsBundle {
  is_sample_data: boolean;
  version: string;
  metadata: BundleMetadata;
  models: ModelMetrics[];
  training_curves: TrainingCurves;
  error_analysis: {
    best_model_id: string;
    confused_pairs: ConfusedPair[];
    misclassified_examples: MisclassifiedExample[];
    per_class_f1: PerClassF1[];
  };
  robustness: {
    note: string;
    tests: RobustnessTest[];
  };
  quantization: QuantizationComparison;
  cascade: CascadeAnalysis;
}
