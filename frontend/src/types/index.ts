/**
 * Comprehensive TypeScript Interfaces for the Fake Review Detection Platform.
 * Accurately aligns with FastAPI Pydantic models in src/api/main.py.
 */

export type RiskTier = 'LOW_RISK' | 'MODERATE_RISK' | 'HIGH_RISK';

export type Language = 'en' | 'hi' | 'hinglish';

export type AppMode = 'research' | 'demo';

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

export interface DatasetStatistics {
  total_products: number;
  fake_products_count: number;
  genuine_products_count: number;
  fake_prevalence_pct: number;
  features_count: number;
}

export interface ModelStatistics {
  registered_models_count: number;
  best_validated_model: string;
  best_fake_f1: number;
  best_roc_auc: number;
}

export interface FeatureGroupInfo {
  name: string;
  feature_count: number;
  key: string;
}

export interface FeatureProgressionItem {
  group: string;
  f1: number;
  auc: number;
  features: number;
  protocol?: string;
}

export interface TopModelItem {
  model: string;
  f1: number;
  auc: number;
  type: string;
  accuracy?: number | null;
}

export interface DashboardOverview {
  status: string;
  dataset_statistics: DatasetStatistics;
  model_statistics: ModelStatistics;
  feature_groups: FeatureGroupInfo[];
  feature_group_progression?: FeatureProgressionItem[];
  top_models?: TopModelItem[];
  protocol_summary?: Record<string, number>;
  clustering_summary?: {
    optimal_k: number;
    optimal_silhouette: number;
  };
  shap_summary?: {
    network_pct: number;
  };
  recent_experiments: ExperimentRecord[];
  system_status: {
    api_version: string;
    environment: string;
    models_loaded: number;
    timestamp: string;
  };
}

export interface ModelRegistryItem {
  model_name: string;
  filename: string;
  feature_group: string;
  required_features_count: number;
  required_features: string[];
  description: string;
}

export interface PredictionPayload {
  product_ID?: number;
  displayed_rating: number;
  model_name: string;
  features: Record<string, number>;
}

export interface PredictionResponse {
  product_ID?: number;
  model_name: string;
  model_version: string;
  p_fake: number;
  probability_type: 'calibrated' | 'raw';
  calibration_method: string;
  is_fake_prediction: boolean;
  risk_score: number;
  risk_tier: RiskTier;
  trust_adjusted_rating: number;
  explanation_english: string;
  explanation_hindi: string;
  explanation_hinglish: string;
}

export interface ProductCatalogItem {
  product_ID: number;
  displayed_rating: number;
  risk_score?: number;
  risk_tier?: RiskTier;
  trust_adjusted_rating?: number;
  features: Record<string, number>;
}

export interface ProductListResponse {
  total: number;
  page: number;
  page_size: number;
  items: Array<Record<string, any>>;
}

export interface ExperimentRecord {
  experiment_id: string;
  source_script?: string;
  algorithm: string;
  feature_group: string;
  dataset_fingerprint?: string;
  split_protocol?: string;
  evaluation_type: string;
  random_seed?: number;
  parameters?: string;
  results_path?: string;
  artifact_path?: string;
  accuracy?: number;
  roc_auc?: number;
  fake_f1?: number;
  pr_auc?: number;
  brier_score?: number;
  execution_status?: string;
  verification_status?: string;
}

export interface ExperimentListResponse {
  total: number;
  page: number;
  page_size: number;
  records: ExperimentRecord[];
}

export interface ModelMetricRecord {
  experiment_id: string;
  model: string;
  feature_group: string;
  accuracy: number;
  fake_precision: number;
  fake_recall: number;
  fake_f1: number;
  macro_f1?: number;
  weighted_f1?: number;
  roc_auc: number;
  pr_auc: number;
  brier_score?: number;
  specificity?: number;
  sensitivity?: number;
  evaluation_protocol?: string;
  evaluation_type?: string;
  false_positive_rate?: number;
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  confusion_matrix?: string;
  expected_calibration_error?: number;
}

export interface ShapFeatureItem {
  feature: string;
  mean_abs_shap: number;
  percent_importance: number;
  modality: 'network' | 'review' | 'image' | 'other';
}

export interface GlobalShapResponse {
  features: ShapFeatureItem[];
  modality_breakdown: Record<string, number>;
  total_features: number;
  top_feature: string;
}

export interface ClusteringSweepMetric {
  k: number;
  inertia: number;
  silhouette: number;
  davies_bouldin: number;
  calinski_harabasz: number;
}

export interface ClusterProfile {
  cluster_ID: number;
  n_products: number;
  share_of_total: number;
  ground_truth_fake_prevalence: number;
  ground_truth_fake_count: number;
  mean_p_fake_predicted: number;
  pct_predicted_fake_threshold_05: number;
  pct_predicted_fake_threshold_07: number;
  mean_avg_review_rating?: number;
  mean_n_of_reviews?: number;
  mean_clustering_coef?: number;
  mean_w_degree?: number;
  ground_truth_genuine_prevalence?: number;
}

export interface ClusteringDetailsResponse {
  sweep_metrics: ClusteringSweepMetric[];
  k2_profiles: ClusterProfile[];
  k20_profiles: ClusterProfile[];
  optimal_k: number | null;
  optimal_silhouette: number | null;
  clustering_summary?: ClusteringSummaryResponse;
}

export interface ClusteringSummaryResponse {
  available_k_values: number[];
  selected_k: number | null;
  selection_criterion: string;
  silhouette_score: number | null;
  davies_bouldin_score: number | null;
  calinski_harabasz_score: number | null;
  cluster_count: number | null;
  dataset_identifier: string;
  result_source: string;
  optimal_k?: number | null;
  optimal_silhouette?: number | null;
  cluster_profiles?: ClusterProfile[];
}

export interface SystemEnvironmentResponse {
  python_version: string;
  full_python_version: string;
  sklearn_version: string;
  xgboost_version: string;
  catboost_version: string;
  fastapi_version: string;
  platform: string;
  docker_supported_python: string;
  host_runtime_status: string;
}

export interface SystemTestReportResponse {
  status: 'available' | 'not_available' | 'error' | 'passed' | 'failed';
  message?: string;
  total_tests: number | null;
  passed: number | null;
  failed: number | null;
  skipped: number | null;
  duration_seconds?: number;
  test_execution_date: string | null;
  environment: string | null;
  test_command: string | null;
  tests?: Array<{ nodeid: string; outcome: string; duration: number }>;
}

export interface DatasetsSummaryResponse {
  labeled_products: number;
  features_count: number;
  positive_class_count: number;
  negative_class_count: number;
  fake_prevalence_pct: number;
  missing_values_count?: number;
  missing_values_status?: string;
  development_split_count?: number;
  holdout_split_count?: number;
  available_feature_groups: string[];
}


export interface NetworkSummaryResponse {
  benchmarks: Array<Record<string, any>>;
  ring_evaluations: Array<Record<string, any>>;
  node_features: string[];
  real_network_available: boolean;
  real_network_note: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: 'reviewer' | 'product';
  is_suspicious: boolean;
  degree: number;
  position: { x: number; y: number };
  pagerank?: number;
  clustering_coef?: number | null;
  coordinated_edge_ratio?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  is_fake: boolean;
  weight: number;
}

export interface NetworkGraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    reviewer_nodes: number;
    product_nodes: number;
    ring_reviewers: number;
    target_products: number;
  };
  mode: string;
}
