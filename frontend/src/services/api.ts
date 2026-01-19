import axios from 'axios'
import { useAuthStore } from '../stores/authStore'
import type {
  Business,
  BusinessStatus,
  Competitor,
  CompetitorAnalysis,
  CompetitorDiscoveryRequest,
  CompetitorPreviewResponse,
  ContentCalendar,
  DiscoveredCompetitor,
  ContentPiece,
  EngagementPrediction,
  EngagementWeights,
  KPIWeightsConfig,
  KPITemplate,
  KPIWeightsPreview,
  MultiOutputPredictionResponse,
  OnboardingResponse,
  ScrapedPost,
  User,
  ViralOpportunity,
} from '../types'

const API_BASE_URL = '/api/v1'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout()
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ============ AUTH ============

export const authApi = {
  register: async (email: string, password: string, fullName?: string) => {
    const { data } = await api.post<User>('/auth/register', {
      email,
      password,
      full_name: fullName,
    })
    return data
  },

  login: async (email: string, password: string) => {
    const { data } = await api.post<{ access_token: string; token_type: string }>(
      '/auth/login',
      { email, password }
    )
    return data
  },
}

// ============ BUSINESS ============

export const businessApi = {
  onboard: async (data: {
    name: string
    business_type: string
    description?: string
    location?: string
    instagram_handle?: string
    tiktok_handle?: string
    linkedin_handle?: string
    active_platforms: string[]
    competitors: { platform: string; handle: string }[]
    content_goals?: string[]
    posting_frequency?: string
    brand_voice?: string
  }) => {
    const { data: response } = await api.post<OnboardingResponse>('/business/onboard', data)
    return response
  },

  getMyBusinesses: async () => {
    const { data } = await api.get<Business[]>('/business/me')
    return data
  },

  getBusiness: async (businessId: number) => {
    const { data } = await api.get<Business>(`/business/${businessId}`)
    return data
  },

  updateBusiness: async (businessId: number, updates: Partial<Business>) => {
    const { data } = await api.put<Business>(`/business/${businessId}`, updates)
    return data
  },

  getStatus: async (businessId: number) => {
    const { data } = await api.get<BusinessStatus>(`/business/${businessId}/status`)
    return data
  },

  // Human-in-the-Loop: Own Profile Configuration
  getOwnProfileConfig: async (businessId: number) => {
    const { data } = await api.get<{
      own_instagram_username: string | null
      own_tiktok_username: string | null
      feedback_loop_enabled: boolean
      message: string
    }>(`/business/${businessId}/own-profile-config`)
    return data
  },

  updateOwnProfileConfig: async (
    businessId: number,
    config: {
      own_instagram_username?: string | null
      own_tiktok_username?: string | null
    }
  ) => {
    const { data } = await api.put<{
      own_instagram_username: string | null
      own_tiktok_username: string | null
      feedback_loop_enabled: boolean
      message: string
    }>(`/business/${businessId}/own-profile-config`, config)
    return data
  },
}

// ============ COMPETITORS ============

export const competitorsApi = {
  getCompetitors: async (businessId: number) => {
    const { data } = await api.get<Competitor[]>(`/competitors/${businessId}`)
    return data
  },

  addCompetitor: async (businessId: number, platform: string, handle: string) => {
    const { data } = await api.post<Competitor>(`/competitors/${businessId}/add`, {
      platform,
      handle,
    })
    return data
  },

  discoverCompetitors: async (request: CompetitorDiscoveryRequest) => {
    const { data } = await api.post<DiscoveredCompetitor[]>('/competitors/discover', request)
    return data
  },

  previewCompetitor: async (platform: string, handle: string) => {
    const { data } = await api.post<CompetitorPreviewResponse>('/competitors/preview', {
      platform,
      handle,
    })
    return data
  },

  getAnalysis: async (businessId: number, competitorId: number) => {
    const { data } = await api.get<CompetitorAnalysis>(
      `/competitors/${businessId}/${competitorId}/analysis`
    )
    return data
  },

  getPosts: async (businessId: number, competitorId: number, limit = 20) => {
    const { data } = await api.get<ScrapedPost[]>(
      `/competitors/${businessId}/${competitorId}/posts`,
      { params: { limit } }
    )
    return data
  },

  deleteCompetitor: async (businessId: number, competitorId: number) => {
    await api.delete(`/competitors/${businessId}/${competitorId}`)
  },
}

// ============ CONTENT ============

export const contentApi = {
  generateCalendar: async (
    businessId: number,
    data: {
      month: number
      year: number
      posts_count?: number
      primary_goal?: string
      platforms?: string[]
      content_mix?: Record<string, number>
      refresh_competitor_data?: boolean
    }
  ) => {
    const { data: response } = await api.post<ContentCalendar>(
      `/content/${businessId}/generate-calendar`,
      data
    )
    return response
  },

  getCalendars: async (businessId: number) => {
    const { data } = await api.get<ContentCalendar[]>(`/content/${businessId}/calendars`)
    return data
  },

  generateSingle: async (
    businessId: number,
    platform: string,
    contentFormat: string,
    goal?: string,
    topic?: string
  ) => {
    const { data } = await api.post<{ content: ContentPiece; variations: ContentPiece[] }>(
      `/content/${businessId}/generate-single`,
      null,
      {
        params: {
          platform,
          content_format: contentFormat,
          goal,
          topic,
        },
      }
    )
    return data
  },

  generateVariations: async (businessId: number, contentId: number, numVariations = 2) => {
    const { data } = await api.post<{ original: ContentPiece; variations: ContentPiece[] }>(
      `/content/${businessId}/content/${contentId}/variations`,
      null,
      { params: { num_variations: numVariations } }
    )
    return data
  },

  getContent: async (businessId: number, contentId: number) => {
    const { data } = await api.get<ContentPiece>(
      `/content/${businessId}/content/${contentId}`
    )
    return data
  },

  updateContent: async (
    businessId: number,
    contentId: number,
    updates: {
      caption?: string
      hook_text?: string
      hashtags?: string[]
      status?: string
      scheduled_date?: string
    }
  ) => {
    const { data } = await api.put<ContentPiece>(
      `/content/${businessId}/content/${contentId}`,
      null,
      { params: updates }
    )
    return data
  },

  getEngagementPrediction: async (businessId: number, contentId: number) => {
    const { data } = await api.get<EngagementPrediction>(
      `/content/${businessId}/content/${contentId}/engagement`
    )
    return data
  },

  exportContent: async (
    businessId: number,
    format: 'csv' | 'json' = 'csv',
    options?: {
      include_scripts?: boolean
      include_filming_guides?: boolean
      include_image_prompts?: boolean
      content_ids?: number[]
    }
  ) => {
    const { data } = await api.post(
      `/content/${businessId}/export`,
      {
        export_format: format,
        ...options,
      },
      {
        responseType: format === 'csv' ? 'blob' : 'json',
      }
    )
    return data
  },

  submitFeedback: async (businessId: number, contentId: number, performance: 'viral' | 'good' | 'flop') => {
    const { data } = await api.post<{ status: string; performance_label: string }>(
      `/content/${businessId}/content/${contentId}/feedback`,
      { performance }
    )
    return data
  },
}

// ============ ML PREDICTIONS ============

export interface MLPredictionRequest {
  caption?: string
  hashtags?: string[]
  content_format?: string
  business_type?: string
  video_duration_seconds?: number
  audio_name?: string
}

export interface FeatureImpact {
  feature: string
  impact: number
}

export interface MLEngagementPrediction {
  score: number
  confidence: number
  explanation: {
    top_positive_factors: FeatureImpact[]
    top_negative_factors: FeatureImpact[]
    explanation_text: string
  }
  feature_importance: FeatureImpact[]
}

export interface MLFormatRecommendation {
  recommended_format: string
  confidence: number
  alternatives: { format: string; score: number }[]
  explanation: Record<string, any>
}

export interface MLTriggerSuggestion {
  trigger_type: string
  impact: string
  examples: string[]
  reason: string
}

export interface MLTriggerSuggestions {
  current_triggers: Record<string, number>
  suggestions: MLTriggerSuggestion[]
  improvement_potential: number
}

export interface MLFullPrediction {
  engagement_prediction: MLEngagementPrediction
  format_recommendation: MLFormatRecommendation
  trigger_suggestions: MLTriggerSuggestions
  optimization_suggestions: string[]
  ml_summary: string
}

export interface GrowthProjectionPoint {
  date: string
  followers: number
  daily_gain: number
  scenario: string
}

export interface GrowthProjectionResponse {
  current_followers: number
  growth_rate_base: number
  projected_gain_from_content: number
  projected_total_gain: number
  projection: GrowthProjectionPoint[]
}

// Model Health Types
export interface ModelHealthAlert {
  niche: string
  type?: string
  message: string
  severity?: string
  score?: number
}

export interface ModelHealthSummary {
  total_niches: number
  healthy_count: number
  warning_count: number
  overall_status: 'healthy' | 'warning' | 'no_data' | 'error'
  avg_mae: number | null
  avg_drift_score: number | null
  last_updated: string | null
  alerts: ModelHealthAlert[]
  error?: string
}

export interface NicheHealth {
  niche: string
  last_evaluated: string | null
  evaluation_type: string
  n_samples: number
  metrics: {
    aggregate_mae: number | null
    aggregate_r2: number | null
    aggregate_rmse: number | null
  }
  per_target: Record<string, { mae: number; r2: number; rmse: number }>
  drift: {
    detected: boolean
    score: number
    alert: string | null
  }
  format_metrics: Record<string, { mae: number; n_samples: number }>
  time_metrics: Record<string, { mae: number; n_samples: number }>
  rpi_metrics: { rpi_mae?: number; rpi_correlation?: number }
  calibration: { error: number; well_calibrated: boolean } | null
  insights: string[]
  history?: Array<{ timestamp: string; mae: number; drift_score: number }>
}

export interface ModelHealthFull {
  status: string
  niches: Record<string, NicheHealth>
  alerts: ModelHealthAlert[]
  last_updated: string | null
}

export const mlApi = {
  getModelStatus: async () => {
    const { data } = await api.get<{
      is_trained: boolean
      model_version: string
      feature_count: number
    }>('/ml/status')
    return data
  },

  // Model Health endpoints
  getModelHealthSummary: async () => {
    const { data } = await api.get<ModelHealthSummary>('/ml/health/summary')
    return data
  },

  getModelHealth: async (niche?: string) => {
    const { data } = await api.get<ModelHealthFull>('/ml/health', {
      params: niche ? { niche } : undefined,
    })
    return data
  },

  trainModel: async (useSyntheticData = true, sampleCount = 500) => {
    const { data } = await api.post<{
      success: boolean
      message: string
      metrics: Record<string, any>
    }>('/ml/train', {
      use_synthetic_data: useSyntheticData,
      sample_count: sampleCount,
    })
    return data
  },

  getFullPrediction: async (content: MLPredictionRequest) => {
    const { data } = await api.post<MLFullPrediction>('/ml/predict', content)
    return data
  },

  predictEngagement: async (content: MLPredictionRequest) => {
    const { data } = await api.post<MLEngagementPrediction>(
      '/ml/predict/engagement',
      content
    )
    return data
  },

  recommendFormat: async (content: MLPredictionRequest) => {
    const { data } = await api.post<MLFormatRecommendation>(
      '/ml/predict/format',
      content
    )
    return data
  },

  suggestTriggers: async (content: MLPredictionRequest) => {
    const { data } = await api.post<MLTriggerSuggestions>(
      '/ml/predict/triggers',
      content
    )
    return data
  },

  analyzeDraft: async (content: Record<string, any>) => {
    const { data } = await api.post<
      MLFullPrediction & {
        improvement_roadmap: {
          priority: string
          area: string
          action: string
          potential_gain: string
        }[]
        ready_to_publish: boolean
      }
    >('/ml/analyze-draft', content)
    return data
  },

  getGrowthPrediction: async (businessId: number) => {
    const { data } = await api.get<GrowthProjectionResponse>(`/growth/${businessId}/prediction`)
    return data
  },
}

// ============ VIRAL SCANNER ============

export const viralApi = {
  scan: async (businessId: number, keyword: string, platform = 'instagram') => {
    const { data } = await api.post<{
      keyword: string
      platform: string
      scan_date: string
      opportunities: ViralOpportunity[]
      general_insights: string[]
    }>(`/viral/${businessId}/scan`, {
      keyword,
      platform,
    })
    return data
  },

  getTrending: async (businessId: number, platform = 'instagram') => {
    const { data } = await api.get<{
      platform: string
      business_niche: string
      trending_now: {
        trend: string
        description: string
        example_hook: string
        engagement_potential: string
        difficulty: string
      }[]
      trending_audios: { name: string; type: string }[]
      recommendations: string[]
    }>(`/viral/${businessId}/trending`, { params: { platform } })
    return data
  },

  getQuickIdea: async (businessId: number, trendType: string) => {
    const { data } = await api.post<{
      trend_type: string
      business: string
      idea: {
        hook: string
        structure: string[]
        filming_tips: string
        suggested_audio: string
        difficulty: string
        time_to_create: string
      }
      hashtags: string[]
      best_posting_times: string[]
      ready_to_film: boolean
    }>(`/viral/${businessId}/quick-idea`, null, { params: { trend_type: trendType } })
    return data
  },
}

// ============ KPI WEIGHTS (Multi-Objective) ============

export const kpiApi = {
  getWeights: async (businessId: number, niche?: string) => {
    const { data } = await api.get<KPIWeightsConfig>('/kpi/weights', {
      params: { business_id: businessId, niche },
    })
    return data
  },

  getAllWeights: async () => {
    const { data } = await api.get<KPIWeightsConfig[]>('/kpi/weights/all')
    return data
  },

  saveWeights: async (
    businessId: number,
    weights: EngagementWeights,
    options?: {
      niche?: string
      template_name?: string
      description?: string
    }
  ) => {
    const { data } = await api.post<KPIWeightsConfig>('/kpi/weights', {
      business_id: businessId,
      weights,
      ...options,
    })
    return data
  },

  createFromTemplate: async (
    businessId: number,
    templateName: string,
    niche?: string,
    description?: string
  ) => {
    const { data } = await api.post<KPIWeightsConfig>(
      '/kpi/weights/from-template',
      null,
      {
        params: {
          business_id: businessId,
          template_name: templateName,
          niche,
          description,
        },
      }
    )
    return data
  },

  deleteWeights: async (weightsId: number) => {
    await api.delete(`/kpi/weights/${weightsId}`)
  },

  getTemplates: async () => {
    const { data } = await api.get<{ templates: KPITemplate[] }>('/kpi/templates')
    return data.templates
  },

  previewWeights: async (
    weights: EngagementWeights,
    sampleMetrics?: {
      likes?: number
      comments?: number
      shares?: number
      saves?: number
      views?: number
    }
  ) => {
    const { data } = await api.post<KPIWeightsPreview>(
      '/kpi/preview',
      null,
      {
        params: {
          ...weights,
          sample_likes: sampleMetrics?.likes || 100,
          sample_comments: sampleMetrics?.comments || 10,
          sample_shares: sampleMetrics?.shares || 5,
          sample_saves: sampleMetrics?.saves || 15,
          sample_views: sampleMetrics?.views || 1000,
        },
      }
    )
    return data
  },
}

// ============ MULTI-OUTPUT PREDICTIONS ============

export interface MultiOutputPredictionRequest {
  caption?: string
  hashtags?: string[]
  content_format?: string
  video_duration_seconds?: number
  hook_energy?: number
  retention_energy?: number
  face_in_hook?: boolean
  tempo?: number
  posted_at?: string
  business_id?: number
  business_type?: string
  custom_weights?: EngagementWeights
  author_avg_likes?: number
  author_avg_comments?: number
  author_avg_shares?: number
  author_avg_saves?: number
  author_avg_views?: number
}

export const multiOutputApi = {
  predict: async (request: MultiOutputPredictionRequest) => {
    const { data } = await api.post<MultiOutputPredictionResponse>(
      '/multi-output/predict',
      request
    )
    return data
  },

  predictBatch: async (requests: MultiOutputPredictionRequest[]) => {
    const { data } = await api.post<MultiOutputPredictionResponse[]>(
      '/multi-output/predict/batch',
      requests
    )
    return data
  },

  getModelStatus: async (niche = 'general') => {
    const { data } = await api.get<{
      is_trained: boolean
      niche: string
      model_version: string
      feature_count: number
      targets: string[]
      is_multi_output: boolean
      training_metrics: Record<string, any> | null
    }>('/multi-output/status', { params: { niche } })
    return data
  },

  trainModel: async (niche = 'general', useSynthetic = true, nSamples = 1000) => {
    const { data } = await api.post<{
      success: boolean
      message: string
      metrics: Record<string, any>
    }>('/multi-output/train', null, {
      params: { niche, use_synthetic: useSynthetic, n_samples: nSamples },
    })
    return data
  },

  compareWeights: async (
    request: MultiOutputPredictionRequest,
    weightsConfigs: EngagementWeights[]
  ) => {
    const { data } = await api.post<{
      comparisons: Array<{
        weights: Record<string, number>
        weighted_rpi: number
        predicted_metrics: Record<string, number>
      }>
      best_config: any
      rpi_range: { min: number; max: number }
    }>('/multi-output/compare-weights', {
      ...request,
      weights_configs: weightsConfigs,
    })
    return data
  },
}

// ============ A/B TESTING ============

export interface ABTestVariantCreate {
  variant_name: string
  content_structure?: Record<string, any>
}

export interface ABTestExperimentCreate {
  test_name: string
  original_content_id?: number
  variants: ABTestVariantCreate[]
}

export interface ABTestVariant {
  id: number
  variant_name: string
  alpha_param: number
  beta_param: number
  content_structure?: Record<string, any>
}

export interface ABTestExperiment {
  id: number
  business_id: number
  test_name: string
  original_content_id?: number
  is_active: boolean
  variants: ABTestVariant[]
}

export const abTestApi = {
  createTest: async (data: ABTestExperimentCreate) => {
    const { data: response } = await api.post<ABTestExperiment>('/abtest/create', data)
    return response
  },

  getResults: async (experimentId: number) => {
    const { data } = await api.get<ABTestExperiment>(`/abtest/${experimentId}`)
    return data
  },

  setWinner: async (experimentId: number, variantName: string) => {
    const { data } = await api.post<{ status: string; message: string }>(
      `/abtest/${experimentId}/winner`,
      null,
      { params: { variant_name: variantName } }
    )
    return data
  },
}

// ============ EXTENSION API (API Keys for sync) ============

export const extensionApi = {
  // Generate new API key for extension sync
  generateApiKey: async (businessId: number) => {
    const { data } = await api.post<{
      api_key: string
      key_suffix: string
      message: string
    }>(`/extension/api-key/${businessId}`)
    return data
  },

  // Get API key status (has key, last used)
  getApiKeyStatus: async (businessId: number) => {
    const { data } = await api.get<{
      has_key: boolean
      key_suffix: string | null
      last_used_at: string | null
    }>(`/extension/api-key/${businessId}/status`)
    return data
  },

  // Revoke API key
  revokeApiKey: async (businessId: number) => {
    const { data } = await api.delete(`/extension/api-key/${businessId}`)
    return data
  },
}

// ============ LIGHT MODE CONFIGURATION ============

export interface LightModeOption {
  value: string
  label: string
  description: string
  whisper_model: string
  max_duration_seconds: number
  ocr_max_frames: number
  estimated_time_seconds: number
  estimated_ram_mb: number
  is_recommended: boolean
}

export interface LightModeConfig {
  business_id: number
  light_mode_enabled: boolean
  whisper_model: string
  max_duration_seconds: number
  ocr_max_frames: number
  use_thumbnail: boolean
  cache_enabled: boolean
  skip_non_video: boolean
  description: string
}

export interface LightModeBenchmark {
  mode: string
  estimated_time_seconds: number
  estimated_ram_mb: number
  whisper_model: string
  ocr_frames: number
  hook_duration: number
  is_current: boolean
  description: string
}

export interface LightModeBenchmarkResponse {
  business_id: number
  current_mode: string
  benchmarks: Record<string, LightModeBenchmark>
  recommendation: { mode: string; reason: string }
  time_comparison: {
    light_vs_full_seconds_saved: number
    light_vs_full_percent_saved: number
    example: string
  }
}

export const lightModeApi = {
  getInfo: async () => {
    const { data } = await api.get<{
      available_modes: LightModeOption[]
      current_default: string
      description: string
    }>('/light-mode/info')
    return data
  },

  getConfig: async (businessId: number) => {
    const { data } = await api.get<LightModeConfig>(`/light-mode/${businessId}`)
    return data
  },

  updateConfig: async (
    businessId: number,
    config: Partial<{
      light_mode_enabled: boolean
      whisper_model: string
      max_duration_seconds: number
      ocr_max_frames: number
      use_thumbnail: boolean
      cache_enabled: boolean
      skip_non_video: boolean
    }>
  ) => {
    const { data } = await api.put<LightModeConfig>(`/light-mode/${businessId}`, config)
    return data
  },

  quickToggle: async (businessId: number, enabled: boolean) => {
    const { data } = await api.post<{
      success: boolean
      business_id: number
      light_mode_enabled: boolean
      message: string
      processing_estimate: string
    }>(`/light-mode/quick-toggle/${businessId}`, null, {
      params: { enabled },
    })
    return data
  },

  getBenchmark: async (businessId: number) => {
    const { data } = await api.get<LightModeBenchmarkResponse>(
      `/light-mode/${businessId}/benchmark`
    )
    return data
  },

  getCacheStats: async () => {
    const { data } = await api.get<{
      total_entries: number
      cache_enabled: boolean
      cache_file: string
      ttl_hours: number
    }>('/light-mode/cache/stats')
    return data
  },

  clearCache: async () => {
    const { data } = await api.post<{ success: boolean; message: string }>(
      '/light-mode/cache/clear'
    )
    return data
  },
}

export { api }
export default api
