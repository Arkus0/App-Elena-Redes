import axios from 'axios'
import { useAuthStore } from '../stores/authStore'
import type {
  Business,
  BusinessStatus,
  Competitor,
  CompetitorAnalysis,
  ContentCalendar,
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
      null,
      { params: { email, password } }
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

export const mlApi = {
  getModelStatus: async () => {
    const { data } = await api.get<{
      is_trained: boolean
      model_version: string
      feature_count: number
    }>('/ml/status')
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

export default api
