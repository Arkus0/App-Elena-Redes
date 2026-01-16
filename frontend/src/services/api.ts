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

export default api
