// Business Types
export type BusinessType =
  | 'inmobiliaria'
  | 'floristeria'
  | 'cafeteria'
  | 'peluqueria'
  | 'tienda_local'
  | 'restaurante'
  | 'gimnasio'
  | 'clinica'
  | 'otros'

export type Platform = 'instagram' | 'tiktok' | 'linkedin'

export type ContentFormat =
  | 'reel'
  | 'carousel'
  | 'static_image'
  | 'story'
  | 'tiktok_video'
  | 'linkedin_post'
  | 'linkedin_carousel'

export type ContentGoal =
  | 'awareness'
  | 'leads'
  | 'foot_traffic'
  | 'sales'
  | 'engagement'
  | 'brand_building'

export type ContentStatus = 'draft' | 'ready' | 'scheduled' | 'published' | 'archived'

// User
export interface User {
  id: number
  email: string
  full_name: string | null
  is_active: boolean
  created_at: string
}

// Business
export interface Business {
  id: number
  name: string
  business_type: BusinessType
  description: string | null
  location: string | null
  instagram_handle: string | null
  tiktok_handle: string | null
  linkedin_handle: string | null
  active_platforms: Platform[]
  content_goals: string[]
  posting_frequency: string
  brand_voice: string
  onboarding_completed: string | null
  last_analysis_at: string | null
  created_at: string
}

// Competitor
export interface Competitor {
  id: number
  platform: Platform
  handle: string
  display_name: string | null
  bio: string | null
  followers_count: number
  posts_count: number
  avg_engagement_rate: number
  last_scraped_at: string | null
  scrape_status: 'pending' | 'in_progress' | 'completed' | 'failed'
  top_performing_formats: string[]
  common_hashtags: string[]
}

export interface CompetitorPreviewResponse {
  platform: Platform
  handle: string
  profile_pic_url: string | null
  full_name: string | null
  biography: string | null
  followers_count: number
  is_private: boolean
  verified: boolean
}

export interface CompetitorDiscoveryRequest {
  hashtags: string[]
  location_keywords: string[]
  niche_keywords: string[]
  min_followers: number
  max_followers: number
  require_active: boolean
  max_days_since_last_post: number
  min_posts_last_month: number
}

export interface DiscoveredCompetitor {
  handle: string
  platform: string
  followers: number
  relevance_score: number
  activity_status: string
  last_post_date: string | null
  match_reasons: string[]
  profile_pic_url: string | null
}

// Scraped Post
export interface ScrapedPost {
  id: number
  platform_post_id: string
  post_url: string | null
  content_format: ContentFormat
  caption_preview: string | null
  thumbnail_url: string | null
  likes_count: number
  comments_count: number
  shares_count: number
  saves_count: number
  views_count: number
  engagement_score: number
  posted_at: string | null
  hook_type: string | null
  cta_type: string | null
}

// Pattern
export interface PatternInsight {
  pattern_type: string
  pattern_name: string
  description: string
  confidence_score: number
  examples_count: number
  avg_engagement: number
}

// Top Post with analysis
export interface TopPost {
  post_id: number
  post_url: string | null
  thumbnail_url: string | null
  content_format: string
  engagement_score: number
  likes: number
  comments: number
  saves: number
  shares: number
  caption_preview: string
  hook_type: string | null
  hook_text: string | null
  cta_type: string | null
  emotional_triggers: string[]
  why_it_worked: string
}

// Competitor Analysis
export interface CompetitorAnalysis {
  competitor_id: number
  handle: string
  platform: Platform
  followers: number
  total_posts_analyzed: number
  analysis_date: string
  top_posts: TopPost[]
  winning_patterns: PatternInsight[]
  best_posting_times: string[]
  best_formats: string[]
  best_content_pillars: string[]
  trending_hashtags: string[]
  recommended_hooks: string[]
  recommended_ctas: string[]
  overall_strategy_summary: string
  key_takeaways: string[]
  content_gaps: string[]
}

// Filming Guide
export interface FilmingGuide {
  setup: string
  equipment_needed: string[]
  lighting_tips: string
  camera_angles: string[]
  props_needed: string[]
  location_suggestions: string[]
  text_overlays: { text: string; timing: string; position: string }[]
  b_roll_ideas: string[]
  estimated_filming_time: string
  editing_tips: string[]
}

// ML Prediction Data
export interface MLPredictionData {
  score: number
  confidence: number
  top_factors: { feature: string; impact: number }[]
  format_recommendation: {
    recommended_format: string
    confidence: number
  }
  triggers_used: string[]
}

// Generated Content
export interface ContentPiece {
  id: number
  title: string
  platform: Platform
  content_format: ContentFormat
  status: ContentStatus
  caption: string
  hashtags: string[]
  hook_text: string | null
  video_script: string | null
  script_structure: Record<string, string> | null
  filming_guide: FilmingGuide | null
  recommended_audio: string | null
  audio_alternatives: string[]
  image_prompt: string | null
  engagement_score: number
  engagement_explanation: string
  similar_viral_posts: { id: number; engagement: number }[]
  patterns_used: string[]
  framework_used: string | null
  content_goal: ContentGoal | null
  cta_type: string | null
  scheduled_date: string | null
  optimal_posting_time: string | null
  variation_label: string | null
  has_variations: boolean
  ml_prediction_data?: MLPredictionData  // Hybrid ML/LLM prediction data
  performance_label?: string | null // viral, good, flop
}

// Content Calendar
export interface ContentCalendar {
  calendar_id: number
  name: string
  month: number
  year: number
  primary_goal: string
  total_pieces: number
  status: string
  content_by_platform: Record<string, number>
  content_by_format: Record<string, number>
  avg_engagement_score: number
  top_predicted_posts: ContentPiece[]
  content_pieces: ContentPiece[]
  calendar_insights: string[]
  key_themes: string[]
  created_at: string
}

// Engagement Prediction
export interface EngagementPrediction {
  content_id: number
  score: number
  confidence: number
  hook_score: number
  cta_score: number
  format_score: number
  timing_score: number
  trend_alignment_score: number
  explanation: string
  strengths: string[]
  weaknesses: string[]
  improvement_suggestions: string[]
  similar_competitor_posts: Record<string, unknown>[]
  predicted_metrics: Record<string, string>
}

// Viral Opportunity
export interface ViralOpportunity {
  trend_id: string
  trend_name: string
  platform: Platform
  description: string
  total_views: number
  total_videos: number
  growth_rate: string
  time_sensitive: boolean
  relevance_score: number
  difficulty_score: number
  content_ideas: {
    title: string
    hook: string
    script_outline: string[]
    filming_tips: string
  }[]
  top_examples: { url: string; views: number; why_it_worked: string }[]
  best_time_to_post: string
  trend_expiry_estimate: string
}

// API Responses
export interface OnboardingResponse {
  business_id: number
  message: string
  competitors_added: number
  scraping_status: string
  estimated_analysis_time: string
  next_steps: string[]
}

export interface BusinessStatus {
  business_id: number
  onboarding_completed: boolean
  last_analysis: string | null
  competitors: {
    total: number
    completed: number
    in_progress: number
    failed: number
    pending: number
  }
  patterns_extracted: number
  ready_for_content: boolean
}

// =============================================================================
// KPI Weights - Multi-Objective Engagement Configuration
// =============================================================================

export interface EngagementWeights {
  likes_weight: number
  comments_weight: number
  shares_weight: number
  saves_weight: number
  views_weight: number
}

export interface KPIWeightsConfig {
  id: number
  user_id: number
  business_id: number
  niche: string | null
  weights: EngagementWeights
  template_name: string | null
  description: string | null
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface KPITemplate {
  name: string
  display_name: string
  description: string
  weights: EngagementWeights
  use_case: string
  icon: string
}

export interface KPIWeightsPreview {
  weights: EngagementWeights
  sample_prediction: Record<string, number>
  weighted_rpi: number
  explanation: string
  comparison_to_default: {
    default_rpi: number
    difference: number
    percent_change: number
  }
}

// Multi-Output Prediction Types
export interface MultiOutputPrediction {
  log_likes: number
  log_comments: number
  log_shares: number
  log_saves: number
  log_views: number
  predicted_likes: number
  predicted_comments: number
  predicted_shares: number
  predicted_saves: number
  predicted_views: number
  weighted_rpi: number
  weights_used: EngagementWeights
  relative_to_baseline: number | null
}

export interface MultiOutputPredictionResponse {
  prediction: MultiOutputPrediction
  shap_likes: Record<string, number> | null
  shap_comments: Record<string, number> | null
  shap_shares: Record<string, number> | null
  shap_saves: Record<string, number> | null
  shap_views: Record<string, number> | null
  explanation: string
  confidence_interval: Record<string, [number, number]>
  model_version: string
  is_multi_output: boolean
}
