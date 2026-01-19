/**
 * UnifiedConfiguration - Complete User Configuration Component
 *
 * REAL-TIME SYNC:
 * ===============
 * This component syncs ALL user configurations with the backend in real-time.
 * When you change settings here, they IMMEDIATELY affect:
 * - Ingest: own_username detects your profile for ML feedback
 * - Embeddings: precision changes feature dimensions
 * - KPI Weights: your custom weights affect RPI calculation
 * - Multimodal: light/full mode affects video processing speed
 *
 * NO PLACEBO - 100% backend-connected!
 *
 * API Endpoint: GET/POST /api/v1/user/config
 */

import React, { useState, useEffect, useCallback } from 'react'
import {
  Settings,
  Sliders,
  Hash,
  User,
  Video,
  Zap,
  Save,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Info,
  ChevronDown,
  ChevronUp,
  Target,
  Eye,
  Users,
  Share2,
  BarChart2,
} from 'lucide-react'

// Types for the unified configuration
interface DiscoveryConfig {
  hashtags: string[]
  location_keywords: string[]
  niche_keywords: string[]
  follower_min: number
  follower_max: number
  min_recent_posts: number
  exclude_verified: boolean
  exclude_private: boolean
}

interface KPIWeights {
  likes: number
  comments: number
  shares: number
  saves: number
  views: number
}

interface LightModeConfig {
  whisper_model: string
  max_duration_seconds: number
  ocr_max_frames: number
  use_thumbnail: boolean
  cache_enabled: boolean
  skip_non_video: boolean
}

interface UserConfig {
  id?: number
  user_id?: number
  business_id?: number
  discovery: DiscoveryConfig
  kpi_weights: KPIWeights
  kpi_template_name?: string
  embedding_precision: 'ultra_low' | 'low' | 'medium' | 'high' | 'max'
  embedding_dims?: number
  multimodal_mode: 'light' | 'full'
  light_mode_config: LightModeConfig
  own_instagram_username?: string
  own_tiktok_username?: string
  description?: string
  is_active?: boolean
  created_at?: string
  updated_at?: string
}

interface PrecisionOption {
  value: string
  label: string
  dimensions: number
  description: string
  performance: string
  is_recommended: boolean
}

interface KPITemplate {
  name: string
  display_name: string
  description: string
  icon: string
  weights: KPIWeights
}

interface ConfigInfo {
  precision_options: PrecisionOption[]
  multimodal_options: { value: string; label: string; description: string; estimated_time_seconds: number }[]
  kpi_templates: KPITemplate[]
}

interface UnifiedConfigurationProps {
  businessId: number
  userId?: number
  onConfigSaved?: (config: UserConfig) => void
}

const DEFAULT_CONFIG: UserConfig = {
  discovery: {
    hashtags: [],
    location_keywords: [],
    niche_keywords: [],
    follower_min: 0,
    follower_max: 1000000,
    min_recent_posts: 5,
    exclude_verified: false,
    exclude_private: true,
  },
  kpi_weights: {
    likes: 1.0,
    comments: 2.0,
    shares: 10.0,
    saves: 5.0,
    views: 3.0,
  },
  embedding_precision: 'low',
  multimodal_mode: 'light',
  light_mode_config: {
    whisper_model: 'tiny',
    max_duration_seconds: 3.0,
    ocr_max_frames: 5,
    use_thumbnail: true,
    cache_enabled: true,
    skip_non_video: true,
  },
}

const KPI_TEMPLATES: KPITemplate[] = [
  {
    name: 'brand_awareness',
    display_name: 'Brand Awareness',
    description: 'Maximiza alcance y visibilidad',
    icon: 'eye',
    weights: { likes: 1.0, comments: 1.0, shares: 5.0, saves: 2.0, views: 10.0 },
  },
  {
    name: 'leads',
    display_name: 'Leads',
    description: 'Enfoca en guardados e interaccion profunda',
    icon: 'target',
    weights: { likes: 1.0, comments: 5.0, shares: 8.0, saves: 10.0, views: 2.0 },
  },
  {
    name: 'community',
    display_name: 'Comunidad',
    description: 'Prioriza comentarios y engagement',
    icon: 'users',
    weights: { likes: 3.0, comments: 10.0, shares: 5.0, saves: 3.0, views: 1.0 },
  },
  {
    name: 'viral',
    display_name: 'Viral',
    description: 'Maximiza shares y alcance organico',
    icon: 'share',
    weights: { likes: 2.0, comments: 3.0, shares: 15.0, saves: 5.0, views: 8.0 },
  },
  {
    name: 'balanced',
    display_name: 'Balanceado',
    description: 'Equilibrio entre todas las metricas',
    icon: 'balance',
    weights: { likes: 1.0, comments: 2.0, shares: 10.0, saves: 5.0, views: 3.0 },
  },
]

const PRECISION_OPTIONS: PrecisionOption[] = [
  {
    value: 'ultra_low',
    label: 'Ultra Baja (16 dims)',
    dimensions: 16,
    description: 'Ultra rapido para PC modesto',
    performance: '~0.2GB RAM, train <15s',
    is_recommended: false,
  },
  {
    value: 'low',
    label: 'Baja (32 dims)',
    dimensions: 32,
    description: 'Recomendado para sobremesa normal',
    performance: '~0.3GB RAM, train <30s',
    is_recommended: true,
  },
  {
    value: 'medium',
    label: 'Media (64 dims)',
    dimensions: 64,
    description: 'Balance precision/velocidad',
    performance: '~0.5GB RAM, train <1min',
    is_recommended: false,
  },
  {
    value: 'high',
    label: 'Alta (128 dims)',
    dimensions: 128,
    description: 'Alta precision con reduccion',
    performance: '~1.0GB RAM, train <2min',
    is_recommended: false,
  },
  {
    value: 'max',
    label: 'Maxima (384 raw)',
    dimensions: 384,
    description: 'Sin reduccion - mejores matices',
    performance: '~2GB RAM',
    is_recommended: false,
  },
]

export const UnifiedConfiguration: React.FC<UnifiedConfigurationProps> = ({
  businessId,
  userId = 1,
  onConfigSaved,
}) => {
  const [config, setConfig] = useState<UserConfig>(DEFAULT_CONFIG)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    kpi: true,
    embedding: true,
    multimodal: false,
    discovery: false,
    profile: true,
  })

  // Hashtag input state
  const [hashtagInput, setHashtagInput] = useState('')
  const [locationInput, setLocationInput] = useState('')
  const [nicheInput, setNicheInput] = useState('')

  // Load config on mount
  useEffect(() => {
    loadConfig()
  }, [businessId, userId])

  const loadConfig = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(
        `/api/v1/user/config?business_id=${businessId}&user_id=${userId}`
      )
      if (!response.ok) throw new Error('Error al cargar configuracion')
      const data = await response.json()
      setConfig(data)
    } catch (err) {
      console.error('Error loading config:', err)
      setError('No se pudo cargar la configuracion. Usando defaults.')
      setConfig(DEFAULT_CONFIG)
    } finally {
      setLoading(false)
    }
  }

  const saveConfig = async () => {
    setSaving(true)
    setError(null)
    setSuccess(null)
    try {
      const response = await fetch(
        `/api/v1/user/config?business_id=${businessId}&user_id=${userId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
        }
      )
      if (!response.ok) throw new Error('Error al guardar configuracion')
      const data = await response.json()
      setConfig(data)
      setSuccess('Configuracion guardada! Los cambios afectan pipelines inmediatamente.')
      onConfigSaved?.(data)
      setTimeout(() => setSuccess(null), 5000)
    } catch (err) {
      console.error('Error saving config:', err)
      setError('No se pudo guardar la configuracion.')
    } finally {
      setSaving(false)
    }
  }

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  const updateKPIWeights = (key: keyof KPIWeights, value: number) => {
    setConfig(prev => ({
      ...prev,
      kpi_weights: { ...prev.kpi_weights, [key]: value },
      kpi_template_name: undefined, // Clear template when manually changed
    }))
  }

  const applyKPITemplate = (template: KPITemplate) => {
    setConfig(prev => ({
      ...prev,
      kpi_weights: { ...template.weights },
      kpi_template_name: template.name,
    }))
  }

  const addHashtag = (type: 'hashtags' | 'location_keywords' | 'niche_keywords', value: string) => {
    const trimmed = value.trim().replace(/^#/, '')
    if (!trimmed) return
    setConfig(prev => ({
      ...prev,
      discovery: {
        ...prev.discovery,
        [type]: [...prev.discovery[type], trimmed],
      },
    }))
  }

  const removeHashtag = (type: 'hashtags' | 'location_keywords' | 'niche_keywords', index: number) => {
    setConfig(prev => ({
      ...prev,
      discovery: {
        ...prev.discovery,
        [type]: prev.discovery[type].filter((_, i) => i !== index),
      },
    }))
  }

  // Calculate weighted RPI preview
  const calculateRPIPreview = () => {
    const w = config.kpi_weights
    const total = w.likes + w.comments + w.shares + w.saves + w.views
    if (total === 0) return { likes: 20, comments: 20, shares: 20, saves: 20, views: 20 }
    return {
      likes: (w.likes / total) * 100,
      comments: (w.comments / total) * 100,
      shares: (w.shares / total) * 100,
      saves: (w.saves / total) * 100,
      views: (w.views / total) * 100,
    }
  }

  const rpiPreview = calculateRPIPreview()

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin text-blue-500" />
        <span className="ml-2">Cargando configuracion...</span>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b pb-4">
        <div className="flex items-center gap-3">
          <Settings className="w-6 h-6 text-blue-600" />
          <div>
            <h2 className="text-xl font-bold text-gray-900">Configuracion Unificada</h2>
            <p className="text-sm text-gray-500">
              Cambios afectan ingest/train/inference en tiempo real
            </p>
          </div>
        </div>
        <button
          onClick={saveConfig}
          disabled={saving}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          Guardar
        </button>
      </div>

      {/* Alerts */}
      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-50 text-red-700 rounded-lg">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}
      {success && (
        <div className="flex items-center gap-2 p-3 bg-green-50 text-green-700 rounded-lg">
          <CheckCircle className="w-5 h-5" />
          {success}
        </div>
      )}

      {/* Own Profile Section */}
      <div className="border rounded-lg">
        <button
          onClick={() => toggleSection('profile')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-50"
        >
          <div className="flex items-center gap-3">
            <User className="w-5 h-5 text-purple-600" />
            <span className="font-medium">Perfil Propio (Feedback Loop)</span>
          </div>
          {expandedSections.profile ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.profile && (
          <div className="p-4 border-t space-y-4">
            <div className="bg-purple-50 p-3 rounded-lg text-sm text-purple-700">
              <Info className="w-4 h-4 inline mr-1" />
              Cuando ingieres contenido de estos usernames, se activa el feedback loop ML
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Instagram Username
                </label>
                <input
                  type="text"
                  value={config.own_instagram_username || ''}
                  onChange={(e) =>
                    setConfig(prev => ({
                      ...prev,
                      own_instagram_username: e.target.value.replace('@', ''),
                    }))
                  }
                  placeholder="tu_usuario"
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  TikTok Username
                </label>
                <input
                  type="text"
                  value={config.own_tiktok_username || ''}
                  onChange={(e) =>
                    setConfig(prev => ({
                      ...prev,
                      own_tiktok_username: e.target.value.replace('@', ''),
                    }))
                  }
                  placeholder="tu_usuario"
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Embedding Precision Section */}
      <div className="border rounded-lg">
        <button
          onClick={() => toggleSection('embedding')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-50"
        >
          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-yellow-600" />
            <span className="font-medium">Precision de Embeddings</span>
            <span className="text-sm text-gray-500">
              ({PRECISION_OPTIONS.find(p => p.value === config.embedding_precision)?.dimensions || 32} dims)
            </span>
          </div>
          {expandedSections.embedding ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.embedding && (
          <div className="p-4 border-t space-y-4">
            <div className="bg-yellow-50 p-3 rounded-lg text-sm text-yellow-700">
              <Info className="w-4 h-4 inline mr-1" />
              Afecta dimensiones de features ML. Menor = mas rapido, Mayor = mas preciso.
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {PRECISION_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  onClick={() =>
                    setConfig(prev => ({
                      ...prev,
                      embedding_precision: option.value as UserConfig['embedding_precision'],
                    }))
                  }
                  className={`p-3 rounded-lg border-2 text-left transition-all ${
                    config.embedding_precision === option.value
                      ? 'border-yellow-500 bg-yellow-50'
                      : 'border-gray-200 hover:border-yellow-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{option.label}</span>
                    {option.is_recommended && (
                      <span className="text-xs bg-yellow-500 text-white px-2 py-0.5 rounded">
                        Recomendado
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{option.description}</p>
                  <p className="text-xs text-gray-400 mt-1">{option.performance}</p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* KPI Weights Section */}
      <div className="border rounded-lg">
        <button
          onClick={() => toggleSection('kpi')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-50"
        >
          <div className="flex items-center gap-3">
            <Sliders className="w-5 h-5 text-green-600" />
            <span className="font-medium">Pesos KPI (RPI)</span>
            {config.kpi_template_name && (
              <span className="text-sm text-green-600">
                ({KPI_TEMPLATES.find(t => t.name === config.kpi_template_name)?.display_name})
              </span>
            )}
          </div>
          {expandedSections.kpi ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.kpi && (
          <div className="p-4 border-t space-y-4">
            <div className="bg-green-50 p-3 rounded-lg text-sm text-green-700">
              <Info className="w-4 h-4 inline mr-1" />
              Estos pesos afectan el calculo del RPI en predicciones y calendarios.
            </div>

            {/* KPI Templates */}
            <div className="flex flex-wrap gap-2">
              {KPI_TEMPLATES.map((template) => (
                <button
                  key={template.name}
                  onClick={() => applyKPITemplate(template)}
                  className={`px-3 py-1.5 rounded-lg text-sm border transition-all ${
                    config.kpi_template_name === template.name
                      ? 'border-green-500 bg-green-50 text-green-700'
                      : 'border-gray-200 hover:border-green-300'
                  }`}
                >
                  {template.display_name}
                </button>
              ))}
            </div>

            {/* Weight Sliders */}
            <div className="space-y-4">
              {(['likes', 'comments', 'shares', 'saves', 'views'] as const).map((key) => (
                <div key={key} className="flex items-center gap-4">
                  <div className="w-24 text-sm font-medium capitalize">{key}</div>
                  <input
                    type="range"
                    min="0"
                    max="20"
                    step="0.5"
                    value={config.kpi_weights[key]}
                    onChange={(e) => updateKPIWeights(key, parseFloat(e.target.value))}
                    className="flex-1"
                  />
                  <div className="w-12 text-sm text-right">{config.kpi_weights[key]}</div>
                  <div className="w-16 text-xs text-gray-400">
                    ({rpiPreview[key].toFixed(1)}%)
                  </div>
                </div>
              ))}
            </div>

            {/* RPI Preview Bar */}
            <div className="mt-4">
              <div className="text-sm font-medium mb-2">Distribucion RPI</div>
              <div className="h-4 w-full flex rounded-lg overflow-hidden">
                <div
                  style={{ width: `${rpiPreview.likes}%` }}
                  className="bg-pink-400"
                  title={`Likes: ${rpiPreview.likes.toFixed(1)}%`}
                />
                <div
                  style={{ width: `${rpiPreview.comments}%` }}
                  className="bg-blue-400"
                  title={`Comments: ${rpiPreview.comments.toFixed(1)}%`}
                />
                <div
                  style={{ width: `${rpiPreview.shares}%` }}
                  className="bg-green-400"
                  title={`Shares: ${rpiPreview.shares.toFixed(1)}%`}
                />
                <div
                  style={{ width: `${rpiPreview.saves}%` }}
                  className="bg-yellow-400"
                  title={`Saves: ${rpiPreview.saves.toFixed(1)}%`}
                />
                <div
                  style={{ width: `${rpiPreview.views}%` }}
                  className="bg-purple-400"
                  title={`Views: ${rpiPreview.views.toFixed(1)}%`}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Likes</span>
                <span>Comments</span>
                <span>Shares</span>
                <span>Saves</span>
                <span>Views</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Multimodal Mode Section */}
      <div className="border rounded-lg">
        <button
          onClick={() => toggleSection('multimodal')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-50"
        >
          <div className="flex items-center gap-3">
            <Video className="w-5 h-5 text-blue-600" />
            <span className="font-medium">Procesamiento Multimodal</span>
            <span className="text-sm text-gray-500">
              ({config.multimodal_mode === 'light' ? '~5s' : '~20s'} por video)
            </span>
          </div>
          {expandedSections.multimodal ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.multimodal && (
          <div className="p-4 border-t space-y-4">
            <div className="bg-blue-50 p-3 rounded-lg text-sm text-blue-700">
              <Info className="w-4 h-4 inline mr-1" />
              Afecta Whisper (audio) y EasyOCR (texto visual) en ingestion.
            </div>
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setConfig(prev => ({ ...prev, multimodal_mode: 'light' }))}
                className={`p-4 rounded-lg border-2 text-left ${
                  config.multimodal_mode === 'light'
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300'
                }`}
              >
                <div className="font-medium">Modo Ligero</div>
                <p className="text-sm text-gray-500">~5s por video</p>
                <p className="text-xs text-gray-400 mt-1">
                  Whisper tiny, 3s audio, 5 frames OCR
                </p>
              </button>
              <button
                onClick={() => setConfig(prev => ({ ...prev, multimodal_mode: 'full' }))}
                className={`p-4 rounded-lg border-2 text-left ${
                  config.multimodal_mode === 'full'
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300'
                }`}
              >
                <div className="font-medium">Modo Completo</div>
                <p className="text-sm text-gray-500">~20s por video</p>
                <p className="text-xs text-gray-400 mt-1">
                  Whisper base, audio completo, todos frames
                </p>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Discovery Section */}
      <div className="border rounded-lg">
        <button
          onClick={() => toggleSection('discovery')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-50"
        >
          <div className="flex items-center gap-3">
            <Hash className="w-5 h-5 text-orange-600" />
            <span className="font-medium">Discovery (Hashtags & Keywords)</span>
          </div>
          {expandedSections.discovery ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.discovery && (
          <div className="p-4 border-t space-y-4">
            <div className="bg-orange-50 p-3 rounded-lg text-sm text-orange-700">
              <Info className="w-4 h-4 inline mr-1" />
              Para filtrar contenido y competidores en descubrimiento.
            </div>

            {/* Hashtags */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Hashtags de Nicho
              </label>
              <div className="flex gap-2 mb-2">
                <input
                  type="text"
                  value={hashtagInput}
                  onChange={(e) => setHashtagInput(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      addHashtag('hashtags', hashtagInput)
                      setHashtagInput('')
                    }
                  }}
                  placeholder="Agregar hashtag..."
                  className="flex-1 px-3 py-2 border rounded-lg"
                />
                <button
                  onClick={() => {
                    addHashtag('hashtags', hashtagInput)
                    setHashtagInput('')
                  }}
                  className="px-3 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700"
                >
                  Agregar
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {config.discovery.hashtags.map((tag, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 bg-orange-100 text-orange-700 rounded-lg text-sm flex items-center gap-1"
                  >
                    #{tag}
                    <button
                      onClick={() => removeHashtag('hashtags', i)}
                      className="text-orange-500 hover:text-orange-700"
                    >
                      &times;
                    </button>
                  </span>
                ))}
              </div>
            </div>

            {/* Follower Range */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Seguidores
                </label>
                <input
                  type="number"
                  value={config.discovery.follower_min}
                  onChange={(e) =>
                    setConfig(prev => ({
                      ...prev,
                      discovery: { ...prev.discovery, follower_min: parseInt(e.target.value) || 0 },
                    }))
                  }
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Seguidores
                </label>
                <input
                  type="number"
                  value={config.discovery.follower_max}
                  onChange={(e) =>
                    setConfig(prev => ({
                      ...prev,
                      discovery: { ...prev.discovery, follower_max: parseInt(e.target.value) || 1000000 },
                    }))
                  }
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Config Summary */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="text-sm font-medium text-gray-700 mb-2">Resumen de Configuracion</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-gray-500">Embeddings</div>
            <div className="font-medium">{config.embedding_precision} ({config.embedding_dims || PRECISION_OPTIONS.find(p => p.value === config.embedding_precision)?.dimensions} dims)</div>
          </div>
          <div>
            <div className="text-gray-500">Multimodal</div>
            <div className="font-medium">{config.multimodal_mode}</div>
          </div>
          <div>
            <div className="text-gray-500">Own Profile</div>
            <div className="font-medium">@{config.own_instagram_username || 'N/A'}</div>
          </div>
          <div>
            <div className="text-gray-500">KPI Template</div>
            <div className="font-medium">{config.kpi_template_name || 'Custom'}</div>
          </div>
        </div>
        {config.updated_at && (
          <div className="text-xs text-gray-400 mt-2">
            Ultima actualizacion: {new Date(config.updated_at).toLocaleString()}
          </div>
        )}
      </div>
    </div>
  )
}

export default UnifiedConfiguration
