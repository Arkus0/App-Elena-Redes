import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Sparkles,
  Calendar,
  Target,
  Layers,
  RefreshCw,
  CheckCircle,
  Cpu,
  TrendingUp,
  AlertTriangle,
  Info,
  Zap,
  Video,
  VideoOff,
  Camera,
  Timer,
  Film,
  Smartphone
} from 'lucide-react'
import toast from 'react-hot-toast'
import { contentApi, mlApi, type MLFullPrediction } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import clsx from 'clsx'

const MONTHS = [
  'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
  'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
]

const GOALS = [
  { value: 'engagement', label: 'Engagement', description: 'Más likes, comentarios y guardados', emoji: '💬' },
  { value: 'awareness', label: 'Awareness', description: 'Mayor alcance y visibilidad', emoji: '👀' },
  { value: 'leads', label: 'Leads', description: 'Generar mensajes y consultas', emoji: '📩' },
  { value: 'foot_traffic', label: 'Visitas', description: 'Más tráfico a la tienda', emoji: '🚶' },
  { value: 'sales', label: 'Ventas', description: 'Impulsar conversiones', emoji: '💰' },
]

const CONTENT_MIX_PRESETS = {
  video_heavy: { name: 'Video Heavy', mix: { reels: 70, carousels: 20, static: 10 } },
  balanced: { name: 'Balanceado', mix: { reels: 50, carousels: 30, static: 20 } },
  educational: { name: 'Educativo', mix: { reels: 40, carousels: 45, static: 15 } },
}

export default function ContentGenerator() {
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const [isGenerating, setIsGenerating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [mlPrediction, setMlPrediction] = useState<MLFullPrediction | null>(null)
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false)
  const [showMlPreview, setShowMlPreview] = useState(true)

  const currentDate = new Date()
  const [formData, setFormData] = useState({
    month: currentDate.getMonth() + 1,
    year: currentDate.getFullYear(),
    posts_count: 15,
    primary_goal: 'engagement',
    platforms: ['instagram', 'tiktok'],
    content_mix: CONTENT_MIX_PRESETS.balanced.mix,
    refresh_competitor_data: true,
    effort_level: 'medium', // low, medium, pro
    current_mood: 'on_camera', // camera_shy, on_camera
  })

  const updateFormData = (updates: Partial<typeof formData>) => {
    setFormData((prev) => ({ ...prev, ...updates }))
  }

  // Fetch ML prediction preview when settings change
  useEffect(() => {
    if (!currentBusiness || !showMlPreview) return

    const fetchMlPrediction = async () => {
      setIsLoadingPrediction(true)
      try {
        const prediction = await mlApi.getFullPrediction({
          caption: '', // Empty for pre-generation preview
          content_format: formData.content_mix.reels > 50 ? 'reel' : 'carousel',
          business_type: currentBusiness.business_type,
          video_duration_seconds: 30,
        })
        setMlPrediction(prediction)
      } catch (error) {
        console.error('ML prediction error:', error)
      } finally {
        setIsLoadingPrediction(false)
      }
    }

    // Debounce the API call
    const timeoutId = setTimeout(fetchMlPrediction, 500)
    return () => clearTimeout(timeoutId)
  }, [currentBusiness, formData.content_mix, showMlPreview])

  const handleGenerate = async () => {
    if (!currentBusiness) return

    setIsGenerating(true)
    setProgress(0)

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + Math.random() * 15, 90))
    }, 500)

    try {
      const calendar = await contentApi.generateCalendar(currentBusiness.id, formData)
      setProgress(100)
      clearInterval(progressInterval)

      toast.success(`¡Calendario generado! ${calendar.total_pieces} piezas de contenido listas`)
      navigate('/calendar')
    } catch (error: any) {
      clearInterval(progressInterval)
      toast.error(error.response?.data?.detail || 'Error generando calendario')
    } finally {
      setIsGenerating(false)
    }
  }

  if (isGenerating) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center max-w-md">
          <div className="relative mb-6">
            <div className="w-24 h-24 rounded-full bg-brand-600/20 mx-auto flex items-center justify-center">
              <Sparkles className="w-12 h-12 text-brand-400 animate-pulse" />
            </div>
            <div
              className="absolute inset-0 rounded-full border-4 border-brand-500 mx-auto w-24 h-24"
              style={{
                clipPath: `polygon(0 0, 100% 0, 100% ${progress}%, 0 ${progress}%)`,
                transition: 'clip-path 0.3s ease-out',
              }}
            />
          </div>

          <h2 className="text-xl font-bold text-white mb-2">Generando tu calendario...</h2>
          <p className="text-gray-400 mb-4">{Math.round(progress)}% completado</p>

          <div className="space-y-2 text-sm text-gray-500">
            {progress < 30 && <p>📊 Analizando patrones de competidores...</p>}
            {progress >= 30 && progress < 60 && <p>✨ Generando contenido de alto engagement...</p>}
            {progress >= 60 && progress < 90 && <p>🎯 Optimizando hooks y CTAs...</p>}
            {progress >= 90 && <p>✅ Finalizando calendario...</p>}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <div className="w-16 h-16 bg-brand-600/20 rounded-full flex items-center justify-center mx-auto mb-4">
          <Sparkles className="w-8 h-8 text-brand-400" />
        </div>
        <h1 className="text-2xl font-bold text-white">Generar Calendario de Contenido</h1>
        <p className="text-gray-400 mt-2">
          Crea contenido de alto engagement basado en lo que funciona
        </p>
      </div>

      {/* Form */}
      <div className="card space-y-6">
        {/* Month/Year Selection */}
        <div>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-3">
            <Calendar className="w-4 h-4" />
            Periodo
          </label>
          <div className="grid grid-cols-2 gap-4">
            <select
              value={formData.month}
              onChange={(e) => updateFormData({ month: parseInt(e.target.value) })}
              className="input-field"
            >
              {MONTHS.map((month, i) => (
                <option key={i} value={i + 1}>{month}</option>
              ))}
            </select>
            <select
              value={formData.year}
              onChange={(e) => updateFormData({ year: parseInt(e.target.value) })}
              className="input-field"
            >
              {[2024, 2025, 2026].map((year) => (
                <option key={year} value={year}>{year}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Posts Count */}
        <div>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-3">
            <Layers className="w-4 h-4" />
            Cantidad de contenido
          </label>
          <div className="flex items-center gap-4">
            {[10, 15, 20, 30].map((count) => (
              <button
                key={count}
                onClick={() => updateFormData({ posts_count: count })}
                className={clsx(
                  'px-4 py-2 rounded-lg transition-colors',
                  formData.posts_count === count
                    ? 'bg-brand-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                )}
              >
                {count}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {formData.posts_count} piezas = ~{Math.round(formData.posts_count / 4)} por semana
          </p>
        </div>

        {/* Primary Goal */}
        <div>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-3">
            <Target className="w-4 h-4" />
            Objetivo principal
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {GOALS.map((goal) => (
              <button
                key={goal.value}
                onClick={() => updateFormData({ primary_goal: goal.value })}
                className={clsx(
                  'p-3 rounded-lg border text-left transition-all flex items-center gap-3',
                  formData.primary_goal === goal.value
                    ? 'border-brand-500 bg-brand-500/10'
                    : 'border-gray-700 hover:border-gray-600'
                )}
              >
                <span className="text-xl">{goal.emoji}</span>
                <div>
                  <p className="text-white font-medium">{goal.label}</p>
                  <p className="text-xs text-gray-400">{goal.description}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Platforms */}
        <div>
          <label className="text-sm font-medium text-gray-300 mb-3 block">Plataformas</label>
          <div className="flex gap-2">
            {['instagram', 'tiktok', 'linkedin'].map((platform) => (
              <button
                key={platform}
                onClick={() => {
                  const platforms = formData.platforms.includes(platform)
                    ? formData.platforms.filter((p) => p !== platform)
                    : [...formData.platforms, platform]
                  if (platforms.length > 0) {
                    updateFormData({ platforms })
                  }
                }}
                className={clsx(
                  'px-4 py-2 rounded-lg transition-colors flex items-center gap-2',
                  formData.platforms.includes(platform)
                    ? platform === 'instagram'
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                      : platform === 'tiktok'
                      ? 'bg-black text-white border border-white'
                      : 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-400'
                )}
              >
                {platform === 'instagram' ? 'Instagram' : platform === 'tiktok' ? 'TikTok' : 'LinkedIn'}
                {formData.platforms.includes(platform) && <CheckCircle className="w-4 h-4" />}
              </button>
            ))}
          </div>
        </div>

        {/* Content Mix */}
        <div>
          <label className="text-sm font-medium text-gray-300 mb-3 block">Mix de contenido</label>
          <div className="grid grid-cols-3 gap-2 mb-4">
            {Object.entries(CONTENT_MIX_PRESETS).map(([key, preset]) => (
              <button
                key={key}
                onClick={() => updateFormData({ content_mix: preset.mix })}
                className={clsx(
                  'p-3 rounded-lg border text-center transition-all',
                  JSON.stringify(formData.content_mix) === JSON.stringify(preset.mix)
                    ? 'border-brand-500 bg-brand-500/10'
                    : 'border-gray-700 hover:border-gray-600'
                )}
              >
                <p className="text-white text-sm font-medium">{preset.name}</p>
                <p className="text-xs text-gray-400 mt-1">
                  {preset.mix.reels}% Reels
                </p>
              </button>
            ))}
          </div>

          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-gray-400">Reels/Videos</span>
              <span className="text-white font-medium">{formData.content_mix.reels}%</span>
            </div>
            <div className="w-full h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full bg-brand-500 rounded-full"
                style={{ width: `${formData.content_mix.reels}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-sm mt-3 mb-2">
              <span className="text-gray-400">Carousels</span>
              <span className="text-white font-medium">{formData.content_mix.carousels}%</span>
            </div>
            <div className="w-full h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full"
                style={{ width: `${formData.content_mix.carousels}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-sm mt-3 mb-2">
              <span className="text-gray-400">Estático</span>
              <span className="text-white font-medium">{formData.content_mix.static}%</span>
            </div>
            <div className="w-full h-2 bg-gray-600 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 rounded-full"
                style={{ width: `${formData.content_mix.static}%` }}
              />
            </div>
          </div>
        </div>

        {/* Production Constraints */}
        <div className="border-t border-gray-700 pt-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Film className="w-5 h-5 text-brand-400" />
            Restricciones de Producción
          </h3>

          <div className="space-y-6">
            {/* Effort Level */}
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-3">
                <Timer className="w-4 h-4" />
                Nivel de Esfuerzo (Tiempo disponible)
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <button
                  onClick={() => updateFormData({ effort_level: 'low' })}
                  className={clsx(
                    'p-3 rounded-lg border text-left transition-all',
                    formData.effort_level === 'low'
                      ? 'border-green-500 bg-green-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Smartphone className="w-4 h-4 text-green-400" />
                    <span className="text-green-400 font-medium">Bajo (5 min)</span>
                  </div>
                  <p className="text-xs text-gray-400">Solo móvil. Una toma. Sin edición compleja.</p>
                </button>

                <button
                  onClick={() => updateFormData({ effort_level: 'medium' })}
                  className={clsx(
                    'p-3 rounded-lg border text-left transition-all',
                    formData.effort_level === 'medium'
                      ? 'border-yellow-500 bg-yellow-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Video className="w-4 h-4 text-yellow-400" />
                    <span className="text-yellow-400 font-medium">Medio (15 min)</span>
                  </div>
                  <p className="text-xs text-gray-400">Edición básica, textos y cortes simples.</p>
                </button>

                <button
                  onClick={() => updateFormData({ effort_level: 'pro' })}
                  className={clsx(
                    'p-3 rounded-lg border text-left transition-all',
                    formData.effort_level === 'pro'
                      ? 'border-red-500 bg-red-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Film className="w-4 h-4 text-red-400" />
                    <span className="text-red-400 font-medium">Pro (45+ min)</span>
                  </div>
                  <p className="text-xs text-gray-400">Transiciones, B-Roll, voz en off, audio sync.</p>
                </button>
              </div>
            </div>

            {/* Current Mood */}
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300 mb-3">
                <Camera className="w-4 h-4" />
                Mood Actual
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => updateFormData({ current_mood: 'camera_shy' })}
                  className={clsx(
                    'p-3 rounded-lg border text-left transition-all flex items-center gap-3',
                    formData.current_mood === 'camera_shy'
                      ? 'border-brand-500 bg-brand-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  )}
                >
                  <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center">
                    <VideoOff className="w-5 h-5 text-gray-400" />
                  </div>
                  <div>
                    <p className="text-white font-medium">Camera Shy</p>
                    <p className="text-xs text-gray-400">Voz en off, B-roll, producto</p>
                  </div>
                </button>

                <button
                  onClick={() => updateFormData({ current_mood: 'on_camera' })}
                  className={clsx(
                    'p-3 rounded-lg border text-left transition-all flex items-center gap-3',
                    formData.current_mood === 'on_camera'
                      ? 'border-brand-500 bg-brand-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  )}
                >
                  <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center">
                    <Camera className="w-5 h-5 text-brand-400" />
                  </div>
                  <div>
                    <p className="text-white font-medium">On Camera</p>
                    <p className="text-xs text-gray-400">Talking head, personal</p>
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Refresh Data Toggle */}
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={formData.refresh_competitor_data}
            onChange={(e) => updateFormData({ refresh_competitor_data: e.target.checked })}
            className="w-5 h-5 rounded bg-gray-700 border-gray-600 text-brand-500 focus:ring-brand-500"
          />
          <div>
            <p className="text-white">Actualizar datos de competidores</p>
            <p className="text-xs text-gray-400">Escanear nuevos posts antes de generar</p>
          </div>
        </label>
      </div>

      {/* ML Prediction Preview */}
      {showMlPreview && (
        <div className="card bg-gradient-to-br from-indigo-900/30 to-purple-900/20 border border-indigo-500/30">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-indigo-400" />
              <h3 className="font-semibold text-white">Predicción ML (Pre-generación)</h3>
            </div>
            <button
              onClick={() => setShowMlPreview(false)}
              className="text-xs text-gray-400 hover:text-white"
            >
              Ocultar
            </button>
          </div>

          {isLoadingPrediction ? (
            <div className="flex items-center justify-center py-6">
              <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
              <span className="ml-2 text-gray-400">Calculando predicción...</span>
            </div>
          ) : mlPrediction ? (
            <div className="space-y-4">
              {/* Engagement Score */}
              <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className={clsx(
                    'w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold',
                    mlPrediction.engagement_prediction.score >= 70 && 'bg-green-500/20 text-green-400',
                    mlPrediction.engagement_prediction.score >= 50 && mlPrediction.engagement_prediction.score < 70 && 'bg-yellow-500/20 text-yellow-400',
                    mlPrediction.engagement_prediction.score < 50 && 'bg-red-500/20 text-red-400'
                  )}>
                    {Math.round(mlPrediction.engagement_prediction.score)}
                  </div>
                  <div>
                    <p className="text-white font-medium">Score Predicho</p>
                    <p className="text-sm text-gray-400">
                      Confianza: {Math.round(mlPrediction.engagement_prediction.confidence)}%
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={clsx(
                    'text-sm font-medium',
                    mlPrediction.engagement_prediction.score >= 70 && 'text-green-400',
                    mlPrediction.engagement_prediction.score >= 50 && mlPrediction.engagement_prediction.score < 70 && 'text-yellow-400',
                    mlPrediction.engagement_prediction.score < 50 && 'text-red-400'
                  )}>
                    {mlPrediction.engagement_prediction.score >= 70 ? 'Excelente' :
                     mlPrediction.engagement_prediction.score >= 50 ? 'Bueno' : 'Mejorable'}
                  </p>
                </div>
              </div>

              {/* Format Recommendation */}
              <div className="p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-purple-400" />
                  <p className="text-sm font-medium text-white">Formato Recomendado</p>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-purple-400 font-medium capitalize">
                    {mlPrediction.format_recommendation.recommended_format}
                  </span>
                  <span className="text-xs text-gray-400">
                    {Math.round(mlPrediction.format_recommendation.confidence)}% confianza
                  </span>
                </div>
              </div>

              {/* Trigger Suggestions */}
              {mlPrediction.trigger_suggestions.suggestions.length > 0 && (
                <div className="p-3 bg-gray-800/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-4 h-4 text-yellow-400" />
                    <p className="text-sm font-medium text-white">Triggers Recomendados</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {mlPrediction.trigger_suggestions.suggestions.slice(0, 3).map((trigger, i) => (
                      <span
                        key={i}
                        className={clsx(
                          'text-xs px-2 py-1 rounded-full',
                          trigger.impact === 'high' && 'bg-green-500/20 text-green-400',
                          trigger.impact === 'medium' && 'bg-yellow-500/20 text-yellow-400',
                          trigger.impact === 'low' && 'bg-gray-500/20 text-gray-400'
                        )}
                      >
                        {trigger.trigger_type}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Optimization Suggestions */}
              {mlPrediction.optimization_suggestions.length > 0 && (
                <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-400" />
                    <p className="text-sm font-medium text-yellow-400">Sugerencias de Optimización</p>
                  </div>
                  <ul className="space-y-1">
                    {mlPrediction.optimization_suggestions.slice(0, 3).map((suggestion, i) => (
                      <li key={i} className="text-xs text-gray-300 flex items-start gap-2">
                        <span className="text-yellow-400">•</span>
                        {suggestion}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* SHAP Explanation */}
              {mlPrediction.engagement_prediction.explanation?.explanation_text && (
                <div className="p-3 bg-gray-800/50 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Info className="w-4 h-4 text-blue-400" />
                    <p className="text-sm font-medium text-white">Explicación ML (SHAP)</p>
                  </div>
                  <p className="text-xs text-gray-400">
                    {mlPrediction.engagement_prediction.explanation.explanation_text}
                  </p>
                </div>
              )}

              {/* Summary */}
              <p className="text-xs text-gray-500 text-center">
                {mlPrediction.ml_summary}
              </p>
            </div>
          ) : (
            <p className="text-gray-400 text-center py-4">
              No se pudo obtener predicción ML
            </p>
          )}
        </div>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        className="btn-primary w-full py-4 text-lg flex items-center justify-center gap-2"
      >
        <Sparkles className="w-5 h-5" />
        Generar Calendario (Híbrido ML + LLM)
      </button>

      <p className="text-center text-sm text-gray-500">
        El modelo ML predice engagement primero, luego el LLM genera contenido optimizado con sus recomendaciones
      </p>
    </div>
  )
}
