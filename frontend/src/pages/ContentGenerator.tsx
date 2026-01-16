import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Sparkles,
  Calendar,
  Target,
  Layers,
  RefreshCw,
  CheckCircle,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { contentApi } from '../services/api'
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

  const currentDate = new Date()
  const [formData, setFormData] = useState({
    month: currentDate.getMonth() + 1,
    year: currentDate.getFullYear(),
    posts_count: 15,
    primary_goal: 'engagement',
    platforms: ['instagram', 'tiktok'],
    content_mix: CONTENT_MIX_PRESETS.balanced.mix,
    refresh_competitor_data: true,
  })

  const updateFormData = (updates: Partial<typeof formData>) => {
    setFormData((prev) => ({ ...prev, ...updates }))
  }

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

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        className="btn-primary w-full py-4 text-lg flex items-center justify-center gap-2"
      >
        <Sparkles className="w-5 h-5" />
        Generar Calendario
      </button>

      <p className="text-center text-sm text-gray-500">
        La generación puede tardar 1-2 minutos mientras analizamos y creamos contenido optimizado
      </p>
    </div>
  )
}
