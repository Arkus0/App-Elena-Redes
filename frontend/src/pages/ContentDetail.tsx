import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Copy,
  Download,
  Edit3,
  Sparkles,
  Video,
  Camera,
  Music,
  Hash,
  MessageSquare,
  Save,
  RefreshCw,
  CheckCircle,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { contentApi } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import type { ContentPiece, EngagementPrediction } from '../types'
import clsx from 'clsx'

export default function ContentDetail() {
  const { contentId } = useParams()
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const [content, setContent] = useState<ContentPiece | null>(null)
  const [prediction, setPrediction] = useState<EngagementPrediction | null>(null)
  const [variations, setVariations] = useState<ContentPiece[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isGeneratingVariations, setIsGeneratingVariations] = useState(false)
  const [activeTab, setActiveTab] = useState<'content' | 'script' | 'filming'>('content')

  useEffect(() => {
    if (!currentBusiness || !contentId) return

    const loadContent = async () => {
      try {
        const [contentData, predictionData] = await Promise.all([
          contentApi.getContent(currentBusiness.id, parseInt(contentId)),
          contentApi.getEngagementPrediction(currentBusiness.id, parseInt(contentId)),
        ])
        setContent(contentData)
        setPrediction(predictionData)
      } catch (error) {
        console.error('Error loading content:', error)
        toast.error('Error cargando contenido')
      } finally {
        setIsLoading(false)
      }
    }

    loadContent()
  }, [currentBusiness, contentId])

  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text)
    toast.success(`${label} copiado`)
  }

  const handleGenerateVariations = async () => {
    if (!currentBusiness || !contentId) return

    setIsGeneratingVariations(true)
    try {
      const data = await contentApi.generateVariations(currentBusiness.id, parseInt(contentId))
      setVariations(data.variations)
      toast.success(`${data.variations.length} variaciones generadas`)
    } catch (error) {
      toast.error('Error generando variaciones')
    } finally {
      setIsGeneratingVariations(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!content) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-400">Contenido no encontrado</p>
        <button onClick={() => navigate('/calendar')} className="btn-primary mt-4">
          Volver al calendario
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={() => navigate(-1)} className="p-2 hover:bg-gray-800 rounded-lg">
          <ArrowLeft className="w-5 h-5 text-gray-400" />
        </button>
        <div className="flex-1">
          <h1 className="text-xl font-bold text-white">{content.title}</h1>
          <div className="flex items-center gap-2 mt-1">
            <span className={clsx(
              'badge',
              content.platform === 'instagram' && 'badge-instagram',
              content.platform === 'tiktok' && 'badge-tiktok',
              content.platform === 'linkedin' && 'badge-linkedin'
            )}>
              {content.platform}
            </span>
            <span className="text-xs text-gray-400 bg-gray-700 px-2 py-0.5 rounded">
              {content.content_format}
            </span>
            {content.framework_used && (
              <span className="text-xs text-brand-400">{content.framework_used}</span>
            )}
          </div>
        </div>
        <EngagementBadge score={content.engagement_score} />
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-700">
        {[
          { id: 'content', label: 'Contenido', icon: Edit3 },
          { id: 'script', label: 'Script', icon: Video },
          { id: 'filming', label: 'Guía de Filmación', icon: Camera },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={clsx(
              'flex items-center gap-2 px-4 py-3 border-b-2 transition-colors',
              activeTab === tab.id
                ? 'border-brand-500 text-white'
                : 'border-transparent text-gray-400 hover:text-white'
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Content Tab */}
          {activeTab === 'content' && (
            <>
              {/* Hook */}
              {content.hook_text && (
                <div className="card">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-white flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-brand-400" />
                      Hook
                    </h3>
                    <button
                      onClick={() => handleCopy(content.hook_text!, 'Hook')}
                      className="p-1 text-gray-400 hover:text-white"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                  </div>
                  <p className="text-xl text-brand-400 font-medium">"{content.hook_text}"</p>
                </div>
              )}

              {/* Caption */}
              <div className="card">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-white">Caption</h3>
                  <button
                    onClick={() => handleCopy(content.caption, 'Caption')}
                    className="p-1 text-gray-400 hover:text-white"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                </div>
                <pre className="text-gray-300 whitespace-pre-wrap font-sans text-sm">
                  {content.caption}
                </pre>
              </div>

              {/* Hashtags */}
              {content.hashtags.length > 0 && (
                <div className="card">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-white flex items-center gap-2">
                      <Hash className="w-4 h-4" />
                      Hashtags
                    </h3>
                    <button
                      onClick={() => handleCopy(content.hashtags.map(h => `#${h}`).join(' '), 'Hashtags')}
                      className="p-1 text-gray-400 hover:text-white"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {content.hashtags.map((tag, i) => (
                      <span key={i} className="text-sm text-brand-400">#{tag}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Audio */}
              {content.recommended_audio && (
                <div className="card">
                  <h3 className="font-medium text-white flex items-center gap-2 mb-2">
                    <Music className="w-4 h-4" />
                    Audio Recomendado
                  </h3>
                  <p className="text-white">{content.recommended_audio}</p>
                  {content.audio_alternatives.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-700">
                      <p className="text-xs text-gray-400 mb-1">Alternativas:</p>
                      <div className="flex flex-wrap gap-2">
                        {content.audio_alternatives.map((alt, i) => (
                          <span key={i} className="text-sm text-gray-300">{alt}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {/* Script Tab */}
          {activeTab === 'script' && content.video_script && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-white">Video Script</h3>
                <button
                  onClick={() => handleCopy(content.video_script!, 'Script')}
                  className="btn-secondary text-sm flex items-center gap-1"
                >
                  <Copy className="w-4 h-4" />
                  Copiar
                </button>
              </div>
              <pre className="text-gray-300 whitespace-pre-wrap font-sans text-sm bg-gray-700/50 p-4 rounded-lg">
                {content.video_script}
              </pre>

              {content.script_structure && Object.keys(content.script_structure).length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <h4 className="font-medium text-white mb-3">Estructura por tiempos</h4>
                  <div className="space-y-3">
                    {Object.entries(content.script_structure).map(([key, value]) => (
                      <div key={key} className="flex gap-3">
                        <span className="text-xs text-brand-400 bg-brand-600/20 px-2 py-1 rounded whitespace-nowrap">
                          {key.replace(/_/g, ' ')}
                        </span>
                        <p className="text-sm text-gray-300">{value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Filming Guide Tab */}
          {activeTab === 'filming' && content.filming_guide && (
            <div className="space-y-4">
              <div className="card">
                <h3 className="font-medium text-white mb-3">Setup</h3>
                <p className="text-gray-300">{content.filming_guide.setup}</p>
              </div>

              <div className="card">
                <h3 className="font-medium text-white mb-3">Equipo necesario</h3>
                <ul className="space-y-1">
                  {content.filming_guide.equipment_needed.map((item, i) => (
                    <li key={i} className="text-gray-300 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="card">
                <h3 className="font-medium text-white mb-3">Ángulos de cámara</h3>
                <ul className="space-y-1">
                  {content.filming_guide.camera_angles.map((angle, i) => (
                    <li key={i} className="text-gray-300">• {angle}</li>
                  ))}
                </ul>
              </div>

              {content.filming_guide.text_overlays.length > 0 && (
                <div className="card">
                  <h3 className="font-medium text-white mb-3">Textos en pantalla</h3>
                  <div className="space-y-2">
                    {content.filming_guide.text_overlays.map((overlay, i) => (
                      <div key={i} className="p-2 bg-gray-700/50 rounded flex items-center justify-between">
                        <span className="text-white">"{overlay.text}"</span>
                        <span className="text-xs text-gray-400">
                          {overlay.timing} · {overlay.position}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="card">
                <h3 className="font-medium text-white mb-3">Tips de edición</h3>
                <ul className="space-y-1">
                  {content.filming_guide.editing_tips.map((tip, i) => (
                    <li key={i} className="text-gray-300">• {tip}</li>
                  ))}
                </ul>
              </div>

              <div className="card bg-brand-600/10 border-brand-500/30">
                <p className="text-sm text-gray-300">
                  ⏱️ Tiempo estimado de filmación: <strong>{content.filming_guide.estimated_filming_time}</strong>
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Engagement Prediction */}
          {prediction && (
            <div className="card">
              <h3 className="font-medium text-white mb-4">Predicción de Engagement</h3>

              <div className="text-center mb-4">
                <div className={clsx(
                  'inline-flex items-center justify-center w-20 h-20 rounded-full text-2xl font-bold',
                  prediction.score >= 80 && 'bg-green-500/20 text-green-400 ring-4 ring-green-500/20',
                  prediction.score >= 50 && prediction.score < 80 && 'bg-yellow-500/20 text-yellow-400 ring-4 ring-yellow-500/20',
                  prediction.score < 50 && 'bg-red-500/20 text-red-400 ring-4 ring-red-500/20'
                )}>
                  {Math.round(prediction.score)}
                </div>
                <p className="text-sm text-gray-400 mt-2">{prediction.confidence}% confianza</p>
              </div>

              <div className="space-y-2 mb-4">
                <ScoreBar label="Hook" score={prediction.hook_score} />
                <ScoreBar label="CTA" score={prediction.cta_score} />
                <ScoreBar label="Formato" score={prediction.format_score} />
                <ScoreBar label="Timing" score={prediction.timing_score} />
                <ScoreBar label="Tendencia" score={prediction.trend_alignment_score} />
              </div>

              <div className="p-3 bg-gray-700/50 rounded-lg">
                <p className="text-sm text-gray-300">{prediction.explanation}</p>
              </div>

              {prediction.predicted_metrics && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <p className="text-xs text-gray-400 uppercase mb-2">Métricas predichas</p>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="text-center p-2 bg-gray-700/50 rounded">
                      <p className="text-white">{prediction.predicted_metrics.likes_range}</p>
                      <p className="text-xs text-gray-400">Likes</p>
                    </div>
                    <div className="text-center p-2 bg-gray-700/50 rounded">
                      <p className="text-white">{prediction.predicted_metrics.comments_range}</p>
                      <p className="text-xs text-gray-400">Comentarios</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="card">
            <h3 className="font-medium text-white mb-4">Acciones</h3>
            <div className="space-y-2">
              <button
                onClick={handleGenerateVariations}
                disabled={isGeneratingVariations}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                {isGeneratingVariations ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                Generar variaciones A/B
              </button>
              <button
                onClick={() => handleCopy(content.caption + '\n\n' + content.hashtags.map(h => `#${h}`).join(' '), 'Todo')}
                className="btn-secondary w-full flex items-center justify-center gap-2"
              >
                <Copy className="w-4 h-4" />
                Copiar todo
              </button>
            </div>
          </div>

          {/* Variations */}
          {variations.length > 0 && (
            <div className="card">
              <h3 className="font-medium text-white mb-4">Variaciones A/B</h3>
              <div className="space-y-3">
                {variations.map((v) => (
                  <div
                    key={v.id}
                    onClick={() => navigate(`/content/${v.id}`)}
                    className="p-3 bg-gray-700/50 rounded-lg cursor-pointer hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-white">Versión {v.variation_label}</span>
                      <span className={clsx(
                        'text-sm',
                        v.engagement_score >= 80 && 'text-green-400',
                        v.engagement_score >= 50 && v.engagement_score < 80 && 'text-yellow-400',
                        v.engagement_score < 50 && 'text-red-400'
                      )}>
                        {Math.round(v.engagement_score)}%
                      </span>
                    </div>
                    {v.hook_text && (
                      <p className="text-xs text-gray-400 truncate">"{v.hook_text}"</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scheduling */}
          {content.scheduled_date && (
            <div className="card">
              <h3 className="font-medium text-white mb-2">Programación</h3>
              <p className="text-white">
                {new Date(content.scheduled_date).toLocaleDateString('es-ES', {
                  weekday: 'long',
                  day: 'numeric',
                  month: 'long',
                })}
              </p>
              {content.optimal_posting_time && (
                <p className="text-gray-400 text-sm">Hora óptima: {content.optimal_posting_time}</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function EngagementBadge({ score }: { score: number }) {
  return (
    <div
      className={clsx(
        'w-14 h-14 rounded-full flex items-center justify-center font-bold text-lg',
        score >= 80 && 'bg-green-500/20 text-green-400 ring-2 ring-green-500/30',
        score >= 50 && score < 80 && 'bg-yellow-500/20 text-yellow-400 ring-2 ring-yellow-500/30',
        score < 50 && 'bg-red-500/20 text-red-400 ring-2 ring-red-500/30'
      )}
    >
      {Math.round(score)}
    </div>
  )
}

function ScoreBar({ label, score }: { label: string; score: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-16">{label}</span>
      <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={clsx(
            'h-full rounded-full',
            score >= 80 && 'bg-green-500',
            score >= 50 && score < 80 && 'bg-yellow-500',
            score < 50 && 'bg-red-500'
          )}
          style={{ width: `${score}%` }}
        />
      </div>
      <span className="text-xs text-white w-8 text-right">{Math.round(score)}</span>
    </div>
  )
}
