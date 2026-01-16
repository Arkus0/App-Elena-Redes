import { useState } from 'react'
import {
  TrendingUp,
  Search,
  Sparkles,
  Clock,
  Zap,
  Video,
  Music,
  ChevronRight,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { viralApi } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import clsx from 'clsx'

const QUICK_IDEAS = [
  { type: 'pov', label: 'POV', emoji: '👀' },
  { type: 'transformation', label: 'Before/After', emoji: '✨' },
  { type: 'tips', label: 'Tips rápidos', emoji: '💡' },
  { type: 'storytime', label: 'Storytime', emoji: '📖' },
  { type: 'dayinlife', label: 'Un día en...', emoji: '🌅' },
]

export default function ViralScanner() {
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const [keyword, setKeyword] = useState('')
  const [platform, setPlatform] = useState('instagram')
  const [isScanning, setIsScanning] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [trending, setTrending] = useState<any>(null)
  const [quickIdea, setQuickIdea] = useState<any>(null)
  const [isLoadingTrending, setIsLoadingTrending] = useState(false)
  const [isLoadingQuick, setIsLoadingQuick] = useState(false)

  const handleScan = async () => {
    if (!currentBusiness || !keyword) return

    setIsScanning(true)
    try {
      const data = await viralApi.scan(currentBusiness.id, keyword, platform)
      setResults(data)
      toast.success(`${data.opportunities.length} oportunidades encontradas`)
    } catch (error) {
      toast.error('Error escaneando tendencias')
    } finally {
      setIsScanning(false)
    }
  }

  const loadTrending = async () => {
    if (!currentBusiness) return

    setIsLoadingTrending(true)
    try {
      const data = await viralApi.getTrending(currentBusiness.id, platform)
      setTrending(data)
    } catch (error) {
      toast.error('Error cargando tendencias')
    } finally {
      setIsLoadingTrending(false)
    }
  }

  const generateQuickIdea = async (trendType: string) => {
    if (!currentBusiness) return

    setIsLoadingQuick(true)
    try {
      const data = await viralApi.getQuickIdea(currentBusiness.id, trendType)
      setQuickIdea(data)
    } catch (error) {
      toast.error('Error generando idea')
    } finally {
      setIsLoadingQuick(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-brand-400" />
          Scanner Viral
        </h1>
        <p className="text-gray-400 mt-1">
          Encuentra tendencias y genera ideas de contenido reactivo
        </p>
      </div>

      {/* Search */}
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-4">Buscar tendencia</h2>
        <div className="flex gap-3">
          <select
            value={platform}
            onChange={(e) => setPlatform(e.target.value)}
            className="input-field w-32"
          >
            <option value="instagram">Instagram</option>
            <option value="tiktok">TikTok</option>
          </select>
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
            <input
              type="text"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleScan()}
              placeholder="Buscar tendencia o keyword..."
              className="input-field pl-10"
            />
          </div>
          <button
            onClick={handleScan}
            disabled={!keyword || isScanning}
            className="btn-primary flex items-center gap-2"
          >
            {isScanning ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <Zap className="w-4 h-4" />
                Escanear
              </>
            )}
          </button>
        </div>
      </div>

      {/* Quick Ideas */}
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-brand-400" />
          Ideas Rápidas
        </h2>
        <p className="text-gray-400 text-sm mb-4">
          Genera una idea lista para filmar basada en formatos que funcionan
        </p>
        <div className="flex flex-wrap gap-2">
          {QUICK_IDEAS.map((idea) => (
            <button
              key={idea.type}
              onClick={() => generateQuickIdea(idea.type)}
              disabled={isLoadingQuick}
              className="btn-secondary flex items-center gap-2"
            >
              <span>{idea.emoji}</span>
              {idea.label}
            </button>
          ))}
        </div>

        {/* Quick Idea Result */}
        {quickIdea && (
          <div className="mt-6 p-4 bg-gray-700/50 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-white">{quickIdea.trend_type.toUpperCase()}</h3>
              <span className="text-xs text-gray-400 bg-gray-600 px-2 py-1 rounded">
                {quickIdea.idea.difficulty} · {quickIdea.idea.time_to_create}
              </span>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-xs text-gray-400 uppercase mb-1">Hook</p>
                <p className="text-white font-medium">"{quickIdea.idea.hook}"</p>
              </div>

              <div>
                <p className="text-xs text-gray-400 uppercase mb-1">Estructura</p>
                <ol className="space-y-1">
                  {quickIdea.idea.structure.map((step: string, i: number) => (
                    <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                      <span className="text-brand-400 font-medium">{i + 1}.</span>
                      {step}
                    </li>
                  ))}
                </ol>
              </div>

              <div>
                <p className="text-xs text-gray-400 uppercase mb-1">Tips de filmación</p>
                <p className="text-sm text-gray-300">{quickIdea.idea.filming_tips}</p>
              </div>

              <div className="flex items-center gap-2">
                <Music className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-300">Audio: {quickIdea.idea.suggested_audio}</span>
              </div>

              <div className="flex flex-wrap gap-2">
                {quickIdea.hashtags.map((tag: string, i: number) => (
                  <span key={i} className="text-xs text-brand-400">#{tag}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Trending in Niche */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Trending en tu nicho</h2>
          <button
            onClick={loadTrending}
            disabled={isLoadingTrending}
            className="btn-secondary text-sm flex items-center gap-1"
          >
            {isLoadingTrending ? (
              <div className="w-4 h-4 border-2 border-gray-400/30 border-t-gray-400 rounded-full animate-spin" />
            ) : (
              'Cargar'
            )}
          </button>
        </div>

        {trending ? (
          <div className="space-y-6">
            {/* Trending formats */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {trending.trending_now.map((trend: any, i: number) => (
                <div key={i} className="p-4 bg-gray-700/50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-white">{trend.trend}</h4>
                    <span className={clsx(
                      'text-xs px-2 py-0.5 rounded',
                      trend.engagement_potential === 'muy alto' && 'bg-green-500/20 text-green-400',
                      trend.engagement_potential === 'alto' && 'bg-yellow-500/20 text-yellow-400'
                    )}>
                      {trend.engagement_potential}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 mb-2">{trend.description}</p>
                  <p className="text-sm text-brand-400">"{trend.example_hook}"</p>
                  <p className="text-xs text-gray-500 mt-2">Dificultad: {trend.difficulty}</p>
                </div>
              ))}
            </div>

            {/* Trending Audios */}
            <div>
              <h3 className="font-medium text-white mb-3 flex items-center gap-2">
                <Music className="w-4 h-4" />
                Audios Trending
              </h3>
              <div className="flex flex-wrap gap-2">
                {trending.trending_audios.map((audio: any, i: number) => (
                  <span
                    key={i}
                    className="text-sm px-3 py-1.5 bg-gray-700 rounded-full text-gray-300"
                  >
                    🎵 {audio.name}
                  </span>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-gradient-to-r from-brand-600/10 to-purple-600/10 rounded-lg p-4 border border-brand-500/30">
              <h3 className="font-medium text-white mb-2">Recomendaciones</h3>
              <ul className="space-y-1">
                {trending.recommendations.map((rec: string, i: number) => (
                  <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                    <span className="text-brand-400">→</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            Haz click en "Cargar" para ver las tendencias actuales
          </p>
        )}
      </div>

      {/* Scan Results */}
      {results && (
        <div className="card">
          <h2 className="text-lg font-semibold text-white mb-4">
            Resultados para "{results.keyword}"
          </h2>

          {results.opportunities.length > 0 ? (
            <div className="space-y-4">
              {results.opportunities.map((opp: any, i: number) => (
                <div key={i} className="p-4 bg-gray-700/50 rounded-lg">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-medium text-white">{opp.trend_name}</h3>
                      <p className="text-sm text-gray-400">{opp.description}</p>
                    </div>
                    <div className="text-right">
                      <span className={clsx(
                        'text-sm font-medium',
                        opp.relevance_score >= 80 && 'text-green-400',
                        opp.relevance_score >= 50 && opp.relevance_score < 80 && 'text-yellow-400',
                        opp.relevance_score < 50 && 'text-gray-400'
                      )}>
                        {Math.round(opp.relevance_score)}% relevante
                      </span>
                      {opp.time_sensitive && (
                        <p className="text-xs text-orange-400 flex items-center gap-1 justify-end mt-1">
                          <Clock className="w-3 h-3" />
                          Urgente
                        </p>
                      )}
                    </div>
                  </div>

                  {opp.content_ideas.length > 0 && (
                    <div className="border-t border-gray-600 pt-3 mt-3">
                      <p className="text-xs text-gray-400 uppercase mb-2">Ideas de contenido</p>
                      {opp.content_ideas.map((idea: any, j: number) => (
                        <div key={j} className="mb-2 last:mb-0">
                          <p className="text-white text-sm font-medium">{idea.title}</p>
                          <p className="text-brand-400 text-sm">Hook: "{idea.hook}"</p>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-600 text-xs text-gray-500">
                    <span>Mejor hora: {opp.best_time_to_post}</span>
                    <span>Vigencia: {opp.trend_expiry_estimate}</span>
                  </div>
                </div>
              ))}

              {/* General Insights */}
              {results.general_insights.length > 0 && (
                <div className="p-4 bg-brand-600/10 rounded-lg border border-brand-500/30">
                  <h4 className="font-medium text-white mb-2">Insights</h4>
                  <ul className="space-y-1">
                    {results.general_insights.map((insight: string, i: number) => (
                      <li key={i} className="text-sm text-gray-300">• {insight}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">
              No se encontraron oportunidades para esta búsqueda
            </p>
          )}
        </div>
      )}
    </div>
  )
}
