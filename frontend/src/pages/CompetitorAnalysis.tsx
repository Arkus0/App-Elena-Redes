import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  BarChart3,
  TrendingUp,
  Clock,
  Hash,
  MessageSquare,
  Heart,
  Bookmark,
  Share2,
  ExternalLink,
  Lightbulb,
} from 'lucide-react'
import { competitorsApi } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import type { CompetitorAnalysis as CompetitorAnalysisType, TopPost } from '../types'
import clsx from 'clsx'

export default function CompetitorAnalysis() {
  const { competitorId } = useParams()
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const [analysis, setAnalysis] = useState<CompetitorAnalysisType | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!currentBusiness || !competitorId) return

    const loadAnalysis = async () => {
      try {
        const data = await competitorsApi.getAnalysis(currentBusiness.id, parseInt(competitorId))
        setAnalysis(data)
      } catch (error) {
        console.error('Error loading analysis:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadAnalysis()
  }, [currentBusiness, competitorId])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!analysis) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-400">Análisis no disponible</p>
        <button onClick={() => navigate('/competitors')} className="btn-primary mt-4">
          Volver a competidores
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={() => navigate('/competitors')} className="p-2 hover:bg-gray-800 rounded-lg">
          <ArrowLeft className="w-5 h-5 text-gray-400" />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white">@{analysis.handle}</h1>
          <p className="text-gray-400">
            {analysis.followers.toLocaleString()} seguidores · {analysis.total_posts_analyzed} posts analizados
          </p>
        </div>
      </div>

      {/* Strategy Summary */}
      <div className="card bg-gradient-to-r from-brand-600/20 to-purple-600/20 border-brand-500/30">
        <h2 className="text-lg font-semibold text-white mb-2">Resumen de Estrategia</h2>
        <p className="text-gray-300">{analysis.overall_strategy_summary}</p>
      </div>

      {/* Key Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Mejores Formatos" items={analysis.best_formats} />
        <StatCard label="Mejores Horarios" items={analysis.best_posting_times} />
        <StatCard label="Pilares de Contenido" items={analysis.best_content_pillars} />
        <StatCard label="Hashtags Top" items={analysis.trending_hashtags.slice(0, 3)} />
      </div>

      {/* Key Takeaways */}
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-yellow-400" />
          Aprendizajes Clave
        </h2>
        <ul className="space-y-2">
          {analysis.key_takeaways.map((takeaway, i) => (
            <li key={i} className="flex items-start gap-2 text-gray-300">
              <span className="text-brand-400 mt-0.5">•</span>
              {takeaway}
            </li>
          ))}
        </ul>
      </div>

      {/* Content Gaps */}
      <div className="card border-orange-500/30">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-orange-400" />
          Oportunidades (Content Gaps)
        </h2>
        <ul className="space-y-2">
          {analysis.content_gaps.map((gap, i) => (
            <li key={i} className="flex items-start gap-2 text-gray-300">
              <span className="text-orange-400 mt-0.5">→</span>
              {gap}
            </li>
          ))}
        </ul>
      </div>

      {/* Top Posts */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Posts Top</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {analysis.top_posts.map((post) => (
            <TopPostCard key={post.post_id} post={post} />
          ))}
        </div>
      </div>

      {/* Winning Patterns */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Patrones Ganadores</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {analysis.winning_patterns.map((pattern, i) => (
            <PatternCard key={i} pattern={pattern} />
          ))}
        </div>
      </div>

      {/* Recommended Hooks & CTAs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-semibold text-white mb-3">Hooks Recomendados</h3>
          <ul className="space-y-2">
            {analysis.recommended_hooks.map((hook, i) => (
              <li key={i} className="text-gray-300 text-sm p-2 bg-gray-700/50 rounded">
                "{hook}"
              </li>
            ))}
          </ul>
        </div>
        <div className="card">
          <h3 className="font-semibold text-white mb-3">CTAs Efectivos</h3>
          <ul className="space-y-2">
            {analysis.recommended_ctas.map((cta, i) => (
              <li key={i} className="text-gray-300 text-sm p-2 bg-gray-700/50 rounded">
                "{cta}"
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, items }: { label: string; items: string[] }) {
  return (
    <div className="card">
      <p className="text-sm text-gray-400 mb-2">{label}</p>
      <div className="space-y-1">
        {items.slice(0, 3).map((item, i) => (
          <p key={i} className="text-white text-sm">{item}</p>
        ))}
      </div>
    </div>
  )
}

function TopPostCard({ post }: { post: TopPost }) {
  return (
    <div className="card-hover">
      <div className="flex gap-4">
        {post.thumbnail_url && (
          <div className="w-20 h-20 rounded-lg bg-gray-700 flex-shrink-0 overflow-hidden">
            <img src={post.thumbnail_url} alt="" className="w-full h-full object-cover" />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span className={clsx(
              'engagement-score text-sm w-8 h-8',
              post.engagement_score >= 80 && 'engagement-score-high',
              post.engagement_score >= 50 && post.engagement_score < 80 && 'engagement-score-medium',
              post.engagement_score < 50 && 'engagement-score-low'
            )}>
              {Math.round(post.engagement_score)}
            </span>
            <span className="text-xs text-gray-400 px-2 py-0.5 bg-gray-700 rounded">
              {post.content_format}
            </span>
          </div>
          <p className="text-sm text-gray-300 line-clamp-2">{post.caption_preview}</p>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mt-4 pt-4 border-t border-gray-700">
        <MetricItem icon={Heart} value={post.likes} />
        <MetricItem icon={MessageSquare} value={post.comments} />
        <MetricItem icon={Bookmark} value={post.saves} />
        <MetricItem icon={Share2} value={post.shares} />
      </div>

      {post.hook_type && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <p className="text-xs text-gray-400">
            Hook: <span className="text-brand-400">{post.hook_type}</span>
            {post.cta_type && (
              <> · CTA: <span className="text-brand-400">{post.cta_type}</span></>
            )}
          </p>
        </div>
      )}

      <p className="text-xs text-gray-400 mt-2">{post.why_it_worked}</p>

      {post.post_url && (
        <a
          href={post.post_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300 mt-2"
        >
          Ver post <ExternalLink className="w-3 h-3" />
        </a>
      )}
    </div>
  )
}

function MetricItem({ icon: Icon, value }: { icon: React.ElementType; value: number }) {
  return (
    <div className="text-center">
      <Icon className="w-4 h-4 text-gray-500 mx-auto mb-1" />
      <p className="text-sm text-white">{value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value}</p>
    </div>
  )
}

function PatternCard({ pattern }: { pattern: { pattern_type: string; pattern_name: string; description: string; confidence_score: number } }) {
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-400 uppercase tracking-wider">{pattern.pattern_type}</span>
        <span className="text-sm text-brand-400">{Math.round(pattern.confidence_score)}% confianza</span>
      </div>
      <h4 className="font-medium text-white mb-1">{pattern.pattern_name}</h4>
      <p className="text-sm text-gray-400">{pattern.description}</p>
    </div>
  )
}
