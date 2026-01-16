import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  BarChart3,
  Users,
  Calendar,
  TrendingUp,
  Sparkles,
  ArrowRight,
  RefreshCw,
  CheckCircle,
  Clock,
  AlertCircle,
} from 'lucide-react'
import { useBusinessStore } from '../stores/businessStore'
import { businessApi, competitorsApi, contentApi } from '../services/api'
import clsx from 'clsx'

export default function Dashboard() {
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const setStatus = useBusinessStore((state) => state.setStatus)
  const setCompetitors = useBusinessStore((state) => state.setCompetitors)
  const setCalendars = useBusinessStore((state) => state.setCalendars)
  const status = useBusinessStore((state) => state.status)
  const competitors = useBusinessStore((state) => state.competitors)
  const calendars = useBusinessStore((state) => state.calendars)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!currentBusiness) {
      navigate('/onboarding')
      return
    }

    const loadData = async () => {
      setIsLoading(true)
      try {
        const [statusData, competitorsData, calendarsData] = await Promise.all([
          businessApi.getStatus(currentBusiness.id),
          competitorsApi.getCompetitors(currentBusiness.id),
          contentApi.getCalendars(currentBusiness.id),
        ])
        setStatus(statusData)
        setCompetitors(competitorsData)
        setCalendars(calendarsData)
      } catch (error) {
        console.error('Error loading dashboard data:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadData()
  }, [currentBusiness, navigate, setStatus, setCompetitors, setCalendars])

  if (!currentBusiness) {
    return null
  }

  const completedCompetitors = competitors.filter((c) => c.scrape_status === 'completed').length
  const totalContent = calendars.reduce((sum, cal) => sum + cal.total_pieces, 0)

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">
          ¡Hola! Bienvenido a {currentBusiness.name}
        </h1>
        <p className="text-gray-400 mt-1">
          Tu co-piloto de contenido de alto engagement está listo
        </p>
      </div>

      {/* Status Banner */}
      {status && !status.ready_for_content && (
        <div className="card bg-gradient-to-r from-brand-600/20 to-purple-600/20 border-brand-500/30">
          <div className="flex items-start gap-4">
            <div className="p-2 bg-brand-500/20 rounded-lg">
              <RefreshCw className="w-6 h-6 text-brand-400 animate-spin" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Análisis en progreso</h3>
              <p className="text-gray-300 text-sm mt-1">
                Estamos analizando a tus competidores para extraer los patrones de engagement que funcionan.
                {status.competitors.completed > 0 &&
                  ` ${status.competitors.completed}/${status.competitors.total} completados.`}
              </p>
              <p className="text-gray-400 text-sm mt-2">
                {status.patterns_extracted} patrones extraídos hasta ahora
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Users}
          label="Competidores"
          value={`${completedCompetitors}/${competitors.length}`}
          sublabel="analizados"
          color="blue"
        />
        <StatCard
          icon={BarChart3}
          label="Patrones"
          value={status?.patterns_extracted || 0}
          sublabel="identificados"
          color="purple"
        />
        <StatCard
          icon={Calendar}
          label="Contenido"
          value={totalContent}
          sublabel="piezas generadas"
          color="green"
        />
        <StatCard
          icon={TrendingUp}
          label="Engagement Promedio"
          value={calendars.length > 0 ? `${Math.round(calendars[0]?.avg_engagement_score || 0)}%` : '-'}
          sublabel="score predicho"
          color="orange"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <QuickActionCard
          title="Generar Calendario"
          description="Crea contenido para todo el mes basado en lo que funciona"
          icon={Calendar}
          action={() => navigate('/calendar/new')}
          primary
        />
        <QuickActionCard
          title="Scanner Viral"
          description="Encuentra tendencias y genera ideas reactivas"
          icon={TrendingUp}
          action={() => navigate('/viral')}
        />
        <QuickActionCard
          title="Ver Competidores"
          description="Analiza qué está funcionando para ellos"
          icon={Users}
          action={() => navigate('/competitors')}
        />
      </div>

      {/* Competitors Status */}
      {competitors.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Estado de Competidores</h2>
            <Link to="/competitors" className="text-brand-400 hover:text-brand-300 text-sm flex items-center gap-1">
              Ver todos <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="space-y-3">
            {competitors.slice(0, 5).map((competitor) => (
              <div
                key={competitor.id}
                className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className={clsx(
                    'badge',
                    competitor.platform === 'instagram' && 'badge-instagram',
                    competitor.platform === 'tiktok' && 'badge-tiktok',
                    competitor.platform === 'linkedin' && 'badge-linkedin'
                  )}>
                    {competitor.platform === 'instagram' ? 'IG' : competitor.platform === 'tiktok' ? 'TT' : 'LI'}
                  </span>
                  <div>
                    <p className="text-white font-medium">@{competitor.handle}</p>
                    <p className="text-gray-400 text-sm">
                      {competitor.followers_count.toLocaleString()} seguidores
                    </p>
                  </div>
                </div>
                <StatusBadge status={competitor.scrape_status} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Calendars */}
      {calendars.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Calendarios Recientes</h2>
            <Link to="/calendar" className="text-brand-400 hover:text-brand-300 text-sm flex items-center gap-1">
              Ver todos <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {calendars.slice(0, 2).map((calendar) => (
              <div
                key={calendar.calendar_id}
                className="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors cursor-pointer"
                onClick={() => navigate(`/calendar`)}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-white">{calendar.name}</h3>
                  <span className={clsx(
                    'text-sm px-2 py-0.5 rounded',
                    calendar.status === 'active' && 'bg-green-500/20 text-green-400',
                    calendar.status === 'draft' && 'bg-yellow-500/20 text-yellow-400'
                  )}>
                    {calendar.status}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <span>{calendar.total_pieces} piezas</span>
                  <span>Score: {Math.round(calendar.avg_engagement_score)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="card bg-gradient-to-br from-gray-800 to-gray-800/50">
        <div className="flex items-start gap-4">
          <div className="p-2 bg-brand-500/20 rounded-lg">
            <Sparkles className="w-6 h-6 text-brand-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Tips para maximizar engagement</h3>
            <ul className="mt-2 space-y-1 text-sm text-gray-300">
              <li>• Publica Reels/TikToks entre 11:00-13:00 y 19:00-21:00</li>
              <li>• Usa hooks que generen curiosidad en los primeros 3 segundos</li>
              <li>• Incluye CTAs claros que inviten a comentar o guardar</li>
              <li>• Combina audios trending con contenido de valor</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({
  icon: Icon,
  label,
  value,
  sublabel,
  color,
}: {
  icon: React.ElementType
  label: string
  value: string | number
  sublabel: string
  color: 'blue' | 'purple' | 'green' | 'orange'
}) {
  const colors = {
    blue: 'from-blue-500/20 to-blue-600/10 text-blue-400',
    purple: 'from-purple-500/20 to-purple-600/10 text-purple-400',
    green: 'from-green-500/20 to-green-600/10 text-green-400',
    orange: 'from-orange-500/20 to-orange-600/10 text-orange-400',
  }

  return (
    <div className="card bg-gradient-to-br p-5">
      <div className={clsx('p-2 rounded-lg w-fit mb-3 bg-gradient-to-br', colors[color])}>
        <Icon className="w-5 h-5" />
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
      <p className="text-gray-400 text-sm">
        {label} <span className="text-gray-500">{sublabel}</span>
      </p>
    </div>
  )
}

function QuickActionCard({
  title,
  description,
  icon: Icon,
  action,
  primary,
}: {
  title: string
  description: string
  icon: React.ElementType
  action: () => void
  primary?: boolean
}) {
  return (
    <button
      onClick={action}
      className={clsx(
        'card-hover text-left group',
        primary && 'ring-2 ring-brand-500/50'
      )}
    >
      <div className="flex items-start gap-4">
        <div className={clsx(
          'p-3 rounded-lg transition-colors',
          primary
            ? 'bg-brand-600 group-hover:bg-brand-500'
            : 'bg-gray-700 group-hover:bg-gray-600'
        )}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="font-semibold text-white group-hover:text-brand-400 transition-colors">
            {title}
          </h3>
          <p className="text-gray-400 text-sm mt-1">{description}</p>
        </div>
      </div>
    </button>
  )
}

function StatusBadge({ status }: { status: string }) {
  const config = {
    completed: { icon: CheckCircle, color: 'text-green-400', label: 'Completado' },
    in_progress: { icon: RefreshCw, color: 'text-brand-400 animate-spin', label: 'Analizando' },
    pending: { icon: Clock, color: 'text-gray-400', label: 'Pendiente' },
    failed: { icon: AlertCircle, color: 'text-red-400', label: 'Error' },
  }[status] || { icon: Clock, color: 'text-gray-400', label: status }

  return (
    <div className="flex items-center gap-1.5">
      <config.icon className={clsx('w-4 h-4', config.color)} />
      <span className={clsx('text-sm', config.color)}>{config.label}</span>
    </div>
  )
}
