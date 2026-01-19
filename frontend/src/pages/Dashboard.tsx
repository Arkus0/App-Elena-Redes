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
  Cpu,
  Zap,
  Activity,
  AlertTriangle,
  Shield,
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { useBusinessStore } from '../stores/businessStore'
import {
  businessApi,
  competitorsApi,
  contentApi,
  mlApi,
  lightModeApi,
  type ModelHealthSummary,
  type LightModeConfig,
  type GrowthProjectionResponse,
} from '../services/api'
import { LightModeConfiguration } from '../components/LightModeConfiguration'
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
  const [mlStatus, setMlStatus] = useState<{
    is_trained: boolean
    model_version: string
    feature_count: number
  } | null>(null)
  const [modelHealth, setModelHealth] = useState<ModelHealthSummary | null>(null)
  const [lightModeConfig, setLightModeConfig] = useState<LightModeConfig | null>(null)
  const [growthProjection, setGrowthProjection] = useState<GrowthProjectionResponse | null>(null)

  useEffect(() => {
    if (!currentBusiness) {
      navigate('/onboarding')
      return
    }

    const loadData = async () => {
      setIsLoading(true)
      try {
        const [
          statusData,
          competitorsData,
          calendarsData,
          mlStatusData,
          healthData,
          lightModeData,
          growthData,
        ] = await Promise.all([
          businessApi.getStatus(currentBusiness.id),
          competitorsApi.getCompetitors(currentBusiness.id),
          contentApi.getCalendars(currentBusiness.id),
          mlApi.getModelStatus().catch(() => null),
          mlApi.getModelHealthSummary().catch(() => null),
          lightModeApi.getConfig(currentBusiness.id).catch(() => null),
          mlApi.getGrowthPrediction(currentBusiness.id).catch(() => null),
        ])
        setStatus(statusData)
        setCompetitors(competitorsData)
        setCalendars(calendarsData)
        setMlStatus(mlStatusData)
        setModelHealth(healthData)
        setLightModeConfig(lightModeData)
        setGrowthProjection(growthData)
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
          sublabel="score ML predicho"
          color="orange"
        />
      </div>

      {/* Growth Projection Chart */}
      {growthProjection && (
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-brand-400" />
                Proyección de Crecimiento
              </h2>
              <p className="text-gray-400 text-sm mt-1">
                Estimación a 30 días basada en tu calendario de contenido
              </p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-white">
                +{growthProjection.projected_total_gain.toLocaleString()}
              </p>
              <p className="text-sm text-green-400">nuevos seguidores estimados</p>
            </div>
          </div>

          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={growthProjection.projection}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                <XAxis
                  dataKey="date"
                  stroke="#9CA3AF"
                  tickFormatter={(val) =>
                    new Date(val).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
                  }
                  tick={{ fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  stroke="#9CA3AF"
                  tick={{ fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  domain={['auto', 'auto']}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    borderColor: '#374151',
                    color: '#F3F4F6',
                  }}
                  itemStyle={{ color: '#F3F4F6' }}
                  labelStyle={{ color: '#9CA3AF' }}
                  labelFormatter={(val) => new Date(val).toLocaleDateString()}
                  formatter={(value: number) => [value.toLocaleString(), 'Seguidores']}
                />
                <Line
                  type="monotone"
                  dataKey="followers"
                  stroke="#8B5CF6"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6, fill: '#8B5CF6' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ML Model Status Banner */}
      {mlStatus && (
        <div className="card bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-indigo-500/30">
          <div className="flex items-start gap-4">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <Cpu className="w-6 h-6 text-indigo-400" />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  Motor ML Híbrido
                  <span className={clsx(
                    'text-xs px-2 py-0.5 rounded-full',
                    mlStatus.is_trained
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-yellow-500/20 text-yellow-400'
                  )}>
                    {mlStatus.is_trained ? 'Activo' : 'Entrenando'}
                  </span>
                </h3>
                <span className="text-xs text-gray-400">v{mlStatus.model_version}</span>
              </div>
              <p className="text-gray-300 text-sm mt-1">
                XGBoost + RandomForest para predicciones de engagement en tiempo real.
                <span className="text-indigo-400 ml-1">{mlStatus.feature_count} features</span> analizadas por contenido.
              </p>
              <div className="flex items-center gap-4 mt-2">
                <div className="flex items-center gap-1 text-sm text-gray-400">
                  <Zap className="w-3 h-3 text-yellow-400" />
                  <span>Predicción instantánea</span>
                </div>
                <div className="flex items-center gap-1 text-sm text-gray-400">
                  <CheckCircle className="w-3 h-3 text-green-400" />
                  <span>SHAP explicaciones</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Health Section */}
      {modelHealth && modelHealth.overall_status !== 'no_data' && (
        <div className={clsx(
          'card',
          modelHealth.overall_status === 'healthy'
            ? 'bg-gradient-to-r from-emerald-600/20 to-teal-600/20 border-emerald-500/30'
            : 'bg-gradient-to-r from-amber-600/20 to-orange-600/20 border-amber-500/30'
        )}>
          <div className="flex items-start gap-4">
            <div className={clsx(
              'p-2 rounded-lg',
              modelHealth.overall_status === 'healthy'
                ? 'bg-emerald-500/20'
                : 'bg-amber-500/20'
            )}>
              {modelHealth.overall_status === 'healthy' ? (
                <Shield className="w-6 h-6 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-6 h-6 text-amber-400" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-white flex items-center gap-2">
                  Salud del Modelo
                  <span className={clsx(
                    'text-xs px-2 py-0.5 rounded-full',
                    modelHealth.overall_status === 'healthy'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : 'bg-amber-500/20 text-amber-400'
                  )}>
                    {modelHealth.overall_status === 'healthy' ? 'Estable' : 'Drift Detectado'}
                  </span>
                </h3>
                {modelHealth.last_updated && (
                  <span className="text-xs text-gray-400">
                    Actualizado: {new Date(modelHealth.last_updated).toLocaleDateString()}
                  </span>
                )}
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
                <div className="bg-gray-800/50 rounded-lg p-2">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Activity className="w-3 h-3" />
                    <span>MAE Promedio</span>
                  </div>
                  <p className="text-lg font-semibold text-white mt-1">
                    {modelHealth.avg_mae !== null ? modelHealth.avg_mae.toFixed(4) : '-'}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-2">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <TrendingUp className="w-3 h-3" />
                    <span>Drift Score</span>
                  </div>
                  <p className={clsx(
                    'text-lg font-semibold mt-1',
                    modelHealth.avg_drift_score !== null && modelHealth.avg_drift_score > 0.3
                      ? 'text-amber-400'
                      : 'text-white'
                  )}>
                    {modelHealth.avg_drift_score !== null ? modelHealth.avg_drift_score.toFixed(3) : '-'}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-2">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <CheckCircle className="w-3 h-3 text-green-400" />
                    <span>Nichos OK</span>
                  </div>
                  <p className="text-lg font-semibold text-green-400 mt-1">
                    {modelHealth.healthy_count}
                  </p>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-2">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <AlertTriangle className="w-3 h-3 text-amber-400" />
                    <span>Con Alertas</span>
                  </div>
                  <p className={clsx(
                    'text-lg font-semibold mt-1',
                    modelHealth.warning_count > 0 ? 'text-amber-400' : 'text-white'
                  )}>
                    {modelHealth.warning_count}
                  </p>
                </div>
              </div>

              {/* Alerts */}
              {modelHealth.alerts && modelHealth.alerts.length > 0 && (
                <div className="mt-3 space-y-2">
                  {modelHealth.alerts.slice(0, 3).map((alert, idx) => (
                    <div
                      key={idx}
                      className="flex items-center gap-2 text-sm bg-amber-500/10 text-amber-300 px-3 py-2 rounded-lg"
                    >
                      <AlertCircle className="w-4 h-4 flex-shrink-0" />
                      <span>{alert.message}</span>
                      {alert.score !== undefined && (
                        <span className="ml-auto text-xs bg-amber-500/20 px-2 py-0.5 rounded">
                          Score: {alert.score.toFixed(3)}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Healthy message */}
              {modelHealth.overall_status === 'healthy' && (
                <p className="text-sm text-emerald-300 mt-3 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4" />
                  Modelo estable, sin drift significativo detectado.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Light Mode Toggle - Multimodal Processing */}
      <LightModeConfiguration
        businessId={currentBusiness.id}
        compact={true}
        onConfigChange={(enabled) => {
          setLightModeConfig(prev => prev ? { ...prev, light_mode_enabled: enabled } : null)
        }}
        className="card bg-gradient-to-r from-yellow-600/10 to-orange-600/10 border-yellow-500/20"
      />

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
