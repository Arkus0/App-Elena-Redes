import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Calendar,
  Plus,
  Download,
  Filter,
  Grid3X3,
  List,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { contentApi } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import type { ContentCalendar as ContentCalendarType, ContentPiece } from '../types'
import clsx from 'clsx'
import toast from 'react-hot-toast'

export default function ContentCalendar() {
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const calendars = useBusinessStore((state) => state.calendars)
  const setCalendars = useBusinessStore((state) => state.setCalendars)

  const [selectedCalendar, setSelectedCalendar] = useState<ContentCalendarType | null>(null)
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [isLoading, setIsLoading] = useState(true)
  const [isExporting, setIsExporting] = useState(false)
  const [feedbackContentId, setFeedbackContentId] = useState<number | null>(null)

  useEffect(() => {
    if (!currentBusiness) return

    const loadCalendars = async () => {
      try {
        const data = await contentApi.getCalendars(currentBusiness.id)
        setCalendars(data)
        if (data.length > 0) {
          setSelectedCalendar(data[0])
        }
      } catch (error) {
        console.error('Error loading calendars:', error)
      } finally {
        setIsLoading(false)
      }
    }

    loadCalendars()
  }, [currentBusiness, setCalendars])

  const handleExport = async (format: 'csv' | 'json') => {
    if (!currentBusiness || !selectedCalendar) return

    setIsExporting(true)
    try {
      const data = await contentApi.exportContent(currentBusiness.id, format, {
        include_scripts: true,
        include_filming_guides: true,
        content_ids: selectedCalendar.content_pieces.map((p) => p.id),
      })

      // Create download
      const blob = format === 'csv' ? data : new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `calendario_${selectedCalendar.month}_${selectedCalendar.year}.${format}`
      a.click()
      window.URL.revokeObjectURL(url)

      toast.success(`Exportado a ${format.toUpperCase()}`)
    } catch (error) {
      toast.error('Error al exportar')
    } finally {
      setIsExporting(false)
    }
  }

  const handleFeedback = async (performance: 'viral' | 'good' | 'flop') => {
    if (!currentBusiness || !feedbackContentId || !selectedCalendar) return

    try {
      await contentApi.submitFeedback(currentBusiness.id, feedbackContentId, performance)
      toast.success('Feedback registrado. ¡Aprendiendo!')

      // Update local state
      const updatedPieces = selectedCalendar.content_pieces.map((p) =>
        p.id === feedbackContentId ? { ...p, performance_label: performance } : p
      )
      setSelectedCalendar({ ...selectedCalendar, content_pieces: updatedPieces })

      setFeedbackContentId(null)
    } catch (error) {
      toast.error('Error al registrar feedback')
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Calendario de Contenido</h1>
          <p className="text-gray-400">Tu contenido de alto engagement organizado</p>
        </div>
        <button onClick={() => navigate('/calendar/new')} className="btn-primary flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Generar nuevo
        </button>
      </div>

      {calendars.length === 0 ? (
        <div className="card text-center py-12">
          <Calendar className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No hay calendarios</h3>
          <p className="text-gray-400 mb-4">Genera tu primer calendario de contenido</p>
          <button onClick={() => navigate('/calendar/new')} className="btn-primary">
            Generar calendario
          </button>
        </div>
      ) : (
        <>
          {/* Calendar Selector */}
          <div className="flex items-center gap-4 overflow-x-auto pb-2">
            {calendars.map((cal) => (
              <button
                key={cal.calendar_id}
                onClick={() => setSelectedCalendar(cal)}
                className={clsx(
                  'px-4 py-2 rounded-lg whitespace-nowrap transition-colors',
                  selectedCalendar?.calendar_id === cal.calendar_id
                    ? 'bg-brand-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                )}
              >
                {cal.name}
              </button>
            ))}
          </div>

          {selectedCalendar && (
            <>
              {/* Calendar Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card text-center">
                  <p className="text-3xl font-bold text-white">{selectedCalendar.total_pieces}</p>
                  <p className="text-sm text-gray-400">Piezas totales</p>
                </div>
                <div className="card text-center">
                  <p className="text-3xl font-bold text-green-400">
                    {Math.round(selectedCalendar.avg_engagement_score)}
                  </p>
                  <p className="text-sm text-gray-400">Score promedio</p>
                </div>
                <div className="card text-center">
                  <p className="text-3xl font-bold text-brand-400">
                    {Object.keys(selectedCalendar.content_by_platform).length}
                  </p>
                  <p className="text-sm text-gray-400">Plataformas</p>
                </div>
                <div className="card text-center">
                  <p className="text-3xl font-bold text-purple-400">
                    {Object.keys(selectedCalendar.content_by_format).length}
                  </p>
                  <p className="text-sm text-gray-400">Formatos</p>
                </div>
              </div>

              {/* Actions Bar */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={clsx(
                      'p-2 rounded-lg',
                      viewMode === 'grid' ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400'
                    )}
                  >
                    <Grid3X3 className="w-5 h-5" />
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    className={clsx(
                      'p-2 rounded-lg',
                      viewMode === 'list' ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400'
                    )}
                  >
                    <List className="w-5 h-5" />
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleExport('csv')}
                    disabled={isExporting}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    CSV
                  </button>
                  <button
                    onClick={() => handleExport('json')}
                    disabled={isExporting}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    JSON
                  </button>
                </div>
              </div>

              {/* Insights */}
              {selectedCalendar.calendar_insights.length > 0 && (
                <div className="card bg-gradient-to-r from-brand-600/10 to-purple-600/10 border-brand-500/30">
                  <h3 className="font-medium text-white mb-2">Insights del calendario</h3>
                  <ul className="space-y-1">
                    {selectedCalendar.calendar_insights.map((insight, i) => (
                      <li key={i} className="text-sm text-gray-300">• {insight}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Content Grid/List */}
              {viewMode === 'grid' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {selectedCalendar.content_pieces.map((piece) => (
                    <ContentCard
                      key={piece.id}
                      piece={piece}
                      onClick={() => navigate(`/content/${piece.id}`)}
                      onFeedbackClick={(e) => {
                        e.stopPropagation()
                        setFeedbackContentId(piece.id)
                      }}
                    />
                  ))}
                </div>
              ) : (
                <div className="space-y-2">
                  {selectedCalendar.content_pieces.map((piece) => (
                    <ContentListItem
                      key={piece.id}
                      piece={piece}
                      onClick={() => navigate(`/content/${piece.id}`)}
                      onFeedbackClick={(e) => {
                        e.stopPropagation()
                        setFeedbackContentId(piece.id)
                      }}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* Feedback Modal */}
      {feedbackContentId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 w-full max-w-md shadow-2xl">
            <h3 className="text-xl font-bold text-white mb-2">¿Cómo funcionó este post?</h3>
            <p className="text-gray-400 mb-6">Tu feedback entrena al algoritmo para mejorar futuras predicciones.</p>

            <div className="grid grid-cols-3 gap-4 mb-6">
              <button
                onClick={() => handleFeedback('viral')}
                className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 transition-all group"
              >
                <span className="text-2xl group-hover:scale-110 transition-transform">🚀</span>
                <span className="font-medium text-purple-400">Viral</span>
              </button>
              <button
                onClick={() => handleFeedback('good')}
                className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl bg-green-500/10 hover:bg-green-500/20 border border-green-500/30 transition-all group"
              >
                <span className="text-2xl group-hover:scale-110 transition-transform">👍</span>
                <span className="font-medium text-green-400">Bueno</span>
              </button>
              <button
                onClick={() => handleFeedback('flop')}
                className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 transition-all group"
              >
                <span className="text-2xl group-hover:scale-110 transition-transform">📉</span>
                <span className="font-medium text-red-400">Flop</span>
              </button>
            </div>

            <button
              onClick={() => setFeedbackContentId(null)}
              className="w-full py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancelar
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function ContentCard({
  piece,
  onClick,
  onFeedbackClick
}: {
  piece: ContentPiece;
  onClick: () => void;
  onFeedbackClick: (e: React.MouseEvent) => void;
}) {
  return (
    <div onClick={onClick} className="card-hover cursor-pointer relative group">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={clsx(
            'badge',
            piece.platform === 'instagram' && 'badge-instagram',
            piece.platform === 'tiktok' && 'badge-tiktok',
            piece.platform === 'linkedin' && 'badge-linkedin'
          )}>
            {piece.platform === 'instagram' ? 'IG' : piece.platform === 'tiktok' ? 'TT' : 'LI'}
          </span>
          <span className="text-xs text-gray-400 bg-gray-700 px-2 py-0.5 rounded">
            {piece.content_format}
          </span>
        </div>
        <EngagementBadge score={piece.engagement_score} />
      </div>

      <h3 className="font-medium text-white mb-2 line-clamp-1">{piece.title}</h3>

      {piece.hook_text && (
        <p className="text-sm text-brand-400 mb-2 line-clamp-1">"{piece.hook_text}"</p>
      )}

      <p className="text-sm text-gray-400 line-clamp-2">{piece.caption}</p>

      {piece.scheduled_date && (
        <div className="mt-3 pt-3 border-t border-gray-700 flex items-center justify-between">
          <span className="text-xs text-gray-500">
            {new Date(piece.scheduled_date).toLocaleDateString('es-ES', {
              day: 'numeric',
              month: 'short',
            })}
          </span>

          {piece.performance_label ? (
            <span className={clsx(
              "text-xs px-2 py-0.5 rounded-full font-medium border",
              piece.performance_label === 'viral' && "bg-purple-500/10 text-purple-400 border-purple-500/30",
              piece.performance_label === 'good' && "bg-green-500/10 text-green-400 border-green-500/30",
              piece.performance_label === 'flop' && "bg-red-500/10 text-red-400 border-red-500/30"
            )}>
              {piece.performance_label === 'viral' ? '🚀 Viral' : piece.performance_label === 'good' ? '👍 Bueno' : '📉 Flop'}
            </span>
          ) : (
            <button
              onClick={onFeedbackClick}
              className="text-xs text-brand-400 hover:text-brand-300 font-medium opacity-0 group-hover:opacity-100 transition-opacity"
            >
              Log Results
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function ContentListItem({
  piece,
  onClick,
  onFeedbackClick
}: {
  piece: ContentPiece;
  onClick: () => void;
  onFeedbackClick: (e: React.MouseEvent) => void;
}) {
  return (
    <div
      onClick={onClick}
      className="card-hover cursor-pointer flex items-center gap-4 p-4 group"
    >
      <EngagementBadge score={piece.engagement_score} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className={clsx(
            'badge',
            piece.platform === 'instagram' && 'badge-instagram',
            piece.platform === 'tiktok' && 'badge-tiktok',
            piece.platform === 'linkedin' && 'badge-linkedin'
          )}>
            {piece.platform === 'instagram' ? 'IG' : piece.platform === 'tiktok' ? 'TT' : 'LI'}
          </span>
          <span className="text-xs text-gray-400">{piece.content_format}</span>
        </div>
        <h3 className="font-medium text-white truncate">{piece.title}</h3>
        {piece.hook_text && (
          <p className="text-sm text-gray-400 truncate">"{piece.hook_text}"</p>
        )}
      </div>

      {piece.scheduled_date && (
        <div className="text-right flex flex-col items-end gap-1">
          <p className="text-sm text-white">
            {new Date(piece.scheduled_date).toLocaleDateString('es-ES', {
              day: 'numeric',
              month: 'short',
            })}
          </p>

          {piece.performance_label ? (
            <span className={clsx(
              "text-xs px-2 py-0.5 rounded-full font-medium border",
              piece.performance_label === 'viral' && "bg-purple-500/10 text-purple-400 border-purple-500/30",
              piece.performance_label === 'good' && "bg-green-500/10 text-green-400 border-green-500/30",
              piece.performance_label === 'flop' && "bg-red-500/10 text-red-400 border-red-500/30"
            )}>
              {piece.performance_label === 'viral' ? '🚀 Viral' : piece.performance_label === 'good' ? '👍 Bueno' : '📉 Flop'}
            </span>
          ) : (
            <button
              onClick={onFeedbackClick}
              className="text-xs text-brand-400 hover:text-brand-300 font-medium opacity-0 group-hover:opacity-100 transition-opacity"
            >
              Log Results
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function EngagementBadge({ score }: { score: number }) {
  return (
    <div
      className={clsx(
        'w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm',
        score >= 80 && 'bg-green-500/20 text-green-400 ring-2 ring-green-500/30',
        score >= 50 && score < 80 && 'bg-yellow-500/20 text-yellow-400 ring-2 ring-yellow-500/30',
        score < 50 && 'bg-red-500/20 text-red-400 ring-2 ring-red-500/30'
      )}
    >
      {Math.round(score)}
    </div>
  )
}
