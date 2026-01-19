import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Users,
  Plus,
  RefreshCw,
  CheckCircle,
  Clock,
  AlertCircle,
  Eye,
  Trash2,
  Instagram,
  ExternalLink,
  Search,
  ArrowLeft,
  ShieldCheck,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { competitorsApi } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import type { Competitor, CompetitorPreviewResponse } from '../types'
import clsx from 'clsx'

export default function Competitors() {
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const competitors = useBusinessStore((state) => state.competitors)
  const setCompetitors = useBusinessStore((state) => state.setCompetitors)
  const addCompetitor = useBusinessStore((state) => state.addCompetitor)
  const removeCompetitor = useBusinessStore((state) => state.removeCompetitor)

  const [isLoading, setIsLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)
  const [newPlatform, setNewPlatform] = useState('instagram')
  const [newHandle, setNewHandle] = useState('')
  const [isAdding, setIsAdding] = useState(false)

  // Search & Verify State
  const [searchStep, setSearchStep] = useState<'search' | 'preview'>('search')
  const [previewData, setPreviewData] = useState<CompetitorPreviewResponse | null>(null)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)

  useEffect(() => {
    if (!currentBusiness) return

    const loadCompetitors = async () => {
      try {
        const data = await competitorsApi.getCompetitors(currentBusiness.id)
        setCompetitors(data)
      } catch (error) {
        toast.error('Error cargando competidores')
      } finally {
        setIsLoading(false)
      }
    }

    loadCompetitors()
  }, [currentBusiness, setCompetitors])

  // Reset modal state
  useEffect(() => {
    if (!showAddModal) {
      setSearchStep('search')
      setPreviewData(null)
      setNewHandle('')
      setNewPlatform('instagram')
    }
  }, [showAddModal])

  const handleSearch = async () => {
    if (!newHandle) return
    setIsPreviewLoading(true)
    try {
      const data = await competitorsApi.previewCompetitor(newPlatform, newHandle.replace('@', ''))
      setPreviewData(data)
      setSearchStep('preview')
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Usuario no encontrado o privado')
    } finally {
      setIsPreviewLoading(false)
    }
  }

  const handleConfirmAdd = async () => {
    if (!currentBusiness || !newHandle) return

    setIsAdding(true)
    try {
      const competitor = await competitorsApi.addCompetitor(
        currentBusiness.id,
        newPlatform,
        newHandle.replace('@', '')
      )
      addCompetitor(competitor)
      setShowAddModal(false)
      toast.success('Competidor añadido. Analizando...')
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Error añadiendo competidor')
    } finally {
      setIsAdding(false)
    }
  }

  const handleDeleteCompetitor = async (competitorId: number) => {
    if (!currentBusiness) return
    if (!confirm('¿Eliminar este competidor?')) return

    try {
      await competitorsApi.deleteCompetitor(currentBusiness.id, competitorId)
      removeCompetitor(competitorId)
      toast.success('Competidor eliminado')
    } catch (error) {
      toast.error('Error eliminando competidor')
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-brand-500 animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Competidores</h1>
          <p className="text-gray-400">Analiza qué funciona para tu competencia</p>
        </div>
        <button onClick={() => setShowAddModal(true)} className="btn-primary flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Añadir competidor
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <p className="text-3xl font-bold text-white">{competitors.length}</p>
          <p className="text-gray-400 text-sm">Total</p>
        </div>
        <div className="card text-center">
          <p className="text-3xl font-bold text-green-400">
            {competitors.filter((c) => c.scrape_status === 'completed').length}
          </p>
          <p className="text-gray-400 text-sm">Analizados</p>
        </div>
        <div className="card text-center">
          <p className="text-3xl font-bold text-brand-400">
            {competitors.filter((c) => c.scrape_status === 'in_progress').length}
          </p>
          <p className="text-gray-400 text-sm">En progreso</p>
        </div>
        <div className="card text-center">
          <p className="text-3xl font-bold text-white">
            {competitors.reduce((sum, c) => sum + c.followers_count, 0).toLocaleString()}
          </p>
          <p className="text-gray-400 text-sm">Seguidores totales</p>
        </div>
      </div>

      {/* Competitors List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {competitors.map((competitor) => (
          <CompetitorCard
            key={competitor.id}
            competitor={competitor}
            onAnalyze={() => navigate(`/competitors/${competitor.id}`)}
            onDelete={() => handleDeleteCompetitor(competitor.id)}
          />
        ))}
      </div>

      {competitors.length === 0 && (
        <div className="card text-center py-12">
          <Users className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No hay competidores</h3>
          <p className="text-gray-400 mb-4">Añade competidores para analizar su contenido</p>
          <button onClick={() => setShowAddModal(true)} className="btn-primary">
            Añadir competidor
          </button>
        </div>
      )}

      {/* Add Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="card max-w-md w-full">
            {/* Header */}
            <div className="flex items-center gap-2 mb-6">
              {searchStep === 'preview' && (
                <button
                  onClick={() => setSearchStep('search')}
                  className="p-1 -ml-2 text-gray-400 hover:text-white"
                >
                  <ArrowLeft className="w-5 h-5" />
                </button>
              )}
              <h3 className="text-lg font-semibold text-white">
                {searchStep === 'search' ? 'Buscar Competidor' : 'Verificar Perfil'}
              </h3>
            </div>

            {searchStep === 'search' ? (
              // Step 1: Search Form
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Plataforma</label>
                  <select
                    value={newPlatform}
                    onChange={(e) => setNewPlatform(e.target.value)}
                    className="input-field"
                  >
                    <option value="instagram">Instagram</option>
                    <option value="tiktok">TikTok</option>
                    <option value="linkedin">LinkedIn</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Usuario (Handle)</label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">@</span>
                    <input
                      type="text"
                      value={newHandle}
                      onChange={(e) => setNewHandle(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                      className="input-field pl-8"
                      placeholder={newPlatform === 'linkedin' ? 'company-name' : 'usuario'}
                      autoFocus
                    />
                  </div>
                </div>

                <div className="flex justify-end gap-3 mt-6 pt-2">
                  <button onClick={() => setShowAddModal(false)} className="btn-secondary">
                    Cancelar
                  </button>
                  <button
                    onClick={handleSearch}
                    disabled={!newHandle || isPreviewLoading}
                    className="btn-primary flex items-center gap-2"
                  >
                    {isPreviewLoading ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <>
                        <Search className="w-4 h-4" />
                        Buscar
                      </>
                    )}
                  </button>
                </div>
              </div>
            ) : (
              // Step 2: Preview Card
              <div className="space-y-6">
                {previewData && (
                  <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <div className="flex items-start gap-4">
                      {/* Avatar */}
                      <div className="relative">
                        <img
                          src={previewData.profile_pic_url || `https://ui-avatars.com/api/?name=${previewData.handle}`}
                          alt={previewData.handle}
                          className="w-16 h-16 rounded-full object-cover border-2 border-brand-500"
                        />
                        <div className={clsx(
                          "absolute -bottom-1 -right-1 p-1 rounded-full",
                          previewData.platform === 'instagram' ? 'bg-pink-500' :
                          previewData.platform === 'tiktok' ? 'bg-black' : 'bg-blue-600'
                        )}>
                          {previewData.platform === 'instagram' ? <Instagram className="w-3 h-3 text-white" /> :
                           <span className="text-[10px] font-bold text-white px-0.5">T</span>}
                        </div>
                      </div>

                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h4 className="font-bold text-white text-lg truncate">{previewData.handle}</h4>
                          {previewData.verified && (
                            <ShieldCheck className="w-4 h-4 text-blue-400" />
                          )}
                        </div>
                        <p className="text-gray-400 text-sm truncate">{previewData.full_name}</p>

                        <div className="flex items-center gap-4 mt-2">
                          <div>
                            <span className="font-bold text-white">{previewData.followers_count.toLocaleString()}</span>
                            <span className="text-gray-500 text-xs ml-1">seguidores</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Bio */}
                    {previewData.biography && (
                      <div className="mt-4 p-3 bg-gray-900/50 rounded text-sm text-gray-300 border-l-2 border-brand-500 italic">
                        "{previewData.biography.length > 100
                          ? previewData.biography.substring(0, 100) + '...'
                          : previewData.biography}"
                      </div>
                    )}

                    {/* Warnings */}
                    {previewData.is_private && (
                      <div className="mt-3 flex items-center gap-2 text-yellow-400 text-sm bg-yellow-400/10 p-2 rounded">
                        <AlertCircle className="w-4 h-4" />
                        <span>Cuenta privada. El análisis será limitado.</span>
                      </div>
                    )}
                  </div>
                )}

                <div className="flex justify-end gap-3 mt-6">
                  <button onClick={() => setSearchStep('search')} className="btn-secondary">
                    Atrás
                  </button>
                  <button
                    onClick={handleConfirmAdd}
                    disabled={isAdding}
                    className="btn-primary flex items-center gap-2"
                  >
                    {isAdding ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <>
                        <CheckCircle className="w-4 h-4" />
                        Confirmar y Analizar
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function CompetitorCard({
  competitor,
  onAnalyze,
  onDelete,
}: {
  competitor: Competitor
  onAnalyze: () => void
  onDelete: () => void
}) {
  const statusConfig = {
    completed: { icon: CheckCircle, color: 'text-green-400', label: 'Analizado' },
    in_progress: { icon: RefreshCw, color: 'text-brand-400', label: 'Analizando', spin: true },
    pending: { icon: Clock, color: 'text-gray-400', label: 'Pendiente' },
    failed: { icon: AlertCircle, color: 'text-red-400', label: 'Error' },
  }[competitor.scrape_status] || { icon: Clock, color: 'text-gray-400', label: competitor.scrape_status }

  return (
    <div className="card-hover group">
      <div className="flex items-start justify-between mb-4">
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
            <p className="font-medium text-white">@{competitor.handle}</p>
            {competitor.display_name && (
              <p className="text-sm text-gray-400">{competitor.display_name}</p>
            )}
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
          className="p-1 text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-2xl font-bold text-white">
            {competitor.followers_count.toLocaleString()}
          </p>
          <p className="text-xs text-gray-400">Seguidores</p>
        </div>
        <div>
          <p className="text-2xl font-bold text-white">{competitor.posts_count}</p>
          <p className="text-xs text-gray-400">Posts</p>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <statusConfig.icon
            className={clsx('w-4 h-4', statusConfig.color, statusConfig.spin && 'animate-spin')}
          />
          <span className={clsx('text-sm', statusConfig.color)}>{statusConfig.label}</span>
        </div>

        {competitor.scrape_status === 'completed' && (
          <button
            onClick={onAnalyze}
            className="flex items-center gap-1 text-sm text-brand-400 hover:text-brand-300"
          >
            <Eye className="w-4 h-4" />
            Ver análisis
          </button>
        )}
      </div>

      {competitor.common_hashtags && competitor.common_hashtags.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="flex flex-wrap gap-1">
            {competitor.common_hashtags.slice(0, 5).map((tag, i) => (
              <span key={i} className="text-xs text-gray-400">
                #{tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
