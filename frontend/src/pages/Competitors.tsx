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
  Sparkles,
  MapPin,
  Target,
  Filter,
  Flame,
  Globe,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { competitorsApi } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import type { Competitor, CompetitorPreviewResponse, DiscoveredCompetitor } from '../types'
import clsx from 'clsx'

export default function Competitors() {
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)
  const competitors = useBusinessStore((state) => state.competitors)
  const setCompetitors = useBusinessStore((state) => state.setCompetitors)
  const addCompetitor = useBusinessStore((state) => state.addCompetitor)
  const removeCompetitor = useBusinessStore((state) => state.removeCompetitor)

  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'list' | 'discovery'>('list')

  // Add Modal State
  const [showAddModal, setShowAddModal] = useState(false)
  const [newPlatform, setNewPlatform] = useState('instagram')
  const [newHandle, setNewHandle] = useState('')
  const [isAdding, setIsAdding] = useState(false)
  const [searchStep, setSearchStep] = useState<'search' | 'preview'>('search')
  const [previewData, setPreviewData] = useState<CompetitorPreviewResponse | null>(null)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)

  // Discovery State
  const [discoveryForm, setDiscoveryForm] = useState({
    hashtags: '',
    location: '',
    niche: '',
    minFollowers: 100,
    maxFollowers: 100000,
    onlyActive: true
  })
  const [discoveryResults, setDiscoveryResults] = useState<DiscoveredCompetitor[]>([])
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [addingDiscovered, setAddingDiscovered] = useState<string | null>(null)

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

  // --- Handlers for "My Competitors" ---

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

  // --- Handlers for "AI Discovery" ---

  const handleDiscovery = async () => {
    if (!discoveryForm.hashtags) {
      toast.error('Debes ingresar al menos un hashtag')
      return
    }

    setIsDiscovering(true)
    try {
      const hashtagsList = discoveryForm.hashtags.split(',').map(h => h.trim().replace('#', '')).filter(Boolean)
      const locationList = discoveryForm.location ? discoveryForm.location.split(',').map(l => l.trim()).filter(Boolean) : []
      const nicheList = discoveryForm.niche ? discoveryForm.niche.split(',').map(n => n.trim()).filter(Boolean) : []

      const results = await competitorsApi.discoverCompetitors({
        hashtags: hashtagsList,
        location_keywords: locationList,
        niche_keywords: nicheList,
        min_followers: discoveryForm.minFollowers,
        max_followers: discoveryForm.maxFollowers,
        require_active: discoveryForm.onlyActive,
        max_days_since_last_post: 30,
        min_posts_last_month: 1
      })

      setDiscoveryResults(results)
      if (results.length === 0) {
        toast('No se encontraron perfiles con esos filtros.', { icon: '🔍' })
      } else {
        toast.success(`Se encontraron ${results.length} competidores potenciales`)
      }
    } catch (error: any) {
      toast.error('Error durante el descubrimiento. Intenta con otros hashtags.')
      console.error(error)
    } finally {
      setIsDiscovering(false)
    }
  }

  const handleAddDiscovered = async (handle: string, platform: string) => {
    if (!currentBusiness) return

    setAddingDiscovered(handle)
    try {
      const competitor = await competitorsApi.addCompetitor(
        currentBusiness.id,
        platform.toLowerCase(),
        handle
      )
      addCompetitor(competitor)
      toast.success(`@${handle} añadido a tus competidores`)
    } catch (error: any) {
      toast.error(`Error añadiendo @${handle}: ${error.response?.data?.detail || 'Desconocido'}`)
    } finally {
      setAddingDiscovered(null)
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
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Competidores</h1>
          <p className="text-gray-400">Analiza qué funciona para tu competencia</p>
        </div>

        <div className="flex bg-gray-800 p-1 rounded-lg self-start">
          <button
            onClick={() => setActiveTab('list')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2",
              activeTab === 'list' ? "bg-brand-600 text-white shadow-sm" : "text-gray-400 hover:text-white"
            )}
          >
            <Users className="w-4 h-4" />
            Mis Competidores
          </button>
          <button
            onClick={() => setActiveTab('discovery')}
            className={clsx(
              "px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2",
              activeTab === 'discovery' ? "bg-brand-600 text-white shadow-sm" : "text-gray-400 hover:text-white"
            )}
          >
            <Sparkles className="w-4 h-4" />
            Descubrimiento IA
          </button>
        </div>
      </div>

      {activeTab === 'list' ? (
        // === EXISTING LIST VIEW ===
        <>
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

          <div className="flex justify-end">
             <button onClick={() => setShowAddModal(true)} className="btn-primary flex items-center gap-2">
              <Plus className="w-4 h-4" />
              Añadir competidor manualmente
            </button>
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
        </>
      ) : (
        // === NEW DISCOVERY VIEW ===
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-brand-400" />
              Filtros de Búsqueda
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    Hashtags (Separados por coma) <span className="text-brand-400">*</span>
                  </label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                      type="text"
                      className="input-field pl-9"
                      placeholder="inmobiliaria, casasdelujo, realestate"
                      value={discoveryForm.hashtags}
                      onChange={e => setDiscoveryForm({...discoveryForm, hashtags: e.target.value})}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                   <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      <MapPin className="w-3 h-3 inline mr-1" />
                      Ubicación (Keywords)
                    </label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="madrid, centro"
                      value={discoveryForm.location}
                      onChange={e => setDiscoveryForm({...discoveryForm, location: e.target.value})}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      <Target className="w-3 h-3 inline mr-1" />
                      Nicho (Keywords)
                    </label>
                    <input
                      type="text"
                      className="input-field"
                      placeholder="lujo, venta"
                      value={discoveryForm.niche}
                      onChange={e => setDiscoveryForm({...discoveryForm, niche: e.target.value})}
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4 border-l border-gray-700 pl-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">
                    <Filter className="w-3 h-3 inline mr-1" />
                    Rango de Seguidores
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      className="input-field"
                      value={discoveryForm.minFollowers}
                      onChange={e => setDiscoveryForm({...discoveryForm, minFollowers: Number(e.target.value)})}
                    />
                    <span className="text-gray-500">-</span>
                    <input
                      type="number"
                      className="input-field"
                      value={discoveryForm.maxFollowers}
                      onChange={e => setDiscoveryForm({...discoveryForm, maxFollowers: Number(e.target.value)})}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between pt-4">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-white flex items-center gap-2">
                      <Clock className="w-4 h-4 text-brand-400" />
                      Solo cuentas activas
                    </span>
                    <span className="text-xs text-gray-500">Excluye cuentas sin posts en 30 días</span>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={discoveryForm.onlyActive}
                      onChange={e => setDiscoveryForm({...discoveryForm, onlyActive: e.target.checked})}
                    />
                    <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-brand-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-brand-600"></div>
                  </label>
                </div>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={handleDiscovery}
                disabled={isDiscovering}
                className="btn-primary flex items-center gap-2 px-6"
              >
                {isDiscovering ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Analizando Perfiles...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    Buscar Competidores
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Area */}
          {discoveryResults.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white">Resultados ({discoveryResults.length})</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {discoveryResults.map((result) => {
                  const isAlreadyAdded = competitors.some(c => c.handle.toLowerCase() === result.handle.toLowerCase())
                  const isAddingThis = addingDiscovered === result.handle

                  return (
                    <div key={result.handle} className="card-hover group relative">
                      <div className="flex items-start gap-4">
                        <img
                          src={result.profile_pic_url || `https://ui-avatars.com/api/?name=${result.handle}&background=random`}
                          alt={result.handle}
                          className="w-12 h-12 rounded-full object-cover border border-gray-700"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <h4 className="font-bold text-white truncate">@{result.handle}</h4>
                            <span className={clsx(
                              "text-xs px-2 py-0.5 rounded-full border",
                              result.activity_status === 'Very Active' ? "bg-green-500/10 text-green-400 border-green-500/20" :
                              result.activity_status === 'Active' ? "bg-blue-500/10 text-blue-400 border-blue-500/20" :
                              "bg-gray-700 text-gray-400 border-gray-600"
                            )}>
                              {result.activity_status}
                            </span>
                          </div>
                          <div className="flex items-center gap-3 mt-1 text-sm text-gray-400">
                            <span>{result.followers.toLocaleString()} seg.</span>
                            {result.relevance_score > 0 && (
                               <span className="flex items-center gap-1 text-brand-400">
                                 <Sparkles className="w-3 h-3" />
                                 {result.relevance_score}%
                               </span>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Match Reasons */}
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {result.match_reasons.map((reason, i) => (
                           <span key={i} className="text-[10px] bg-gray-800 text-gray-300 px-1.5 py-0.5 rounded">
                             {reason}
                           </span>
                        ))}
                      </div>

                      <div className="mt-4 flex gap-2">
                        <a
                          href={`https://instagram.com/${result.handle}`}
                          target="_blank"
                          rel="noreferrer"
                          className="btn-secondary flex-1 flex items-center justify-center gap-1 text-xs"
                        >
                          <ExternalLink className="w-3 h-3" />
                          Ver Perfil
                        </a>

                        {isAlreadyAdded ? (
                          <button disabled className="btn-secondary flex-1 flex items-center justify-center gap-1 text-xs opacity-50 cursor-not-allowed">
                            <CheckCircle className="w-3 h-3" />
                            Añadido
                          </button>
                        ) : (
                          <button
                            onClick={() => handleAddDiscovered(result.handle, result.platform)}
                            disabled={isAddingThis}
                            className="btn-primary flex-1 flex items-center justify-center gap-1 text-xs"
                          >
                            {isAddingThis ? (
                              <RefreshCw className="w-3 h-3 animate-spin" />
                            ) : (
                              <>
                                <Plus className="w-3 h-3" />
                                Añadir
                              </>
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Add Modal (Manual) */}
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
