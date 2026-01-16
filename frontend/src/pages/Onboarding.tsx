import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Building2,
  Instagram,
  Hash,
  Users,
  ArrowRight,
  ArrowLeft,
  Plus,
  X,
  Sparkles,
  CheckCircle,
} from 'lucide-react'
import toast from 'react-hot-toast'
import { businessApi } from '../services/api'
import { useBusinessStore } from '../stores/businessStore'
import clsx from 'clsx'

const BUSINESS_TYPES = [
  { value: 'inmobiliaria', label: 'Inmobiliaria', emoji: '🏠' },
  { value: 'floristeria', label: 'Floristería', emoji: '💐' },
  { value: 'cafeteria', label: 'Cafetería', emoji: '☕' },
  { value: 'peluqueria', label: 'Peluquería', emoji: '💇' },
  { value: 'tienda_local', label: 'Tienda Local', emoji: '🛍️' },
  { value: 'restaurante', label: 'Restaurante', emoji: '🍽️' },
  { value: 'gimnasio', label: 'Gimnasio', emoji: '💪' },
  { value: 'clinica', label: 'Clínica', emoji: '🏥' },
  { value: 'otros', label: 'Otros', emoji: '✨' },
]

const STEPS = [
  { title: 'Tu negocio', description: 'Cuéntanos sobre ti' },
  { title: 'Redes sociales', description: 'Tus handles' },
  { title: 'Competidores', description: 'A quién analizamos' },
  { title: 'Objetivos', description: 'Qué quieres lograr' },
]

export default function Onboarding() {
  const navigate = useNavigate()
  const setCurrentBusiness = useBusinessStore((state) => state.setCurrentBusiness)
  const [step, setStep] = useState(0)
  const [isLoading, setIsLoading] = useState(false)

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    business_type: '',
    description: '',
    location: '',
    instagram_handle: '',
    tiktok_handle: '',
    linkedin_handle: '',
    active_platforms: ['instagram', 'tiktok'] as string[],
    competitors: [] as { platform: string; handle: string }[],
    content_goals: ['engagement'] as string[],
    posting_frequency: '3-5_per_week',
    brand_voice: 'friendly_professional',
  })

  const [newCompetitor, setNewCompetitor] = useState({ platform: 'instagram', handle: '' })

  const updateFormData = (updates: Partial<typeof formData>) => {
    setFormData((prev) => ({ ...prev, ...updates }))
  }

  const addCompetitor = () => {
    if (!newCompetitor.handle) return
    if (formData.competitors.length >= 7) {
      toast.error('Máximo 7 competidores')
      return
    }
    updateFormData({
      competitors: [...formData.competitors, { ...newCompetitor, handle: newCompetitor.handle.replace('@', '') }],
    })
    setNewCompetitor({ platform: 'instagram', handle: '' })
  }

  const removeCompetitor = (index: number) => {
    updateFormData({
      competitors: formData.competitors.filter((_, i) => i !== index),
    })
  }

  const handleSubmit = async () => {
    if (formData.competitors.length < 3) {
      toast.error('Añade al menos 3 competidores para un análisis efectivo')
      return
    }

    setIsLoading(true)
    try {
      const response = await businessApi.onboard(formData)

      // Get the created business
      const businesses = await businessApi.getMyBusinesses()
      const newBusiness = businesses.find((b) => b.id === response.business_id)
      if (newBusiness) {
        setCurrentBusiness(newBusiness)
      }

      toast.success('¡Negocio configurado! Analizando competidores...')
      navigate('/')
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Error al crear negocio')
    } finally {
      setIsLoading(false)
    }
  }

  const canProceed = () => {
    switch (step) {
      case 0:
        return formData.name && formData.business_type
      case 1:
        return formData.instagram_handle || formData.tiktok_handle
      case 2:
        return formData.competitors.length >= 3
      case 3:
        return formData.content_goals.length > 0
      default:
        return true
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white">Configura tu negocio</h1>
          <p className="text-gray-400 mt-2">
            En 2 minutos tendrás acceso a contenido de alto engagement
          </p>
        </div>

        {/* Progress */}
        <div className="flex items-center justify-between mb-8">
          {STEPS.map((s, i) => (
            <div key={i} className="flex items-center">
              <div
                className={clsx(
                  'w-10 h-10 rounded-full flex items-center justify-center font-medium transition-colors',
                  i < step && 'bg-brand-600 text-white',
                  i === step && 'bg-brand-500 text-white ring-4 ring-brand-500/30',
                  i > step && 'bg-gray-700 text-gray-400'
                )}
              >
                {i < step ? <CheckCircle className="w-5 h-5" /> : i + 1}
              </div>
              {i < STEPS.length - 1 && (
                <div
                  className={clsx(
                    'w-12 md:w-24 h-1 mx-2',
                    i < step ? 'bg-brand-600' : 'bg-gray-700'
                  )}
                />
              )}
            </div>
          ))}
        </div>

        {/* Step Content */}
        <div className="card">
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-white">{STEPS[step].title}</h2>
            <p className="text-gray-400">{STEPS[step].description}</p>
          </div>

          {/* Step 0: Business Info */}
          {step === 0 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Nombre de tu negocio
                </label>
                <div className="relative">
                  <Building2 className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => updateFormData({ name: e.target.value })}
                    className="input-field pl-10"
                    placeholder="Floristería Rosa"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Tipo de negocio
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {BUSINESS_TYPES.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => updateFormData({ business_type: type.value })}
                      className={clsx(
                        'p-3 rounded-lg border text-left transition-all',
                        formData.business_type === type.value
                          ? 'border-brand-500 bg-brand-500/10'
                          : 'border-gray-700 hover:border-gray-600'
                      )}
                    >
                      <span className="text-xl">{type.emoji}</span>
                      <p className="text-sm text-white mt-1">{type.label}</p>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Ubicación (ciudad)
                </label>
                <input
                  type="text"
                  value={formData.location}
                  onChange={(e) => updateFormData({ location: e.target.value })}
                  className="input-field"
                  placeholder="Barcelona"
                />
              </div>
            </div>
          )}

          {/* Step 1: Social Media */}
          {step === 1 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Instagram (obligatorio)
                </label>
                <div className="relative">
                  <Instagram className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type="text"
                    value={formData.instagram_handle}
                    onChange={(e) => updateFormData({ instagram_handle: e.target.value.replace('@', '') })}
                    className="input-field pl-10"
                    placeholder="tu_negocio"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  TikTok (recomendado)
                </label>
                <div className="relative">
                  <Hash className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                  <input
                    type="text"
                    value={formData.tiktok_handle}
                    onChange={(e) => updateFormData({ tiktok_handle: e.target.value.replace('@', '') })}
                    className="input-field pl-10"
                    placeholder="tu_negocio"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  LinkedIn (opcional)
                </label>
                <input
                  type="text"
                  value={formData.linkedin_handle}
                  onChange={(e) => updateFormData({ linkedin_handle: e.target.value })}
                  className="input-field"
                  placeholder="perfil-linkedin"
                />
              </div>
            </div>
          )}

          {/* Step 2: Competitors */}
          {step === 2 && (
            <div className="space-y-6">
              <div className="bg-gray-700/50 rounded-lg p-4">
                <p className="text-sm text-gray-300">
                  <strong>Añade 3-7 competidores</strong> de tu nicho. Pueden ser locales o de referencia.
                  Analizaremos su contenido más exitoso para generar el tuyo.
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Añadir competidor
                </label>
                <div className="flex gap-2">
                  <select
                    value={newCompetitor.platform}
                    onChange={(e) => setNewCompetitor({ ...newCompetitor, platform: e.target.value })}
                    className="input-field w-32"
                  >
                    <option value="instagram">Instagram</option>
                    <option value="tiktok">TikTok</option>
                    <option value="linkedin">LinkedIn</option>
                  </select>
                  <div className="relative flex-1">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">@</span>
                    <input
                      type="text"
                      value={newCompetitor.handle}
                      onChange={(e) => setNewCompetitor({ ...newCompetitor, handle: e.target.value.replace('@', '') })}
                      onKeyPress={(e) => e.key === 'Enter' && addCompetitor()}
                      className="input-field pl-8"
                      placeholder="competidor_ejemplo"
                    />
                  </div>
                  <button onClick={addCompetitor} className="btn-primary px-3">
                    <Plus className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {formData.competitors.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm text-gray-400">
                    {formData.competitors.length}/7 competidores ({formData.competitors.length < 3 ? 'mínimo 3' : '✓'})
                  </p>
                  {formData.competitors.map((comp, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <span className={clsx(
                          'badge',
                          comp.platform === 'instagram' && 'badge-instagram',
                          comp.platform === 'tiktok' && 'badge-tiktok',
                          comp.platform === 'linkedin' && 'badge-linkedin'
                        )}>
                          {comp.platform === 'instagram' ? 'IG' : comp.platform === 'tiktok' ? 'TT' : 'LI'}
                        </span>
                        <span className="text-white">@{comp.handle}</span>
                      </div>
                      <button
                        onClick={() => removeCompetitor(i)}
                        className="p-1 text-gray-400 hover:text-red-400"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Suggested competitors by business type */}
              {formData.business_type && (
                <div className="border-t border-gray-700 pt-4">
                  <p className="text-sm text-gray-400 mb-2">Sugerencias para {formData.business_type}:</p>
                  <div className="flex flex-wrap gap-2">
                    {getSuggestedCompetitors(formData.business_type).map((handle, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          if (!formData.competitors.find((c) => c.handle === handle)) {
                            updateFormData({
                              competitors: [...formData.competitors, { platform: 'instagram', handle }],
                            })
                          }
                        }}
                        className="text-sm px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-full text-gray-300"
                      >
                        @{handle}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Step 3: Goals */}
          {step === 3 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  ¿Qué quieres lograr con tu contenido?
                </label>
                <div className="space-y-2">
                  {[
                    { value: 'engagement', label: 'Más engagement (likes, comentarios, guardados)', emoji: '💬' },
                    { value: 'awareness', label: 'Más visibilidad y alcance', emoji: '👀' },
                    { value: 'leads', label: 'Generar leads / mensajes directos', emoji: '📩' },
                    { value: 'foot_traffic', label: 'Más visitas a la tienda física', emoji: '🚶' },
                    { value: 'sales', label: 'Aumentar ventas', emoji: '💰' },
                  ].map((goal) => (
                    <button
                      key={goal.value}
                      onClick={() => {
                        const goals = formData.content_goals.includes(goal.value)
                          ? formData.content_goals.filter((g) => g !== goal.value)
                          : [...formData.content_goals, goal.value]
                        updateFormData({ content_goals: goals })
                      }}
                      className={clsx(
                        'w-full p-3 rounded-lg border text-left transition-all flex items-center gap-3',
                        formData.content_goals.includes(goal.value)
                          ? 'border-brand-500 bg-brand-500/10'
                          : 'border-gray-700 hover:border-gray-600'
                      )}
                    >
                      <span className="text-xl">{goal.emoji}</span>
                      <span className="text-white">{goal.label}</span>
                      {formData.content_goals.includes(goal.value) && (
                        <CheckCircle className="w-5 h-5 text-brand-400 ml-auto" />
                      )}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Frecuencia de publicación deseada
                </label>
                <select
                  value={formData.posting_frequency}
                  onChange={(e) => updateFormData({ posting_frequency: e.target.value })}
                  className="input-field"
                >
                  <option value="1-2_per_week">1-2 veces por semana</option>
                  <option value="3-5_per_week">3-5 veces por semana (recomendado)</option>
                  <option value="daily">Diario</option>
                  <option value="multiple_daily">Varias veces al día</option>
                </select>
              </div>
            </div>
          )}

          {/* Navigation */}
          <div className="flex justify-between mt-8 pt-6 border-t border-gray-700">
            {step > 0 ? (
              <button onClick={() => setStep(step - 1)} className="btn-secondary flex items-center gap-2">
                <ArrowLeft className="w-4 h-4" />
                Anterior
              </button>
            ) : (
              <div />
            )}

            {step < STEPS.length - 1 ? (
              <button
                onClick={() => setStep(step + 1)}
                disabled={!canProceed()}
                className="btn-primary flex items-center gap-2"
              >
                Siguiente
                <ArrowRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={!canProceed() || isLoading}
                className="btn-primary flex items-center gap-2"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Comenzar análisis
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function getSuggestedCompetitors(businessType: string): string[] {
  const suggestions: Record<string, string[]> = {
    floristeria: ['floristeria_ejemplo', 'flores_bcn', 'ramos_madrid', 'floristeria_premium'],
    inmobiliaria: ['inmobiliaria_ejemplo', 'pisos_bcn', 'casas_madrid', 'real_estate_spain'],
    cafeteria: ['cafe_ejemplo', 'coffee_barcelona', 'cafeteria_madrid', 'specialty_coffee'],
    peluqueria: ['peluqueria_ejemplo', 'hair_bcn', 'salon_madrid', 'hairstylist_pro'],
    tienda_local: ['tienda_ejemplo', 'shop_bcn', 'retail_madrid', 'local_store'],
    restaurante: ['restaurante_ejemplo', 'food_bcn', 'comida_madrid', 'gastro_spain'],
    gimnasio: ['gym_ejemplo', 'fitness_bcn', 'crossfit_madrid', 'entrenador_personal'],
    clinica: ['clinica_ejemplo', 'dental_bcn', 'estetica_madrid', 'salud_spain'],
  }
  return suggestions[businessType] || ['ejemplo_1', 'ejemplo_2', 'ejemplo_3']
}
