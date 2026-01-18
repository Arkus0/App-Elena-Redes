/**
 * MyProfileSettings - Human-in-the-Loop Configuration
 *
 * Allows the user to configure their own Instagram/TikTok username
 * so the extension can detect when content belongs to their own account
 * and trigger the ML feedback loop for continuous learning.
 */
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  User,
  Instagram,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Info,
  Brain,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { useBusinessStore } from '../stores/businessStore'
import { businessApi } from '../services/api'
import clsx from 'clsx'

interface OwnProfileConfig {
  own_instagram_username: string | null
  own_tiktok_username: string | null
  feedback_loop_enabled: boolean
  message: string
}

export default function MyProfileSettings() {
  const navigate = useNavigate()
  const currentBusiness = useBusinessStore((state) => state.currentBusiness)

  const [config, setConfig] = useState<OwnProfileConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  // Form state
  const [instagramUsername, setInstagramUsername] = useState('')
  const [tiktokUsername, setTiktokUsername] = useState('')

  useEffect(() => {
    if (!currentBusiness) {
      navigate('/onboarding')
      return
    }

    loadConfig()
  }, [currentBusiness, navigate])

  const loadConfig = async () => {
    if (!currentBusiness) return

    setIsLoading(true)
    try {
      const data = await businessApi.getOwnProfileConfig(currentBusiness.id)
      setConfig(data)
      setInstagramUsername(data.own_instagram_username || '')
      setTiktokUsername(data.own_tiktok_username || '')
    } catch (error) {
      console.error('Error loading config:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSave = async () => {
    if (!currentBusiness) return

    setIsSaving(true)
    setSaveMessage(null)

    try {
      const data = await businessApi.updateOwnProfileConfig(currentBusiness.id, {
        own_instagram_username: instagramUsername.trim() || null,
        own_tiktok_username: tiktokUsername.trim() || null,
      })
      setConfig(data)
      setSaveMessage({ type: 'success', text: data.message })
    } catch (error) {
      console.error('Error saving config:', error)
      setSaveMessage({ type: 'error', text: 'Error al guardar la configuración' })
    } finally {
      setIsSaving(false)
    }
  }

  // Clean username input (remove @ if present)
  const cleanUsername = (value: string) => value.replace(/^@/, '').trim()

  if (!currentBusiness) {
    return null
  }

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <User className="w-7 h-7 text-brand-400" />
          Mi Perfil IG/TikTok
        </h1>
        <p className="text-gray-400 mt-1">
          Configura tu username para activar el feedback loop de ML
        </p>
      </div>

      {/* Explanation Card */}
      <div className="card bg-gradient-to-r from-brand-600/20 to-purple-600/20 border-brand-500/30">
        <div className="flex items-start gap-4">
          <div className="p-2 bg-brand-500/20 rounded-lg">
            <Brain className="w-6 h-6 text-brand-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white">Human-in-the-Loop Learning</h3>
            <p className="text-gray-300 text-sm mt-1">
              Cuando configuras tu username, la extensión Elena Bridge detecta automáticamente
              cuando visitas <strong>tu propio perfil o posts</strong>. Las métricas reales
              (likes, comentarios, guardados, views) se envían al modelo ML para que
              <strong> aprenda de tu rendimiento real</strong>.
            </p>
            <div className="flex items-center gap-4 mt-3">
              <div className="flex items-center gap-1 text-sm text-gray-400">
                <Zap className="w-3 h-3 text-yellow-400" />
                <span>Cierre automático del loop</span>
              </div>
              <div className="flex items-center gap-1 text-sm text-gray-400">
                <TrendingUp className="w-3 h-3 text-green-400" />
                <span>Modelo mejora continuamente</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Status Banner */}
      {config && (
        <div
          className={clsx(
            'card',
            config.feedback_loop_enabled
              ? 'bg-green-500/10 border-green-500/30'
              : 'bg-yellow-500/10 border-yellow-500/30'
          )}
        >
          <div className="flex items-center gap-3">
            {config.feedback_loop_enabled ? (
              <>
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-green-400 font-medium">
                  Feedback Loop Activo
                </span>
              </>
            ) : (
              <>
                <AlertCircle className="w-5 h-5 text-yellow-400" />
                <span className="text-yellow-400 font-medium">
                  Feedback Loop Desactivado
                </span>
              </>
            )}
          </div>
          <p className="text-gray-400 text-sm mt-2">{config.message}</p>
        </div>
      )}

      {/* Configuration Form */}
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-6">
          Configurar Usernames
        </h2>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-6 h-6 text-brand-400 animate-spin" />
          </div>
        ) : (
          <div className="space-y-6">
            {/* Instagram Username */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                <div className="flex items-center gap-2">
                  <Instagram className="w-4 h-4 text-pink-400" />
                  Username de Instagram
                </div>
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-3 flex items-center text-gray-500">
                  @
                </span>
                <input
                  type="text"
                  value={instagramUsername}
                  onChange={(e) => setInstagramUsername(cleanUsername(e.target.value))}
                  placeholder="inmoalmeria"
                  className="w-full pl-8 pr-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Tu cuenta de Instagram desde donde publicas contenido
              </p>
            </div>

            {/* TikTok Username */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-cyan-400" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z" />
                  </svg>
                  Username de TikTok
                </div>
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-3 flex items-center text-gray-500">
                  @
                </span>
                <input
                  type="text"
                  value={tiktokUsername}
                  onChange={(e) => setTiktokUsername(cleanUsername(e.target.value))}
                  placeholder="inmoalmeria"
                  className="w-full pl-8 pr-4 py-2 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Tu cuenta de TikTok desde donde publicas contenido
              </p>
            </div>

            {/* Save Message */}
            {saveMessage && (
              <div
                className={clsx(
                  'p-3 rounded-lg text-sm',
                  saveMessage.type === 'success'
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-red-500/20 text-red-400'
                )}
              >
                {saveMessage.text}
              </div>
            )}

            {/* Save Button */}
            <button
              onClick={handleSave}
              disabled={isSaving}
              className={clsx(
                'w-full py-3 rounded-lg font-medium transition-colors',
                isSaving
                  ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  : 'bg-brand-600 hover:bg-brand-500 text-white'
              )}
            >
              {isSaving ? (
                <span className="flex items-center justify-center gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Guardando...
                </span>
              ) : (
                'Guardar Configuración'
              )}
            </button>
          </div>
        )}
      </div>

      {/* How it Works */}
      <div className="card">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-400" />
          ¿Cómo funciona?
        </h2>
        <ol className="space-y-4 text-gray-300 text-sm">
          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 bg-brand-500/20 text-brand-400 rounded-full flex items-center justify-center text-xs font-bold">
              1
            </span>
            <span>
              <strong>Configura tu username</strong> arriba (sin @). Esto se guarda en el servidor
              y se sincroniza con la extensión.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 bg-brand-500/20 text-brand-400 rounded-full flex items-center justify-center text-xs font-bold">
              2
            </span>
            <span>
              <strong>Visita tu propio perfil</strong> o cualquiera de tus posts/reels en Instagram o TikTok
              con la extensión Elena Bridge activa.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 bg-brand-500/20 text-brand-400 rounded-full flex items-center justify-center text-xs font-bold">
              3
            </span>
            <span>
              <strong>Haz clic en "Guardar Post"</strong>. La extensión detecta automáticamente que
              es tu cuenta y marca el contenido para feedback loop.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="flex-shrink-0 w-6 h-6 bg-brand-500/20 text-brand-400 rounded-full flex items-center justify-center text-xs font-bold">
              4
            </span>
            <span>
              <strong>El modelo ML aprende</strong> de las métricas reales de tu contenido,
              mejorando las predicciones futuras para tu negocio específico.
            </span>
          </li>
        </ol>
      </div>

      {/* Sync Extension Button (Optional) */}
      <div className="card bg-gray-800/50">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium text-white">Sincronizar con Extensión</h3>
            <p className="text-sm text-gray-400 mt-1">
              La extensión carga la configuración automáticamente, pero puedes
              forzar una sincronización manual.
            </p>
          </div>
          <button
            onClick={() => {
              // This would trigger a message to the extension
              // For now, just show an alert
              alert('La extensión sincronizará la configuración en la próxima extracción.')
            }}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm text-gray-300 transition-colors"
          >
            Sincronizar
          </button>
        </div>
      </div>
    </div>
  )
}
