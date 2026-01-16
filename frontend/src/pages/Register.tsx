import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Zap, Mail, Lock, User, ArrowRight } from 'lucide-react'
import toast from 'react-hot-toast'
import { authApi } from '../services/api'
import { useAuthStore } from '../stores/authStore'

export default function Register() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()
  const setAuth = useAuthStore((state) => state.setAuth)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      // Register
      const user = await authApi.register(email, password, fullName)

      // Auto login after registration
      const { access_token } = await authApi.login(email, password)
      setAuth(user, access_token)

      toast.success('¡Cuenta creada! Configura tu negocio')
      navigate('/onboarding')
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Error al crear cuenta')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        {/* Logo */}
        <div className="flex justify-center">
          <div className="flex items-center gap-2">
            <Zap className="w-12 h-12 text-brand-500" />
            <span className="text-3xl font-bold gradient-text">BrandPulse AI</span>
          </div>
        </div>
        <h2 className="mt-6 text-center text-2xl font-bold text-white">
          Crea tu cuenta gratis
        </h2>
        <p className="mt-2 text-center text-sm text-gray-400">
          Empieza a generar contenido viral para tu negocio
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="fullName" className="block text-sm font-medium text-gray-300">
                Nombre completo
              </label>
              <div className="mt-1 relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  id="fullName"
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="input-field pl-10"
                  placeholder="Tu nombre"
                />
              </div>
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300">
                Email
              </label>
              <div className="mt-1 relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  id="email"
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input-field pl-10"
                  placeholder="tu@email.com"
                />
              </div>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300">
                Contraseña
              </label>
              <div className="mt-1 relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  id="password"
                  type="password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-field pl-10"
                  placeholder="Mínimo 6 caracteres"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  Crear cuenta
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-gray-400">
            ¿Ya tienes cuenta?{' '}
            <Link to="/login" className="text-brand-400 hover:text-brand-300 font-medium">
              Inicia sesión
            </Link>
          </p>
        </div>

        {/* Features preview */}
        <div className="mt-8 grid grid-cols-2 gap-4">
          {[
            { icon: '📊', text: 'Análisis de competidores' },
            { icon: '🎯', text: 'Contenido optimizado' },
            { icon: '📅', text: 'Calendarios mensuales' },
            { icon: '🚀', text: 'Ideas virales' },
          ].map((feature, i) => (
            <div
              key={i}
              className="flex items-center gap-2 p-3 bg-gray-800/50 rounded-lg border border-gray-700"
            >
              <span className="text-xl">{feature.icon}</span>
              <span className="text-sm text-gray-300">{feature.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
