import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Zap, Mail, Lock, ArrowRight } from 'lucide-react'
import toast from 'react-hot-toast'
import { authApi, businessApi } from '../services/api'
import { useAuthStore } from '../stores/authStore'
import { useBusinessStore } from '../stores/businessStore'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()
  const setAuth = useAuthStore((state) => state.setAuth)
  const setCurrentBusiness = useBusinessStore((state) => state.setCurrentBusiness)
  const setBusinesses = useBusinessStore((state) => state.setBusinesses)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      // Login
      const { access_token } = await authApi.login(email, password)

      // Set auth state (this will also set the token in localStorage)
      setAuth({ id: 0, email, full_name: null, is_active: true, created_at: '' }, access_token)

      // Get user's businesses
      const businesses = await businessApi.getMyBusinesses()
      setBusinesses(businesses)

      if (businesses.length > 0) {
        setCurrentBusiness(businesses[0])
        toast.success('¡Bienvenido de vuelta!')
        navigate('/')
      } else {
        toast.success('¡Bienvenido! Configura tu primer negocio')
        navigate('/onboarding')
      }
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Error al iniciar sesión')
    } finally {
      setIsLoading(false)
    }
  }

  // Demo login for testing
  const handleDemoLogin = async () => {
    setEmail('demo@brandpulse.ai')
    setPassword('demo123')
    // Create mock auth for demo
    setAuth(
      { id: 1, email: 'demo@brandpulse.ai', full_name: 'Usuario Demo', is_active: true, created_at: new Date().toISOString() },
      'demo-token'
    )
    setCurrentBusiness({
      id: 1,
      name: 'Floristería Rosa',
      business_type: 'floristeria',
      description: 'Tu floristería de confianza en Barcelona',
      location: 'Barcelona',
      instagram_handle: 'floristeria_rosa',
      tiktok_handle: 'floristeriarosa',
      linkedin_handle: null,
      active_platforms: ['instagram', 'tiktok'],
      content_goals: ['engagement', 'leads'],
      posting_frequency: '3-5_per_week',
      brand_voice: 'friendly_professional',
      onboarding_completed: new Date().toISOString(),
      last_analysis_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
    })
    toast.success('¡Bienvenido al modo demo!')
    navigate('/')
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
          Inicia sesión en tu cuenta
        </h2>
        <p className="mt-2 text-center text-sm text-gray-400">
          Genera contenido de alto engagement para tu negocio
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-6">
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
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-field pl-10"
                  placeholder="••••••••"
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
                  Iniciar sesión
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          <div className="mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-700" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-gray-800 text-gray-400">o</span>
              </div>
            </div>

            <button
              onClick={handleDemoLogin}
              className="mt-4 w-full btn-secondary flex items-center justify-center gap-2"
            >
              <Sparkles className="w-4 h-4" />
              Probar modo demo
            </button>
          </div>

          <p className="mt-6 text-center text-sm text-gray-400">
            ¿No tienes cuenta?{' '}
            <Link to="/register" className="text-brand-400 hover:text-brand-300 font-medium">
              Regístrate gratis
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}

function Sparkles(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
      <path d="M5 3v4" />
      <path d="M19 17v4" />
      <path d="M3 5h4" />
      <path d="M17 19h4" />
    </svg>
  )
}
