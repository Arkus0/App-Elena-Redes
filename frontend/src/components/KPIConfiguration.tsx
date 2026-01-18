/**
 * KPI Configuration Component
 *
 * Allows users to configure engagement weights for multi-objective predictions.
 * Supports:
 * - Custom weight sliders (1-20)
 * - Predefined templates (Brand Awareness, Leads, Community, etc.)
 * - Live preview of RPI impact
 * - Save/load configurations
 *
 * @author BrandPulse AI
 */

import { useState, useEffect, useCallback } from 'react'
import { kpiApi, multiOutputApi } from '../services/api'
import type {
  EngagementWeights,
  KPIWeightsConfig,
  KPITemplate,
  KPIWeightsPreview,
} from '../types'

// Icons
const icons: Record<string, string> = {
  eye: '👁️',
  target: '🎯',
  users: '👥',
  share: '🚀',
  balance: '⚖️',
  chart: '📊',
}

// Default weights
const DEFAULT_WEIGHTS: EngagementWeights = {
  likes_weight: 1.0,
  comments_weight: 2.0,
  shares_weight: 10.0,
  saves_weight: 5.0,
  views_weight: 3.0,
}

interface KPIConfigurationProps {
  businessId: number
  businessType?: string
  onWeightsSaved?: (weights: EngagementWeights) => void
}

export function KPIConfiguration({
  businessId,
  businessType,
  onWeightsSaved,
}: KPIConfigurationProps) {
  // State
  const [weights, setWeights] = useState<EngagementWeights>(DEFAULT_WEIGHTS)
  const [templates, setTemplates] = useState<KPITemplate[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null)
  const [preview, setPreview] = useState<KPIWeightsPreview | null>(null)
  const [savedConfig, setSavedConfig] = useState<KPIWeightsConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Load initial data
  useEffect(() => {
    async function loadData() {
      setLoading(true)
      setError(null)

      try {
        // Load templates and saved config in parallel
        const [templatesData, configData] = await Promise.all([
          kpiApi.getTemplates(),
          kpiApi.getWeights(businessId).catch(() => null),
        ])

        setTemplates(templatesData)

        if (configData && configData.id > 0) {
          setSavedConfig(configData)
          setWeights(configData.weights)
          setSelectedTemplate(configData.template_name)
        }
      } catch (err) {
        console.error('Error loading KPI config:', err)
        setError('Error al cargar configuracion')
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [businessId])

  // Update preview when weights change
  useEffect(() => {
    const updatePreview = async () => {
      try {
        const previewData = await kpiApi.previewWeights(weights)
        setPreview(previewData)
      } catch (err) {
        console.error('Preview error:', err)
      }
    }

    const debounce = setTimeout(updatePreview, 300)
    return () => clearTimeout(debounce)
  }, [weights])

  // Handle weight change
  const handleWeightChange = (metric: keyof EngagementWeights, value: number) => {
    setWeights((prev) => ({
      ...prev,
      [metric]: value,
    }))
    setSelectedTemplate(null) // Clear template when manually adjusting
  }

  // Apply template
  const applyTemplate = (template: KPITemplate) => {
    setWeights(template.weights)
    setSelectedTemplate(template.name)
  }

  // Save weights
  const saveWeights = async () => {
    setSaving(true)
    setError(null)

    try {
      const saved = await kpiApi.saveWeights(businessId, weights, {
        template_name: selectedTemplate || undefined,
        description: selectedTemplate
          ? `Configuracion '${selectedTemplate}'`
          : 'Configuracion personalizada',
      })

      setSavedConfig(saved)
      onWeightsSaved?.(weights)
    } catch (err) {
      console.error('Save error:', err)
      setError('Error al guardar configuracion')
    } finally {
      setSaving(false)
    }
  }

  // Reset to defaults
  const resetToDefaults = () => {
    setWeights(DEFAULT_WEIGHTS)
    setSelectedTemplate(null)
  }

  // Calculate normalized percentages
  const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0)
  const getPercentage = (weight: number) =>
    totalWeight > 0 ? ((weight / totalWeight) * 100).toFixed(0) : '0'

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/3"></div>
          <div className="h-4 bg-gray-200 rounded w-2/3"></div>
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900">
          Configurar mis KPIs
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Personaliza como se calcula el RPI segun tus objetivos de negocio
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Templates */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">
            Plantillas predefinidas
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {templates.map((template) => (
              <button
                key={template.name}
                onClick={() => applyTemplate(template)}
                className={`p-3 rounded-lg border-2 transition-all text-left ${
                  selectedTemplate === template.name
                    ? 'border-indigo-500 bg-indigo-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <span className="text-2xl">{icons[template.icon] || '📊'}</span>
                <div className="mt-1 font-medium text-sm text-gray-900">
                  {template.display_name}
                </div>
                <div className="text-xs text-gray-500 line-clamp-2">
                  {template.description}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Weight Sliders */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-700">
              Pesos de metricas
            </h3>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm text-indigo-600 hover:text-indigo-800"
            >
              {showAdvanced ? 'Ocultar detalles' : 'Ver detalles'}
            </button>
          </div>

          <div className="space-y-4">
            {/* Likes */}
            <WeightSlider
              label="Likes"
              value={weights.likes_weight}
              onChange={(v) => handleWeightChange('likes_weight', v)}
              percentage={getPercentage(weights.likes_weight)}
              description={showAdvanced ? 'Engagement basico, facil de obtener' : undefined}
              color="pink"
            />

            {/* Comments */}
            <WeightSlider
              label="Comentarios"
              value={weights.comments_weight}
              onChange={(v) => handleWeightChange('comments_weight', v)}
              percentage={getPercentage(weights.comments_weight)}
              description={showAdvanced ? 'Indica engagement profundo y comunidad' : undefined}
              color="blue"
            />

            {/* Shares */}
            <WeightSlider
              label="Compartidos"
              value={weights.shares_weight}
              onChange={(v) => handleWeightChange('shares_weight', v)}
              percentage={getPercentage(weights.shares_weight)}
              description={showAdvanced ? 'Potencial viral, amplificacion organica' : undefined}
              color="green"
            />

            {/* Saves */}
            <WeightSlider
              label="Guardados"
              value={weights.saves_weight}
              onChange={(v) => handleWeightChange('saves_weight', v)}
              percentage={getPercentage(weights.saves_weight)}
              description={showAdvanced ? 'Intencion de compra, contenido valioso' : undefined}
              color="yellow"
            />

            {/* Views */}
            <WeightSlider
              label="Views (Reels)"
              value={weights.views_weight}
              onChange={(v) => handleWeightChange('views_weight', v)}
              percentage={getPercentage(weights.views_weight)}
              description={showAdvanced ? 'Alcance, especialmente importante para Reels' : undefined}
              color="purple"
            />
          </div>
        </div>

        {/* Preview */}
        {preview && (
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Vista previa
            </h3>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-indigo-600">
                  RPI: {preview.weighted_rpi.toFixed(1)}
                </div>
                <div className="text-sm text-gray-500">
                  {preview.explanation}
                </div>
              </div>
              <div className="text-right">
                <div
                  className={`text-sm font-medium ${
                    preview.comparison_to_default.difference > 0
                      ? 'text-green-600'
                      : preview.comparison_to_default.difference < 0
                      ? 'text-red-600'
                      : 'text-gray-600'
                  }`}
                >
                  {preview.comparison_to_default.difference > 0 ? '+' : ''}
                  {preview.comparison_to_default.difference.toFixed(1)} vs default
                </div>
                <div className="text-xs text-gray-400">
                  ({preview.comparison_to_default.percent_change.toFixed(1)}%)
                </div>
              </div>
            </div>

            {/* Weight distribution bar */}
            <div className="mt-3">
              <div className="h-2 flex rounded-full overflow-hidden">
                <div
                  className="bg-pink-400"
                  style={{ width: `${getPercentage(weights.likes_weight)}%` }}
                />
                <div
                  className="bg-blue-400"
                  style={{ width: `${getPercentage(weights.comments_weight)}%` }}
                />
                <div
                  className="bg-green-400"
                  style={{ width: `${getPercentage(weights.shares_weight)}%` }}
                />
                <div
                  className="bg-yellow-400"
                  style={{ width: `${getPercentage(weights.saves_weight)}%` }}
                />
                <div
                  className="bg-purple-400"
                  style={{ width: `${getPercentage(weights.views_weight)}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Likes</span>
                <span>Coment.</span>
                <span>Shares</span>
                <span>Saves</span>
                <span>Views</span>
              </div>
            </div>
          </div>
        )}

        {/* Business-specific tip */}
        {businessType && (
          <div className="bg-indigo-50 rounded-lg p-4 text-sm">
            <div className="font-medium text-indigo-900">
              Recomendacion para {businessType}
            </div>
            <div className="text-indigo-700 mt-1">
              {getBusinessRecommendation(businessType)}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-between pt-4 border-t border-gray-200">
          <button
            onClick={resetToDefaults}
            className="text-sm text-gray-600 hover:text-gray-800"
          >
            Restablecer defaults
          </button>

          <div className="flex gap-3">
            {savedConfig && savedConfig.id > 0 && (
              <span className="text-sm text-gray-400 self-center">
                Guardado: {new Date(savedConfig.updated_at).toLocaleDateString()}
              </span>
            )}
            <button
              onClick={saveWeights}
              disabled={saving}
              className={`px-4 py-2 rounded-md font-medium text-white ${
                saving
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700'
              }`}
            >
              {saving ? 'Guardando...' : 'Guardar configuracion'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// Weight Slider Component
interface WeightSliderProps {
  label: string
  value: number
  onChange: (value: number) => void
  percentage: string
  description?: string
  color: 'pink' | 'blue' | 'green' | 'yellow' | 'purple'
}

function WeightSlider({
  label,
  value,
  onChange,
  percentage,
  description,
  color,
}: WeightSliderProps) {
  const colorClasses = {
    pink: 'accent-pink-500',
    blue: 'accent-blue-500',
    green: 'accent-green-500',
    yellow: 'accent-yellow-500',
    purple: 'accent-purple-500',
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-sm font-medium text-gray-700">{label}</label>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">{percentage}%</span>
          <input
            type="number"
            value={value}
            onChange={(e) => onChange(Math.max(0, Math.min(20, Number(e.target.value))))}
            className="w-16 px-2 py-1 text-sm border border-gray-300 rounded-md"
            min="0"
            max="20"
            step="0.5"
          />
        </div>
      </div>
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min="0"
        max="20"
        step="0.5"
        className={`w-full h-2 rounded-lg appearance-none cursor-pointer ${colorClasses[color]}`}
      />
      {description && (
        <div className="text-xs text-gray-400 mt-1">{description}</div>
      )}
    </div>
  )
}

// Business-specific recommendations
function getBusinessRecommendation(businessType: string): string {
  const recommendations: Record<string, string> = {
    inmobiliaria:
      'Para inmobiliarias, prioriza guardados (interes en propiedades) y compartidos (referidos).',
    restaurante:
      'Para restaurantes, equilibra comentarios (reservas, preguntas) y guardados (guardar para visitar).',
    cafeteria:
      'Para cafeterias, enfoca en comunidad (comentarios) y alcance local (views).',
    peluqueria:
      'Para peluquerias, prioriza guardados (inspiracion) y compartidos (recomendaciones).',
    gimnasio:
      'Para gimnasios, equilibra comunidad (comentarios motivacionales) y guardados (rutinas).',
    clinica:
      'Para clinicas, prioriza guardados (informacion util) y comentarios (consultas).',
    floristeria:
      'Para floristerias, enfoca en guardados (inspiracion) y views (descubrimiento visual).',
    tienda_local:
      'Para tiendas locales, balancea guardados (productos deseados) y compartidos (recomendaciones).',
  }

  return (
    recommendations[businessType] ||
    'Ajusta los pesos segun tus objetivos principales de negocio.'
  )
}

export default KPIConfiguration
