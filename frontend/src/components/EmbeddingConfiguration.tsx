/**
 * EmbeddingConfiguration Component
 * =================================
 *
 * Allows users to configure embedding precision for ML predictions.
 *
 * Precision Levels:
 * - Baja (128 dims): Ultra fast, lower precision
 * - Media (256 dims): Balanced
 * - Alta (384 dims): Full precision
 * - Maxima (384 dims raw): Best for regional slang/nuances (default)
 *
 * Default is "max" (full raw embeddings) - safe for SMB volumes:
 * - 100-2000 posts typical
 * - Training < 1 minute
 * - RAM < 2GB on normal desktop
 */

import React, { useState, useEffect } from 'react';
import { Cpu, Zap, TrendingUp, HelpCircle, Check, AlertCircle } from 'lucide-react';
import { api } from '../services/api';

// Types
interface PrecisionOption {
  value: string;
  label: string;
  dimensions: number;
  description: string;
  performance: string;
  use_case: string;
  is_recommended: boolean;
}

interface EmbeddingConfig {
  business_id: number;
  precision: string;
  dimensions: number;
  description: string;
  performance_notes: string;
}

interface BenchmarkEstimate {
  dimensions_per_modality: number;
  total_embedding_features: number;
  total_features: number;
  estimated_ram_mb: number;
  estimated_train_seconds: number;
  is_lightweight: boolean;
  is_current: boolean;
}

interface BenchmarkResponse {
  business_id: number;
  current_precision: string;
  n_samples: number;
  benchmarks: Record<string, BenchmarkEstimate>;
  recommendation: {
    precision: string;
    reason: string;
  };
}

interface EmbeddingConfigurationProps {
  businessId: number;
  onConfigChange?: (precision: string) => void;
  className?: string;
}

// Precision level icons
const PrecisionIcon = ({ precision }: { precision: string }) => {
  switch (precision) {
    case 'low':
      return <Zap className="w-5 h-5 text-yellow-500" />;
    case 'medium':
      return <TrendingUp className="w-5 h-5 text-blue-500" />;
    case 'high':
    case 'max':
      return <Cpu className="w-5 h-5 text-green-500" />;
    default:
      return <Cpu className="w-5 h-5 text-gray-500" />;
  }
};

export const EmbeddingConfiguration: React.FC<EmbeddingConfigurationProps> = ({
  businessId,
  onConfigChange,
  className = '',
}) => {
  // State
  const [options, setOptions] = useState<PrecisionOption[]>([]);
  const [currentConfig, setCurrentConfig] = useState<EmbeddingConfig | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkResponse | null>(null);
  const [selectedPrecision, setSelectedPrecision] = useState<string>('max');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTooltip, setShowTooltip] = useState<string | null>(null);

  // Fetch available options and current config
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch available precision options
        const infoResponse = await api.get('/embeddings/info');
        setOptions(infoResponse.data.available_precisions);

        // Fetch current business config
        const configResponse = await api.get(`/embeddings/${businessId}`);
        setCurrentConfig(configResponse.data);
        setSelectedPrecision(configResponse.data.precision);

        // Fetch benchmark estimates
        const benchmarkResponse = await api.get(`/embeddings/${businessId}/benchmark`);
        setBenchmarks(benchmarkResponse.data);
      } catch (err) {
        console.error('Error fetching embedding config:', err);
        setError('Error al cargar configuracion de embeddings');
        // Set defaults if API fails
        setOptions([
          {
            value: 'low',
            label: 'Baja',
            dimensions: 128,
            description: 'Ultra rapido, menor precision semantica',
            performance: 'Mas rapido, menor RAM',
            use_case: 'Volumen alto',
            is_recommended: false,
          },
          {
            value: 'medium',
            label: 'Media',
            dimensions: 256,
            description: 'Balance entre precision y velocidad',
            performance: 'Buen balance',
            use_case: 'Balance',
            is_recommended: false,
          },
          {
            value: 'high',
            label: 'Alta',
            dimensions: 384,
            description: 'Precision completa',
            performance: 'Todas las dimensiones',
            use_case: 'Precision maxima',
            is_recommended: false,
          },
          {
            value: 'max',
            label: 'Maxima (Recomendada)',
            dimensions: 384,
            description: 'Embeddings raw completos - mejor matices creativos y slang local',
            performance: 'Seguro en PC normal: train <1min, RAM <2GB',
            use_case: 'SMB tipico',
            is_recommended: true,
          },
        ]);
        setSelectedPrecision('max');
      } finally {
        setLoading(false);
      }
    };

    if (businessId) {
      fetchData();
    }
  }, [businessId]);

  // Handle precision change
  const handlePrecisionChange = async (precision: string) => {
    setSelectedPrecision(precision);
    setSaving(true);
    setError(null);

    try {
      const response = await api.put(`/embeddings/${businessId}`, {
        precision: precision,
      });
      setCurrentConfig(response.data);
      onConfigChange?.(precision);
    } catch (err) {
      console.error('Error updating embedding precision:', err);
      setError('Error al actualizar precision. Intenta de nuevo.');
      // Revert to previous
      setSelectedPrecision(currentConfig?.precision || 'max');
    } finally {
      setSaving(false);
    }
  };

  // Get benchmark for a precision level
  const getBenchmark = (precision: string): BenchmarkEstimate | null => {
    return benchmarks?.benchmarks?.[precision] || null;
  };

  if (loading) {
    return (
      <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            <div className="h-16 bg-gray-200 rounded"></div>
            <div className="h-16 bg-gray-200 rounded"></div>
            <div className="h-16 bg-gray-200 rounded"></div>
            <div className="h-16 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-indigo-600" />
            <h3 className="text-lg font-medium text-gray-900">
              Precision de Embeddings
            </h3>
          </div>
          <div
            className="relative"
            onMouseEnter={() => setShowTooltip('info')}
            onMouseLeave={() => setShowTooltip(null)}
          >
            <HelpCircle className="w-5 h-5 text-gray-400 cursor-help" />
            {showTooltip === 'info' && (
              <div className="absolute right-0 z-10 w-72 p-3 mt-2 text-sm bg-gray-900 text-white rounded-lg shadow-lg">
                <p className="font-medium mb-1">Full dims captura mejor slang regional</p>
                <p className="text-gray-300">
                  Almeria/andaluz, jerga local, matices creativos.
                  Seguro en PC normal con tus datos (train rapido).
                </p>
              </div>
            )}
          </div>
        </div>
        <p className="mt-1 text-sm text-gray-500">
          Configura la dimension de embeddings para predicciones ML
        </p>
      </div>

      {/* Error message */}
      {error && (
        <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
          <span className="text-sm text-red-700">{error}</span>
        </div>
      )}

      {/* Precision Options */}
      <div className="p-6 space-y-3">
        {options.map((option) => {
          const benchmark = getBenchmark(option.value);
          const isSelected = selectedPrecision === option.value;

          return (
            <button
              key={option.value}
              onClick={() => handlePrecisionChange(option.value)}
              disabled={saving}
              className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
                isSelected
                  ? 'border-indigo-500 bg-indigo-50'
                  : 'border-gray-200 hover:border-gray-300 bg-white'
              } ${saving ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  <div className="mt-0.5">
                    <PrecisionIcon precision={option.value} />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium text-gray-900">
                        {option.label}
                      </span>
                      <span className="text-sm text-gray-500">
                        ({option.dimensions} dims)
                      </span>
                      {option.is_recommended && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                          Recomendada
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-sm text-gray-600">
                      {option.description}
                    </p>
                    <p className="mt-1 text-xs text-gray-500">
                      {option.performance}
                    </p>
                    {benchmark && (
                      <div className="mt-2 flex items-center space-x-4 text-xs text-gray-500">
                        <span>
                          RAM: ~{benchmark.estimated_ram_mb.toFixed(1)} MB
                        </span>
                        <span>
                          Train: ~{benchmark.estimated_train_seconds.toFixed(1)}s
                        </span>
                        {benchmark.is_lightweight && (
                          <span className="text-green-600 font-medium">
                            Lightweight
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex-shrink-0">
                  {isSelected && (
                    <Check className="w-5 h-5 text-indigo-600" />
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Recommendation */}
      {benchmarks?.recommendation && (
        <div className="px-6 pb-6">
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <span className="font-medium">Recomendacion: </span>
              {benchmarks.recommendation.reason}
            </p>
          </div>
        </div>
      )}

      {/* Note about retraining */}
      {currentConfig && selectedPrecision !== currentConfig.precision && (
        <div className="px-6 pb-6">
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800">
              <span className="font-medium">Nota: </span>
              Cambiar la precision puede requerir reentrenar el modelo para mejores resultados.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmbeddingConfiguration;
