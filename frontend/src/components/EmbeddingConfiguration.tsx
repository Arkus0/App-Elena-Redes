/**
 * EmbeddingConfiguration Component
 * =================================
 *
 * Allows users to configure embedding precision for ML predictions.
 *
 * Precision Levels:
 * - Ultra Baja (64 dims): Ultra rapido para PC modesto
 * - Baja (128 dims): Recomendado sobremesa normal Almeria (NEW DEFAULT)
 * - Media (256 dims): Balance precision/velocidad
 * - Alta (384 dims): Full dims con TruncatedSVD
 * - Maxima (full raw): Sin reduccion - mejor matices creativos/slang
 *
 * Default changed to "low" (128 dims) - safe for typical sobremesa:
 * - 100-2000 posts typical
 * - Training < 30 seconds
 * - RAM < 1GB on normal desktop
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
    case 'ultra_low':
      return <Zap className="w-5 h-5 text-orange-500" />;
    case 'low':
      return <Zap className="w-5 h-5 text-yellow-500" />;
    case 'medium':
      return <TrendingUp className="w-5 h-5 text-blue-500" />;
    case 'high':
      return <Cpu className="w-5 h-5 text-purple-500" />;
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
  const [selectedPrecision, setSelectedPrecision] = useState<string>('low');
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
            value: 'ultra_low',
            label: 'Ultra Baja (64 dims - ultra rapido)',
            dimensions: 64,
            description: 'Ultra rapido para PC modesto/sobremesa Almeria',
            performance: 'Muy rapido, minimo RAM (~0.3GB)',
            use_case: 'PC modesto, volumen muy alto',
            is_recommended: false,
          },
          {
            value: 'low',
            label: 'Baja (128 dims - Recomendada)',
            dimensions: 128,
            description: 'Recomendado para sobremesa normal Almeria',
            performance: 'Rapido, bajo RAM (~0.5GB). Train <30s',
            use_case: 'Sobremesa normal',
            is_recommended: true,
          },
          {
            value: 'medium',
            label: 'Media (256 dims)',
            dimensions: 256,
            description: 'Balance entre precision y velocidad',
            performance: 'Buen balance, RAM moderado (~1GB)',
            use_case: 'Balance',
            is_recommended: false,
          },
          {
            value: 'high',
            label: 'Alta (384 dims)',
            dimensions: 384,
            description: 'Precision completa con TruncatedSVD',
            performance: 'Full dims con reduccion. Train <1min',
            use_case: 'Precision maxima con reduccion',
            is_recommended: false,
          },
          {
            value: 'max',
            label: 'Maxima (full raw)',
            dimensions: 384,
            description: 'Embeddings raw sin reduccion - mejor matices creativos y slang local (Almeria/andaluz)',
            performance: 'Sin reduccion, RAM ~2GB. Solo si tienes buen hardware',
            use_case: 'Hardware potente',
            is_recommended: false,
          },
        ]);
        setSelectedPrecision('low');
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
      setSelectedPrecision(currentConfig?.precision || 'low');
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
                <p className="font-medium mb-1">Elige segun tu PC - Baja recomendado sobremesa normal Almeria</p>
                <p className="text-gray-300">
                  Ultra Baja (64 dims): PC modesto muy rapido.
                  Baja (128 dims): Recomendado sobremesa normal.
                  Maxima (full raw): Solo si tienes buen hardware, mejor matices creativos/slang.
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
