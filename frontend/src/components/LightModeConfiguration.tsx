/**
 * LightModeConfiguration Component
 * =================================
 *
 * Toggle and configure lightweight multimodal processing (Whisper/EasyOCR).
 *
 * Light Mode Benefits:
 * - ~5s processing vs ~20s full (75% faster)
 * - Whisper 'tiny' model, first 3 seconds only (hook analysis)
 * - EasyOCR first 5 frames or thumbnail only
 * - Hash-based cache for instant re-processing
 * - Minimal quality loss for engagement prediction
 *
 * Recommended for sobremesa normal (typical desktop).
 */

import React, { useState, useEffect } from 'react';
import { Zap, Clock, HardDrive, Check, AlertCircle, ToggleLeft, ToggleRight, ChevronDown, ChevronUp } from 'lucide-react';
import { api } from '../services/api';

// Types
interface LightModeOption {
  value: string;
  label: string;
  description: string;
  whisper_model: string;
  max_duration_seconds: number;
  ocr_max_frames: number;
  estimated_time_seconds: number;
  estimated_ram_mb: number;
  is_recommended: boolean;
}

interface LightModeConfig {
  business_id: number;
  light_mode_enabled: boolean;
  whisper_model: string;
  max_duration_seconds: number;
  ocr_max_frames: number;
  use_thumbnail: boolean;
  cache_enabled: boolean;
  skip_non_video: boolean;
  description: string;
}

interface BenchmarkEstimate {
  mode: string;
  estimated_time_seconds: number;
  estimated_ram_mb: number;
  whisper_model: string;
  ocr_frames: number;
  hook_duration: number;
  is_current: boolean;
  description: string;
}

interface LightModeConfigurationProps {
  businessId: number;
  onConfigChange?: (enabled: boolean) => void;
  className?: string;
  compact?: boolean;
}

export const LightModeConfiguration: React.FC<LightModeConfigurationProps> = ({
  businessId,
  onConfigChange,
  className = '',
  compact = false,
}) => {
  // State
  const [config, setConfig] = useState<LightModeConfig | null>(null);
  const [options, setOptions] = useState<LightModeOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(!compact);

  // Fetch config on mount
  useEffect(() => {
    const fetchConfig = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch current config
        const configResponse = await api.get(`/light-mode/${businessId}`);
        setConfig(configResponse.data);

        // Fetch available options
        const infoResponse = await api.get('/light-mode/info');
        setOptions(infoResponse.data.available_modes);
      } catch (err) {
        console.error('Error fetching light mode config:', err);
        setError('Error al cargar configuracion de modo ligero');
        // Set defaults
        setConfig({
          business_id: businessId,
          light_mode_enabled: true,
          whisper_model: 'tiny',
          max_duration_seconds: 3.0,
          ocr_max_frames: 5,
          use_thumbnail: true,
          cache_enabled: true,
          skip_non_video: true,
          description: 'Modo ligero - procesamiento rapido (~5s)',
        });
      } finally {
        setLoading(false);
      }
    };

    if (businessId) {
      fetchConfig();
    }
  }, [businessId]);

  // Handle quick toggle
  const handleToggle = async () => {
    if (!config) return;

    setToggling(true);
    setError(null);

    try {
      const newEnabled = !config.light_mode_enabled;
      const response = await api.post(
        `/light-mode/quick-toggle/${businessId}`,
        null,
        { params: { enabled: newEnabled } }
      );

      setConfig({
        ...config,
        light_mode_enabled: response.data.light_mode_enabled,
        description: response.data.processing_estimate,
      });

      onConfigChange?.(newEnabled);
    } catch (err) {
      console.error('Error toggling light mode:', err);
      setError('Error al cambiar modo ligero');
    } finally {
      setToggling(false);
    }
  };

  // Compact view (for dashboard)
  if (compact) {
    return (
      <div className={`flex items-center justify-between p-3 bg-gray-800/50 rounded-lg ${className}`}>
        <div className="flex items-center space-x-3">
          <div className={`p-1.5 rounded-lg ${config?.light_mode_enabled ? 'bg-green-500/20' : 'bg-gray-700'}`}>
            <Zap className={`w-4 h-4 ${config?.light_mode_enabled ? 'text-green-400' : 'text-gray-400'}`} />
          </div>
          <div>
            <p className="text-sm font-medium text-white">Modo Ligero Multimodal</p>
            <p className="text-xs text-gray-400">
              {config?.light_mode_enabled ? '~5s por video (75% mas rapido)' : '~20s por video (full)'}
            </p>
          </div>
        </div>
        <button
          onClick={handleToggle}
          disabled={loading || toggling}
          className={`relative p-1 rounded-full transition-colors ${
            config?.light_mode_enabled
              ? 'bg-green-500 hover:bg-green-600'
              : 'bg-gray-600 hover:bg-gray-500'
          } ${(loading || toggling) ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {config?.light_mode_enabled ? (
            <ToggleRight className="w-6 h-6 text-white" />
          ) : (
            <ToggleLeft className="w-6 h-6 text-white" />
          )}
        </button>
      </div>
    );
  }

  // Full view
  if (loading) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
          <div className="h-20 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg ${className}`}>
      {/* Header with toggle */}
      <div className="px-6 py-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${config?.light_mode_enabled ? 'bg-green-500/20' : 'bg-gray-700'}`}>
              <Zap className={`w-5 h-5 ${config?.light_mode_enabled ? 'text-green-400' : 'text-gray-400'}`} />
            </div>
            <div>
              <h3 className="text-lg font-medium text-white">
                Modo Ligero Multimodal
              </h3>
              <p className="text-sm text-gray-400">
                Optimiza Whisper/EasyOCR para procesamiento rapido
              </p>
            </div>
          </div>

          {/* Main toggle */}
          <button
            onClick={handleToggle}
            disabled={toggling}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              config?.light_mode_enabled
                ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            } ${toggling ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {config?.light_mode_enabled ? (
              <>
                <Check className="w-4 h-4" />
                <span>Activado</span>
              </>
            ) : (
              <span>Desactivado</span>
            )}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="mx-6 mt-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
          <span className="text-sm text-red-300">{error}</span>
        </div>
      )}

      {/* Status banner */}
      <div className="px-6 py-4">
        <div className={`p-4 rounded-lg ${
          config?.light_mode_enabled
            ? 'bg-green-500/10 border border-green-500/30'
            : 'bg-amber-500/10 border border-amber-500/30'
        }`}>
          <div className="flex items-start space-x-3">
            <div className={`p-2 rounded-lg ${
              config?.light_mode_enabled ? 'bg-green-500/20' : 'bg-amber-500/20'
            }`}>
              <Clock className={`w-5 h-5 ${
                config?.light_mode_enabled ? 'text-green-400' : 'text-amber-400'
              }`} />
            </div>
            <div className="flex-1">
              <h4 className={`font-medium ${
                config?.light_mode_enabled ? 'text-green-300' : 'text-amber-300'
              }`}>
                {config?.light_mode_enabled
                  ? 'Procesamiento Rapido Activado'
                  : 'Procesamiento Completo Activado'
                }
              </h4>
              <p className="text-sm text-gray-300 mt-1">
                {config?.light_mode_enabled
                  ? 'Whisper tiny (3s), OCR 5 frames. ~5s por Reel/Video. 75% mas rapido, calidad hook optima.'
                  : 'Whisper base completo, OCR todos los frames. ~20s por video. Maxima precision.'
                }
              </p>

              {/* Time comparison */}
              <div className="mt-3 flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-gray-300">
                    Tiempo: <span className="font-medium text-white">
                      {config?.light_mode_enabled ? '~5s' : '~20s'}
                    </span>
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <HardDrive className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-gray-300">
                    RAM: <span className="font-medium text-white">
                      {config?.light_mode_enabled ? '~200MB' : '~500MB'}
                    </span>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Details section (collapsible) */}
      <div className="px-6 pb-4">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center space-x-2 text-sm text-gray-400 hover:text-gray-300"
        >
          {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          <span>{showDetails ? 'Ocultar detalles' : 'Ver detalles'}</span>
        </button>

        {showDetails && (
          <div className="mt-4 space-y-4">
            {/* Processing modes */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {options.map((option) => {
                const isCurrent = config?.light_mode_enabled
                  ? option.value === 'light'
                  : option.value === 'full';

                return (
                  <div
                    key={option.value}
                    className={`p-4 rounded-lg border ${
                      isCurrent
                        ? 'border-green-500/50 bg-green-500/10'
                        : 'border-gray-700 bg-gray-700/30'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="font-medium text-white">{option.label}</h5>
                      {option.is_recommended && (
                        <span className="px-2 py-0.5 text-xs font-medium bg-green-500/20 text-green-400 rounded-full">
                          Recomendado
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-400 mb-3">{option.description}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>Tiempo: ~{option.estimated_time_seconds}s</span>
                      <span>RAM: ~{option.estimated_ram_mb}MB</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Current config details */}
            {config && (
              <div className="p-4 bg-gray-700/30 rounded-lg">
                <h5 className="text-sm font-medium text-gray-300 mb-3">Configuracion Actual</h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Modelo Whisper</span>
                    <p className="text-white font-medium">{config.whisper_model}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Duracion Max</span>
                    <p className="text-white font-medium">{config.max_duration_seconds}s</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Frames OCR</span>
                    <p className="text-white font-medium">{config.ocr_max_frames}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Cache</span>
                    <p className="text-white font-medium">{config.cache_enabled ? 'Activo' : 'Inactivo'}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default LightModeConfiguration;
