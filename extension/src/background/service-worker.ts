/**
 * Elena Bridge - Service Worker (Background Script)
 *
 * Maneja la comunicación entre el content script y la API backend.
 * Todas las peticiones HTTP se hacen desde aquí para evitar problemas de CORS.
 */

import type { ExtractionResult, ApiResponse, MessagePayload } from '../types';

// Configuración de la API
const API_CONFIG = {
  baseUrl: 'http://localhost:8000',
  endpoints: {
    ingest: '/api/ingest/raw',
    health: '/health'
  },
  timeout: 30000, // 30 segundos
  retryAttempts: 3,
  retryDelay: 1000 // 1 segundo entre reintentos
};

/**
 * Verifica si el servidor backend está disponible
 */
async function checkServerHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.health}`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Envía datos al backend con reintentos
 */
async function sendToApi(data: ExtractionResult): Promise<ApiResponse> {
  const payload = {
    source: 'elena_bridge_extension',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    profile: data.data,
    recentPosts: data.recentPosts || [],
    metadata: {
      extractionMethod: data.data?.extractionMethod,
      platform: data.data?.platform,
      sourceUrl: data.data?.sourceUrl
    }
  };

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= API_CONFIG.retryAttempts; attempt++) {
    try {
      console.log(`[Elena Bridge SW] Sending to API (attempt ${attempt}/${API_CONFIG.retryAttempts})`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);

      const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.ingest}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Elena-Bridge-Version': '1.0.0',
          'X-Extension-ID': chrome.runtime.id
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result = await response.json();

      console.log('[Elena Bridge SW] API response:', result);

      return {
        success: true,
        message: result.message || 'Datos enviados correctamente',
        taskId: result.task_id
      };

    } catch (error) {
      lastError = error as Error;
      console.warn(`[Elena Bridge SW] Attempt ${attempt} failed:`, error);

      if (attempt < API_CONFIG.retryAttempts) {
        // Esperar antes de reintentar (con backoff exponencial)
        await new Promise(resolve =>
          setTimeout(resolve, API_CONFIG.retryDelay * attempt)
        );
      }
    }
  }

  // Todos los intentos fallaron
  const errorMessage = lastError?.message || 'Error desconocido';

  // Verificar si el servidor está caído
  const serverOnline = await checkServerHealth();

  if (!serverOnline) {
    return {
      success: false,
      message: 'No se pudo conectar al servidor',
      error: 'El servidor backend no está disponible. Asegúrate de que esté ejecutándose en localhost:8000'
    };
  }

  return {
    success: false,
    message: 'Error al enviar datos',
    error: errorMessage
  };
}

/**
 * Maneja mensajes del content script y popup
 */
chrome.runtime.onMessage.addListener(
  (message: MessagePayload, _sender, sendResponse) => {
    console.log('[Elena Bridge SW] Received message:', message.action);

    switch (message.action) {
      case 'SEND_TO_API':
        if (message.data) {
          // Usar una IIFE async para manejar la promesa
          (async () => {
            try {
              const result = await sendToApi(message.data as ExtractionResult);
              sendResponse(result);
            } catch (error) {
              sendResponse({
                success: false,
                message: 'Error interno',
                error: (error as Error).message
              });
            }
          })();
          return true; // Indica que sendResponse se llamará de forma asíncrona
        }
        sendResponse({
          success: false,
          message: 'No hay datos para enviar',
          error: 'data field is missing'
        });
        break;

      case 'GET_STATUS':
        // Verificar estado del servidor
        (async () => {
          const online = await checkServerHealth();
          sendResponse({
            serverOnline: online,
            apiUrl: API_CONFIG.baseUrl
          });
        })();
        return true;

      default:
        sendResponse({
          success: false,
          message: 'Acción no reconocida',
          error: `Unknown action: ${message.action}`
        });
    }

    return false;
  }
);

/**
 * Evento de instalación de la extensión
 */
chrome.runtime.onInstalled.addListener((details) => {
  console.log('[Elena Bridge SW] Extension installed:', details.reason);

  if (details.reason === 'install') {
    // Primera instalación
    console.log('[Elena Bridge SW] Welcome to Elena Bridge!');

    // Verificar conexión con el servidor
    checkServerHealth().then(online => {
      if (online) {
        console.log('[Elena Bridge SW] Backend server is online');
      } else {
        console.warn('[Elena Bridge SW] Backend server is offline. Please start it.');
      }
    });
  }
});

/**
 * Evento cuando el service worker se activa
 */
self.addEventListener('activate', () => {
  console.log('[Elena Bridge SW] Service worker activated');
});

// Log de inicio
console.log('[Elena Bridge SW] Service worker loaded');
