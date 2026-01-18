/**
 * Elena Bridge - Service Worker (Background Script)
 *
 * Maneja la comunicación entre el content script y la API backend.
 * Soporta tanto contenido individual (posts, reels, videos) como perfiles completos.
 * Todas las peticiones HTTP se hacen desde aquí para evitar problemas de CORS.
 */

import type { ExtractionResult, ApiResponse, MessagePayload, UserConfig } from '../types';

// ============================================================================
// Human-in-the-Loop: User Configuration Management
// ============================================================================

/**
 * Obtiene la configuración del usuario desde chrome.storage
 * Incluye own_username para detectar posts propios
 */
async function getUserConfig(): Promise<UserConfig> {
  return new Promise((resolve) => {
    chrome.storage.sync.get(['userConfig'], (result) => {
      resolve(result.userConfig || {});
    });
  });
}

/**
 * Detecta si el contenido extraído es del perfil propio de la clienta
 * Compara el autor del contenido con el username configurado
 *
 * @param data - Datos extraídos (contenido o perfil)
 * @param config - Configuración del usuario con own_username
 * @returns true si el contenido es del perfil propio
 */
function isOwnProfileContent(data: ExtractionResult, config: UserConfig): boolean {
  if (!data.success) return false;

  // Obtener el username del contenido/perfil
  let contentUsername = '';
  let platform = '';

  if (data.content) {
    contentUsername = data.content.author.username.toLowerCase().replace('@', '');
    platform = data.content.platform;
  } else if (data.profile) {
    contentUsername = data.profile.username.toLowerCase().replace('@', '');
    platform = data.profile.platform;
  }

  if (!contentUsername) return false;

  // Comparar con el username configurado según la plataforma
  let ownUsername = '';

  if (platform === 'instagram' && config.ownInstagramUsername) {
    ownUsername = config.ownInstagramUsername.toLowerCase().replace('@', '');
  } else if (platform === 'tiktok' && config.ownTiktokUsername) {
    ownUsername = config.ownTiktokUsername.toLowerCase().replace('@', '');
  }

  const isOwn = ownUsername !== '' && contentUsername === ownUsername;

  if (isOwn) {
    console.log(`[Elena Bridge SW] PERFIL PROPIO detectado: @${contentUsername} (${platform})`);
  }

  return isOwn;
}

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
 * Construye el payload para la API según el tipo de datos extraídos
 *
 * @param data - Datos extraídos del contenido/perfil
 * @param isOwnProfile - True si el contenido es del perfil propio (activa feedback loop ML)
 */
function buildPayload(data: ExtractionResult, isOwnProfile: boolean = false) {
  const basePayload = {
    source: 'elena_bridge_extension',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    pageType: data.pageType,
    // Human-in-the-Loop: flag para activar feedback loop en backend
    isOwnProfile
  };

  // Contenido individual (posts, reels, videos)
  if (data.content) {
    return {
      ...basePayload,
      type: 'content',
      content: data.content,
      metadata: {
        contentType: data.content.contentType,
        platform: data.content.platform,
        contentId: data.content.contentId,
        author: data.content.author.username,
        sourceUrl: data.content.sourceUrl,
        extractionMethod: data.content.extractionMethod,
        // Indica al backend que debe registrar métricas reales
        isOwnProfile
      }
    };
  }

  // Perfil completo
  if (data.profile) {
    return {
      ...basePayload,
      type: 'profile',
      profile: data.profile,
      recentPosts: data.recentPosts || [],
      metadata: {
        platform: data.profile.platform,
        username: data.profile.username,
        sourceUrl: data.profile.sourceUrl,
        extractionMethod: data.profile.extractionMethod,
        isOwnProfile
      }
    };
  }

  // Fallback para datos legacy
  return {
    ...basePayload,
    type: 'unknown',
    raw: data,
    isOwnProfile
  };
}

/**
 * Envía datos al backend con reintentos
 * Detecta automáticamente si el contenido es del perfil propio para activar feedback loop
 */
async function sendToApi(data: ExtractionResult): Promise<ApiResponse> {
  // Human-in-the-Loop: obtener configuración y detectar perfil propio
  const userConfig = await getUserConfig();
  const isOwnProfile = isOwnProfileContent(data, userConfig);

  if (isOwnProfile) {
    console.log('[Elena Bridge SW] Activando feedback loop ML - contenido de perfil propio');
  }

  const payload = buildPayload(data, isOwnProfile);

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= API_CONFIG.retryAttempts; attempt++) {
    try {
      console.log(`[Elena Bridge SW] Sending to API (attempt ${attempt}/${API_CONFIG.retryAttempts})`);
      console.log('[Elena Bridge SW] Payload type:', payload.type);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);

      const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.ingest}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Elena-Bridge-Version': '1.0.0',
          'X-Extension-ID': chrome.runtime.id,
          'X-Content-Type': payload.type
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

      // Human-in-the-Loop: guardar configuración de usuario
      case 'SAVE_USER_CONFIG':
        if (message.config) {
          chrome.storage.sync.set({ userConfig: message.config }, () => {
            console.log('[Elena Bridge SW] User config saved:', message.config);
            sendResponse({ success: true });
          });
          return true;
        }
        sendResponse({ success: false, error: 'No config provided' });
        break;

      // Human-in-the-Loop: obtener configuración actual
      case 'GET_USER_CONFIG':
        (async () => {
          const config = await getUserConfig();
          sendResponse({ success: true, config });
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
