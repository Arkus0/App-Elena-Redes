/**
 * Elena Bridge - Content Script (Injector)
 *
 * Inyecta un botón flotante discreto en perfiles de Instagram y TikTok.
 * Comportamiento "stealth": simula lectura humana, no hace peticiones a APIs de redes sociales.
 */

import { isInstagramProfilePage, extractInstagramProfile } from '../utils/instagram-extractor';
import { isTikTokProfilePage, extractTikTokProfile } from '../utils/tiktok-extractor';
import type { ExtractionResult, Platform, AnalysisStatus } from '../types';

// Identificador único para evitar inyecciones duplicadas
const BUTTON_ID = 'elena-bridge-analyze-btn';
const CONTAINER_ID = 'elena-bridge-container';

// Estado del análisis
let currentStatus: AnalysisStatus = 'idle';

/**
 * Detecta la plataforma actual basándose en la URL
 */
function detectPlatform(): Platform {
  const url = window.location.href;

  if (url.includes('instagram.com')) return 'instagram';
  if (url.includes('tiktok.com')) return 'tiktok';

  return 'unknown';
}

/**
 * Verifica si estamos en una página de perfil válida
 */
function isProfilePage(): boolean {
  const platform = detectPlatform();

  switch (platform) {
    case 'instagram':
      return isInstagramProfilePage();
    case 'tiktok':
      return isTikTokProfilePage();
    default:
      return false;
  }
}

/**
 * Actualiza el estado visual del botón
 */
function updateButtonState(button: HTMLButtonElement, status: AnalysisStatus): void {
  currentStatus = status;

  const iconSpan = button.querySelector('.elena-icon') as HTMLSpanElement;
  const textSpan = button.querySelector('.elena-text') as HTMLSpanElement;

  button.disabled = status === 'extracting' || status === 'sending';

  switch (status) {
    case 'idle':
      iconSpan.textContent = '\u{1F50D}'; // Lupa
      textSpan.textContent = 'Analizar con Elena';
      button.className = 'elena-bridge-btn';
      break;
    case 'extracting':
      iconSpan.textContent = '\u{23F3}'; // Reloj
      textSpan.textContent = 'Extrayendo...';
      button.className = 'elena-bridge-btn elena-loading';
      break;
    case 'sending':
      iconSpan.textContent = '\u{1F4E4}'; // Envío
      textSpan.textContent = 'Enviando...';
      button.className = 'elena-bridge-btn elena-loading';
      break;
    case 'success':
      iconSpan.textContent = '\u{2705}'; // Check verde
      textSpan.textContent = 'Enviado';
      button.className = 'elena-bridge-btn elena-success';
      // Volver a estado idle después de 3 segundos
      setTimeout(() => updateButtonState(button, 'idle'), 3000);
      break;
    case 'error':
      iconSpan.textContent = '\u{274C}'; // X roja
      textSpan.textContent = 'Error';
      button.className = 'elena-bridge-btn elena-error';
      // Volver a estado idle después de 3 segundos
      setTimeout(() => updateButtonState(button, 'idle'), 3000);
      break;
  }
}

/**
 * Extrae datos del perfil según la plataforma
 */
function extractProfileData(): ExtractionResult {
  const platform = detectPlatform();

  switch (platform) {
    case 'instagram':
      return extractInstagramProfile();
    case 'tiktok':
      return extractTikTokProfile();
    default:
      return {
        success: false,
        data: null,
        error: 'Plataforma no soportada'
      };
  }
}

/**
 * Maneja el clic en el botón de análisis
 */
async function handleAnalyzeClick(event: Event): Promise<void> {
  event.preventDefault();
  event.stopPropagation();

  const button = event.currentTarget as HTMLButtonElement;

  if (currentStatus === 'extracting' || currentStatus === 'sending') {
    console.log('[Elena Bridge] Analysis already in progress');
    return;
  }

  console.log('[Elena Bridge] Button clicked - starting extraction');
  updateButtonState(button, 'extracting');

  // Pequeña pausa para simular comportamiento humano
  await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 500));

  try {
    const result = extractProfileData();

    if (!result.success || !result.data) {
      console.error('[Elena Bridge] Extraction failed:', result.error);
      updateButtonState(button, 'error');
      return;
    }

    console.log('[Elena Bridge] Extraction successful:', result.data.username);
    updateButtonState(button, 'sending');

    // Enviar al service worker para que haga la petición a la API
    chrome.runtime.sendMessage({
      action: 'SEND_TO_API',
      data: result
    }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('[Elena Bridge] Message error:', chrome.runtime.lastError);
        updateButtonState(button, 'error');
        return;
      }

      if (response?.success) {
        console.log('[Elena Bridge] Data sent successfully');
        updateButtonState(button, 'success');
      } else {
        console.error('[Elena Bridge] API error:', response?.error);
        updateButtonState(button, 'error');
      }
    });

  } catch (error) {
    console.error('[Elena Bridge] Unexpected error:', error);
    updateButtonState(button, 'error');
  }
}

/**
 * Crea el botón flotante de Elena Bridge
 */
function createFloatingButton(): HTMLElement {
  // Contenedor principal
  const container = document.createElement('div');
  container.id = CONTAINER_ID;
  container.className = 'elena-bridge-container';

  // Botón
  const button = document.createElement('button');
  button.id = BUTTON_ID;
  button.className = 'elena-bridge-btn';
  button.type = 'button';

  // Icono
  const icon = document.createElement('span');
  icon.className = 'elena-icon';
  icon.textContent = '\u{1F50D}'; // Lupa

  // Texto
  const text = document.createElement('span');
  text.className = 'elena-text';
  text.textContent = 'Analizar con Elena';

  button.appendChild(icon);
  button.appendChild(text);

  // Event listener
  button.addEventListener('click', handleAnalyzeClick);

  container.appendChild(button);

  return container;
}

/**
 * Inyecta el botón en la página
 */
function injectButton(): void {
  // Verificar si ya existe
  if (document.getElementById(CONTAINER_ID)) {
    console.log('[Elena Bridge] Button already exists');
    return;
  }

  // Verificar si estamos en una página de perfil
  if (!isProfilePage()) {
    console.log('[Elena Bridge] Not a profile page, skipping injection');
    return;
  }

  const platform = detectPlatform();
  console.log(`[Elena Bridge] Injecting button for ${platform} profile`);

  const container = createFloatingButton();
  document.body.appendChild(container);

  console.log('[Elena Bridge] Button injected successfully');
}

/**
 * Elimina el botón si ya no estamos en una página de perfil
 */
function removeButtonIfNeeded(): void {
  const container = document.getElementById(CONTAINER_ID);

  if (container && !isProfilePage()) {
    container.remove();
    console.log('[Elena Bridge] Button removed - no longer on profile page');
  }
}

/**
 * Maneja cambios de navegación (SPA)
 */
function handleNavigation(): void {
  // Esperar a que el DOM se actualice
  setTimeout(() => {
    if (isProfilePage()) {
      injectButton();
    } else {
      removeButtonIfNeeded();
    }
  }, 1000);
}

/**
 * Inicializa el content script
 */
function init(): void {
  console.log('[Elena Bridge] Content script loaded');

  // Inyectar botón inicial (con delay para asegurar que el DOM esté listo)
  setTimeout(injectButton, 1500);

  // Observar cambios de URL (para SPAs)
  let lastUrl = window.location.href;

  // MutationObserver para detectar cambios de navegación
  const observer = new MutationObserver(() => {
    if (window.location.href !== lastUrl) {
      lastUrl = window.location.href;
      console.log('[Elena Bridge] URL changed:', lastUrl);
      handleNavigation();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  // También escuchar popstate para navegación con botones del navegador
  window.addEventListener('popstate', handleNavigation);

  // Escuchar mensajes del popup
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.action === 'GET_STATUS') {
      sendResponse({
        isProfilePage: isProfilePage(),
        platform: detectPlatform(),
        status: currentStatus
      });
      return true;
    }

    if (message.action === 'EXTRACT_PROFILE') {
      const result = extractProfileData();
      sendResponse(result);
      return true;
    }

    return false;
  });
}

// Iniciar cuando el DOM esté listo
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
