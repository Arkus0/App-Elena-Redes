/**
 * Elena Bridge - Content Script (Injector)
 *
 * Inyecta un botón flotante discreto en páginas de contenido de Instagram y TikTok.
 * Funciona con: Posts, Reels, Videos, y Perfiles
 *
 * Comportamiento "stealth": simula lectura humana, no hace peticiones a APIs de redes sociales.
 */

import { isInstagramContentPage, extractInstagramContent, detectInstagramPageType } from '../utils/instagram-content-extractor';
import { isTikTokContentPage, extractTikTokContent, detectTikTokPageType } from '../utils/tiktok-content-extractor';
import { isInstagramProfilePage, extractInstagramProfile } from '../utils/instagram-extractor';
import { isTikTokProfilePage, extractTikTokProfile } from '../utils/tiktok-extractor';
import type { ExtractionResult, Platform, PageType, AnalysisStatus } from '../types';

// Identificadores únicos para evitar inyecciones duplicadas
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
 * Detecta el tipo de página actual
 */
function detectCurrentPageType(): PageType {
  const platform = detectPlatform();

  switch (platform) {
    case 'instagram':
      return detectInstagramPageType();
    case 'tiktok':
      return detectTikTokPageType();
    default:
      return 'unknown';
  }
}

/**
 * Verifica si estamos en una página donde podemos extraer contenido
 */
function isExtractablePage(): boolean {
  const platform = detectPlatform();
  const pageType = detectCurrentPageType();

  if (platform === 'unknown') return false;

  // Páginas de contenido individual (lo principal)
  if (pageType === 'post' || pageType === 'reel' || pageType === 'video') {
    return true;
  }

  // También permitir perfiles para análisis completo
  if (pageType === 'profile') {
    if (platform === 'instagram') return isInstagramProfilePage();
    if (platform === 'tiktok') return isTikTokProfilePage();
  }

  return false;
}

/**
 * Obtiene el texto del botón según el tipo de página
 */
function getButtonText(pageType: PageType): string {
  switch (pageType) {
    case 'post':
      return 'Guardar Post';
    case 'reel':
      return 'Guardar Reel';
    case 'video':
      return 'Guardar Video';
    case 'profile':
      return 'Analizar Perfil';
    default:
      return 'Analizar con Elena';
  }
}

/**
 * Obtiene el ícono según el tipo de página
 */
function getButtonIcon(pageType: PageType): string {
  switch (pageType) {
    case 'post':
    case 'reel':
    case 'video':
      return '\u{1F4BE}'; // Floppy disk (guardar)
    case 'profile':
      return '\u{1F50D}'; // Lupa (analizar)
    default:
      return '\u{2728}'; // Sparkles
  }
}

/**
 * Actualiza el estado visual del botón
 */
function updateButtonState(button: HTMLButtonElement, status: AnalysisStatus, pageType: PageType): void {
  currentStatus = status;

  const iconSpan = button.querySelector('.elena-icon') as HTMLSpanElement;
  const textSpan = button.querySelector('.elena-text') as HTMLSpanElement;

  button.disabled = status === 'extracting' || status === 'sending';

  switch (status) {
    case 'idle':
      iconSpan.textContent = getButtonIcon(pageType);
      textSpan.textContent = getButtonText(pageType);
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
      textSpan.textContent = 'Guardado!';
      button.className = 'elena-bridge-btn elena-success';
      setTimeout(() => updateButtonState(button, 'idle', pageType), 3000);
      break;
    case 'error':
      iconSpan.textContent = '\u{274C}'; // X roja
      textSpan.textContent = 'Error';
      button.className = 'elena-bridge-btn elena-error';
      setTimeout(() => updateButtonState(button, 'idle', pageType), 3000);
      break;
  }
}

/**
 * Extrae datos según la plataforma y tipo de página
 */
function extractData(): ExtractionResult {
  const platform = detectPlatform();
  const pageType = detectCurrentPageType();

  console.log(`[Elena Bridge] Extracting ${pageType} from ${platform}`);

  // Contenido individual (posts, reels, videos)
  if (pageType === 'post' || pageType === 'reel') {
    if (platform === 'instagram') {
      return extractInstagramContent();
    }
  }

  if (pageType === 'video') {
    if (platform === 'tiktok') {
      return extractTikTokContent();
    }
  }

  // Perfiles completos
  if (pageType === 'profile') {
    if (platform === 'instagram') {
      const result = extractInstagramProfile();
      return {
        success: result.success,
        pageType: 'profile',
        profile: result.profile || undefined,
        recentPosts: result.recentPosts,
        error: result.error
      };
    }
    if (platform === 'tiktok') {
      const result = extractTikTokProfile();
      return {
        success: result.success,
        pageType: 'profile',
        profile: result.profile || undefined,
        recentPosts: result.recentPosts,
        error: result.error
      };
    }
  }

  return {
    success: false,
    pageType: 'unknown',
    error: 'Plataforma o tipo de página no soportados'
  };
}

/**
 * Maneja el clic en el botón de análisis
 */
async function handleAnalyzeClick(event: Event): Promise<void> {
  event.preventDefault();
  event.stopPropagation();

  const button = event.currentTarget as HTMLButtonElement;
  const pageType = detectCurrentPageType();

  if (currentStatus === 'extracting' || currentStatus === 'sending') {
    console.log('[Elena Bridge] Extraction already in progress');
    return;
  }

  console.log('[Elena Bridge] Button clicked - starting extraction');
  updateButtonState(button, 'extracting', pageType);

  // Pequeña pausa para simular comportamiento humano
  await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));

  try {
    const result = extractData();

    if (!result.success) {
      console.error('[Elena Bridge] Extraction failed:', result.error);
      updateButtonState(button, 'error', pageType);
      return;
    }

    console.log('[Elena Bridge] Extraction successful');
    console.log('[Elena Bridge] Page type:', result.pageType);

    if (result.content) {
      console.log('[Elena Bridge] Content ID:', result.content.contentId);
      console.log('[Elena Bridge] Author:', result.content.author.username);
    } else if (result.profile) {
      console.log('[Elena Bridge] Profile:', result.profile.username);
    }

    updateButtonState(button, 'sending', pageType);

    // Enviar al service worker para que haga la petición a la API
    chrome.runtime.sendMessage({
      action: 'SEND_TO_API',
      data: result
    }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('[Elena Bridge] Message error:', chrome.runtime.lastError);
        updateButtonState(button, 'error', pageType);
        return;
      }

      if (response?.success) {
        console.log('[Elena Bridge] Data sent successfully. Task ID:', response.taskId);
        updateButtonState(button, 'success', pageType);
      } else {
        console.error('[Elena Bridge] API error:', response?.error);
        updateButtonState(button, 'error', pageType);
      }
    });

  } catch (error) {
    console.error('[Elena Bridge] Unexpected error:', error);
    updateButtonState(button, 'error', pageType);
  }
}

/**
 * Crea el botón flotante de Elena Bridge
 */
function createFloatingButton(): HTMLElement {
  const pageType = detectCurrentPageType();

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
  icon.textContent = getButtonIcon(pageType);

  // Texto
  const text = document.createElement('span');
  text.className = 'elena-text';
  text.textContent = getButtonText(pageType);

  button.appendChild(icon);
  button.appendChild(text);

  // Event listener
  button.addEventListener('click', handleAnalyzeClick);

  container.appendChild(button);

  return container;
}

/**
 * Actualiza el botón existente para reflejar el tipo de página actual
 */
function updateExistingButton(): void {
  const button = document.getElementById(BUTTON_ID) as HTMLButtonElement;
  if (!button) return;

  const pageType = detectCurrentPageType();
  const iconSpan = button.querySelector('.elena-icon') as HTMLSpanElement;
  const textSpan = button.querySelector('.elena-text') as HTMLSpanElement;

  if (currentStatus === 'idle') {
    iconSpan.textContent = getButtonIcon(pageType);
    textSpan.textContent = getButtonText(pageType);
  }
}

/**
 * Inyecta el botón en la página
 */
function injectButton(): void {
  // Verificar si ya existe
  if (document.getElementById(CONTAINER_ID)) {
    // Actualizar texto si cambió el tipo de página
    updateExistingButton();
    return;
  }

  // Verificar si estamos en una página válida
  if (!isExtractablePage()) {
    console.log('[Elena Bridge] Not an extractable page, skipping injection');
    return;
  }

  const platform = detectPlatform();
  const pageType = detectCurrentPageType();
  console.log(`[Elena Bridge] Injecting button for ${platform} ${pageType}`);

  const container = createFloatingButton();
  document.body.appendChild(container);

  console.log('[Elena Bridge] Button injected successfully');
}

/**
 * Elimina el botón si ya no estamos en una página válida
 */
function removeButtonIfNeeded(): void {
  const container = document.getElementById(CONTAINER_ID);

  if (container && !isExtractablePage()) {
    container.remove();
    console.log('[Elena Bridge] Button removed - not on extractable page');
  }
}

/**
 * Maneja cambios de navegación (SPA)
 */
function handleNavigation(): void {
  // Esperar a que el DOM se actualice
  setTimeout(() => {
    if (isExtractablePage()) {
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
        isContentPage: isExtractablePage(),
        pageType: detectCurrentPageType(),
        platform: detectPlatform(),
        status: currentStatus
      });
      return true;
    }

    if (message.action === 'EXTRACT_CONTENT') {
      const result = extractData();
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
