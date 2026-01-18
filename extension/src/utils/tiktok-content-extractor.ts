/**
 * TikTok Content Extractor
 * Extrae datos de videos individuales de TikTok
 * Prioriza SIGI_STATE, fallback a DOM scraping
 */

import type {
  ContentData,
  MediaItem,
  ContentMetrics,
  AudioInfo,
  ExtractionResult,
  PageType
} from '../types';

// Interfaces para SIGI_STATE de TikTok
interface TikTokVideoData {
  id?: string;
  desc?: string;
  createTime?: number;
  author?: {
    uniqueId?: string;
    nickname?: string;
    avatarLarger?: string;
    avatarMedium?: string;
    verified?: boolean;
  };
  stats?: {
    playCount?: number;
    diggCount?: number;
    commentCount?: number;
    shareCount?: number;
    collectCount?: number;
  };
  video?: {
    cover?: string;
    dynamicCover?: string;
    playAddr?: string;
    downloadAddr?: string;
    duration?: number;
    width?: number;
    height?: number;
  };
  music?: {
    title?: string;
    authorName?: string;
    original?: boolean;
    playUrl?: string;
  };
  challenges?: Array<{ title: string }>;
  textExtra?: Array<{ hashtagName?: string; userId?: string; awemeId?: string }>;
}

interface TikTokSigiState {
  ItemModule?: Record<string, TikTokVideoData>;
  UserModule?: {
    users?: Record<string, {
      uniqueId?: string;
      nickname?: string;
      avatarLarger?: string;
      verified?: boolean;
    }>;
  };
}

// Parsea números con formato (ej: "1.2M", "500K")
function parseFormattedNumber(text: string | null | undefined): number | null {
  if (!text) return null;
  const cleaned = text.trim().toLowerCase().replace(/,/g, '').replace(/\s/g, '');

  if (cleaned.includes('b')) {
    return Math.round(parseFloat(cleaned.replace('b', '')) * 1_000_000_000);
  }
  if (cleaned.includes('m')) {
    return Math.round(parseFloat(cleaned.replace('m', '')) * 1_000_000);
  }
  if (cleaned.includes('k')) {
    return Math.round(parseFloat(cleaned.replace('k', '')) * 1_000);
  }

  const num = parseInt(cleaned, 10);
  return isNaN(num) ? null : num;
}

// Extrae hashtags de un caption o challenges
function extractHashtags(caption: string, challenges?: Array<{ title: string }>): string[] {
  const fromCaption = caption.match(/#[\w\u00C0-\u024F]+/g) || [];
  const fromChallenges = challenges?.map(c => `#${c.title.toLowerCase()}`) || [];
  const combined = [...new Set([...fromCaption.map(h => h.toLowerCase()), ...fromChallenges])];
  return combined;
}

// Extrae menciones de un caption
function extractMentions(caption: string): string[] {
  const matches = caption.match(/@[\w.]+/g);
  return matches ? matches.map(m => m.toLowerCase()) : [];
}

// Busca SIGI_STATE en la página
function getSigiState(): TikTokSigiState | null {
  try {
    // Método 1: window.SIGI_STATE directo
    const win = window as Window & { SIGI_STATE?: TikTokSigiState };
    if (win.SIGI_STATE?.ItemModule) {
      console.log('[Elena Bridge] TikTok SIGI_STATE found on window');
      return win.SIGI_STATE;
    }

    // Método 2: Script tag con id especial
    const sigiScript = document.getElementById('SIGI_STATE') ||
                       document.getElementById('__UNIVERSAL_DATA_FOR_REHYDRATION__');
    if (sigiScript) {
      const data = JSON.parse(sigiScript.textContent || '');
      const state = data.__DEFAULT_SCOPE__ || data;
      if (state?.ItemModule) {
        console.log('[Elena Bridge] TikTok SIGI_STATE found via script tag');
        return state;
      }
    }

    // Método 3: __NEXT_DATA__
    const nextData = document.getElementById('__NEXT_DATA__');
    if (nextData) {
      const parsed = JSON.parse(nextData.textContent || '');
      if (parsed?.props?.pageProps?.itemInfo?.itemStruct) {
        const videoData = parsed.props.pageProps.itemInfo.itemStruct;
        console.log('[Elena Bridge] TikTok data found via __NEXT_DATA__');
        return {
          ItemModule: { [videoData.id]: videoData }
        };
      }
    }

    // Método 4: Buscar en scripts inline
    const scripts = document.querySelectorAll('script:not([src])');
    for (const script of scripts) {
      const content = script.textContent || '';
      if (content.includes('ItemModule') || content.includes('SIGI_STATE')) {
        const match = content.match(/window\['SIGI_STATE'\]\s*=\s*({[\s\S]*?});/);
        if (match) {
          try {
            const state = JSON.parse(match[1]);
            if (state?.ItemModule) {
              console.log('[Elena Bridge] TikTok SIGI_STATE found via inline script');
              return state;
            }
          } catch {
            // JSON inválido
          }
        }
      }
    }

    return null;
  } catch (error) {
    console.warn('[Elena Bridge] Error getting SIGI_STATE:', error);
    return null;
  }
}

// Obtiene el ID del video de la URL
function getVideoIdFromUrl(): string | null {
  const url = window.location.href;

  // Patrón: tiktok.com/@user/video/ID
  const videoMatch = url.match(/\/video\/(\d+)/);
  if (videoMatch) return videoMatch[1];

  // Patrón alternativo: tiktok.com/t/ID
  const shortMatch = url.match(/\/t\/([^/?]+)/);
  if (shortMatch) return shortMatch[1];

  return null;
}

// Extrae datos del DOM cuando no hay SIGI_STATE
function extractFromDOM(): ContentData | null {
  try {
    const url = window.location.href;
    const videoId = getVideoIdFromUrl();

    if (!videoId) return null;

    // Autor - buscar en el header del video
    const authorLink = document.querySelector('a[href^="/@"]') as HTMLAnchorElement;
    const authorUsername = authorLink?.href?.match(/@([^/]+)/)?.[1] || '';
    const authorNameEl = document.querySelector('[data-e2e="browse-username"]') ||
                         document.querySelector('[data-e2e="video-author-uniqueid"]');
    const authorImg = document.querySelector('img[alt*="avatar"], img[alt*="profile"]') as HTMLImageElement;
    const isVerified = !!document.querySelector('[data-e2e="browse-verified"]') ||
                       !!document.querySelector('svg[class*="verified"]');

    // Caption/descripción
    const captionEl = document.querySelector('[data-e2e="browse-video-desc"]') ||
                      document.querySelector('[data-e2e="video-desc"]') ||
                      document.querySelector('h1');
    const caption = captionEl?.textContent?.trim() || null;

    // Video element
    const videoEl = document.querySelector('video') as HTMLVideoElement;
    const media: MediaItem[] = [];

    if (videoEl) {
      media.push({
        type: 'video',
        url: videoEl.src || null,
        thumbnailUrl: videoEl.poster || null,
        duration: videoEl.duration || undefined
      });
    }

    // Métricas
    const likesEl = document.querySelector('[data-e2e="like-count"], [data-e2e="browse-like-count"]');
    const commentsEl = document.querySelector('[data-e2e="comment-count"], [data-e2e="browse-comment-count"]');
    const sharesEl = document.querySelector('[data-e2e="share-count"]');
    const savesEl = document.querySelector('[data-e2e="undefined-count"]'); // Bookmarks

    const metrics: ContentMetrics = {
      likes: parseFormattedNumber(likesEl?.textContent),
      comments: parseFormattedNumber(commentsEl?.textContent),
      shares: parseFormattedNumber(sharesEl?.textContent),
      saves: parseFormattedNumber(savesEl?.textContent),
      views: null,
      plays: null
    };

    // Audio info
    const musicEl = document.querySelector('[data-e2e="browse-music"], [data-e2e="video-music"]');
    let audio: AudioInfo | null = null;

    if (musicEl) {
      const musicText = musicEl.textContent || '';
      audio = {
        title: musicText,
        artist: null,
        isOriginal: musicText.toLowerCase().includes('original'),
        audioUrl: null
      };
    }

    return {
      platform: 'tiktok',
      contentType: 'video',
      contentId: videoId,
      contentUrl: url,
      author: {
        username: authorUsername,
        displayName: authorNameEl?.textContent?.trim() || null,
        profilePicUrl: authorImg?.src || null,
        isVerified
      },
      media,
      caption,
      hashtags: caption ? extractHashtags(caption) : [],
      mentions: caption ? extractMentions(caption) : [],
      metrics,
      audio,
      postedAt: null,
      extractedAt: new Date().toISOString(),
      sourceUrl: url,
      extractionMethod: 'dom_scraping'
    };
  } catch (error) {
    console.error('[Elena Bridge] Error extracting TikTok from DOM:', error);
    return null;
  }
}

// Extrae desde SIGI_STATE
function extractFromSigiState(state: TikTokSigiState): ContentData | null {
  try {
    if (!state.ItemModule) return null;

    // Obtener el video actual (normalmente hay uno solo en ItemModule para páginas de video)
    const videoId = getVideoIdFromUrl();
    const videoIds = Object.keys(state.ItemModule);

    // Buscar el video por ID o tomar el primero
    const targetId = videoId && state.ItemModule[videoId] ? videoId : videoIds[0];
    const video = state.ItemModule[targetId];

    if (!video) return null;

    // Extraer autor
    const author = video.author || {};

    // Media
    const media: MediaItem[] = [{
      type: 'video',
      url: video.video?.playAddr || video.video?.downloadAddr || null,
      thumbnailUrl: video.video?.cover || video.video?.dynamicCover || null,
      width: video.video?.width,
      height: video.video?.height,
      duration: video.video?.duration
    }];

    // Audio
    let audio: AudioInfo | null = null;
    if (video.music) {
      audio = {
        title: video.music.title || null,
        artist: video.music.authorName || null,
        isOriginal: video.music.original || false,
        audioUrl: video.music.playUrl || null
      };
    }

    const caption = video.desc || null;

    return {
      platform: 'tiktok',
      contentType: 'video',
      contentId: video.id || targetId,
      contentUrl: window.location.href,
      author: {
        username: author.uniqueId || '',
        displayName: author.nickname || null,
        profilePicUrl: author.avatarLarger || author.avatarMedium || null,
        isVerified: author.verified || false
      },
      media,
      caption,
      hashtags: extractHashtags(caption || '', video.challenges),
      mentions: caption ? extractMentions(caption) : [],
      metrics: {
        likes: video.stats?.diggCount || null,
        comments: video.stats?.commentCount || null,
        shares: video.stats?.shareCount || null,
        saves: video.stats?.collectCount || null,
        views: null,
        plays: video.stats?.playCount || null
      },
      audio,
      postedAt: video.createTime
        ? new Date(video.createTime * 1000).toISOString()
        : null,
      extractedAt: new Date().toISOString(),
      sourceUrl: window.location.href,
      extractionMethod: 'hydrated_data'
    };
  } catch (error) {
    console.error('[Elena Bridge] Error extracting from SIGI_STATE:', error);
    return null;
  }
}

/**
 * Función principal para extraer contenido de TikTok
 */
export function extractTikTokContent(): ExtractionResult {
  console.log('[Elena Bridge] Starting TikTok content extraction...');

  // Intentar SIGI_STATE primero
  const sigiState = getSigiState();

  if (sigiState) {
    const content = extractFromSigiState(sigiState);
    if (content) {
      console.log('[Elena Bridge] Extracted via SIGI_STATE:', content.contentId);
      return {
        success: true,
        pageType: 'video',
        content
      };
    }
  }

  // Fallback a DOM
  console.log('[Elena Bridge] Falling back to DOM scraping');
  const content = extractFromDOM();

  if (content) {
    return {
      success: true,
      pageType: 'video',
      content
    };
  }

  return {
    success: false,
    pageType: 'unknown',
    error: 'No se pudo extraer el contenido. Asegúrate de estar viendo un video de TikTok.'
  };
}

/**
 * Detecta el tipo de página de TikTok
 */
export function detectTikTokPageType(): PageType {
  const url = window.location.href;

  if (url.includes('/video/')) return 'video';
  if (url.match(/tiktok\.com\/@[^/]+\/?$/)) return 'profile';

  return 'unknown';
}

/**
 * Verifica si estamos en una página de contenido de TikTok
 */
export function isTikTokContentPage(): boolean {
  const pageType = detectTikTokPageType();
  return pageType === 'video';
}
