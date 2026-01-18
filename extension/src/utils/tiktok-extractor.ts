import type { ProfileData, ProfileStats, ExtractionResult, ExtractedPost } from '../types';

/**
 * TikTok Profile Extractor
 * Prioriza SIGI_STATE (datos hidratados), fallback a scraping de DOM
 * Diseñado para comportamiento "stealth"
 */

interface TikTokUserInfo {
  uniqueId?: string;
  nickname?: string;
  signature?: string;
  avatarLarger?: string;
  avatarMedium?: string;
  verified?: boolean;
  privateAccount?: boolean;
  followerCount?: number;
  followingCount?: number;
  heartCount?: number;
  videoCount?: number;
}

interface TikTokVideoItem {
  id?: string;
  desc?: string;
  createTime?: number;
  stats?: {
    playCount?: number;
    diggCount?: number;
    commentCount?: number;
    shareCount?: number;
  };
  video?: {
    cover?: string;
    dynamicCover?: string;
  };
}

interface TikTokSigiState {
  UserModule?: {
    users?: Record<string, TikTokUserInfo>;
  };
  ItemModule?: Record<string, TikTokVideoItem>;
  ItemList?: {
    user?: {
      list?: string[];
    };
  };
}

// Intenta obtener SIGI_STATE de TikTok
function getSigiState(): TikTokSigiState | null {
  try {
    // Método 1: window.SIGI_STATE
    const win = window as Window & { SIGI_STATE?: TikTokSigiState };
    if (win.SIGI_STATE) {
      console.log('[Elena Bridge] SIGI_STATE found directly on window');
      return win.SIGI_STATE;
    }

    // Método 2: Buscar en scripts con id="SIGI_STATE" o __NEXT_DATA__
    const sigiScript = document.getElementById('SIGI_STATE') ||
                       document.getElementById('__UNIVERSAL_DATA_FOR_REHYDRATION__');

    if (sigiScript) {
      const data = JSON.parse(sigiScript.textContent || '');
      console.log('[Elena Bridge] SIGI_STATE found via script tag');
      return data.__DEFAULT_SCOPE__ || data;
    }

    // Método 3: Buscar en __NEXT_DATA__
    const nextData = document.getElementById('__NEXT_DATA__');
    if (nextData) {
      const parsed = JSON.parse(nextData.textContent || '');
      if (parsed?.props?.pageProps?.userInfo) {
        console.log('[Elena Bridge] User data found via __NEXT_DATA__');
        return {
          UserModule: {
            users: {
              [parsed.props.pageProps.userInfo.user?.uniqueId || 'user']:
                parsed.props.pageProps.userInfo.user
            }
          }
        };
      }
    }

    // Método 4: Buscar scripts inline con SIGI_STATE
    const scripts = document.querySelectorAll('script:not([src])');
    for (const script of scripts) {
      const content = script.textContent || '';
      if (content.includes('SIGI_STATE') || content.includes('UserModule')) {
        // Intentar extraer JSON
        const match = content.match(/window\['SIGI_STATE'\]\s*=\s*({[\s\S]*?});/);
        if (match) {
          try {
            const data = JSON.parse(match[1]);
            console.log('[Elena Bridge] SIGI_STATE found via inline script');
            return data;
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

// Parsea números con formato (ej: "1.2M", "500K")
function parseFormattedNumber(text: string | null | undefined): number | null {
  if (!text) return null;

  const cleaned = text.trim().toLowerCase().replace(/,/g, '');

  if (cleaned.includes('m')) {
    return Math.round(parseFloat(cleaned.replace('m', '')) * 1_000_000);
  }
  if (cleaned.includes('k')) {
    return Math.round(parseFloat(cleaned.replace('k', '')) * 1_000);
  }
  if (cleaned.includes('b')) {
    return Math.round(parseFloat(cleaned.replace('b', '')) * 1_000_000_000);
  }

  const num = parseInt(cleaned, 10);
  return isNaN(num) ? null : num;
}

// Extrae datos del DOM cuando no hay SIGI_STATE
function extractFromDOM(): ProfileData | null {
  try {
    // Detectar username de la URL
    const pathMatch = window.location.pathname.match(/^\/@([^/]+)/);
    if (!pathMatch) return null;

    const username = pathMatch[1];

    // Selectores para TikTok (pueden cambiar con updates)
    // Display name
    const displayName = document.querySelector('[data-e2e="user-subtitle"]')?.textContent?.trim() ||
                        document.querySelector('h1[data-e2e="user-title"]')?.textContent?.trim() ||
                        document.querySelector('h2[data-e2e="user-subtitle"]')?.textContent?.trim() ||
                        null;

    // Bio
    const bio = document.querySelector('[data-e2e="user-bio"]')?.textContent?.trim() ||
                document.querySelector('h2[data-e2e="user-bio"]')?.textContent?.trim() ||
                null;

    // Verificado
    const isVerified = !!document.querySelector('[data-e2e="verified-badge"]') ||
                       !!document.querySelector('svg[class*="verified"]') ||
                       !!document.querySelector('[class*="verified"]');

    // Privado (TikTok raramente tiene cuentas privadas visibles)
    const isPrivate = false;

    // Foto de perfil
    const profilePic = document.querySelector('[data-e2e="user-avatar"] img') as HTMLImageElement ||
                       document.querySelector('img[alt*="avatar"]') as HTMLImageElement;
    const profilePicUrl = profilePic?.src || null;

    // Estadísticas
    const stats: ProfileStats = {
      following: null,
      followers: null,
      likes: null,
      posts: null
    };

    // Buscar contadores con data-e2e
    const followingEl = document.querySelector('[data-e2e="following-count"]');
    const followersEl = document.querySelector('[data-e2e="followers-count"]');
    const likesEl = document.querySelector('[data-e2e="likes-count"]');

    stats.following = parseFormattedNumber(followingEl?.textContent);
    stats.followers = parseFormattedNumber(followersEl?.textContent);
    stats.likes = parseFormattedNumber(likesEl?.textContent);

    // Contar videos visibles
    const videoElements = document.querySelectorAll('[data-e2e="user-post-item"]');
    if (videoElements.length > 0) {
      // Esto es una estimación, TikTok carga de forma lazy
      stats.posts = videoElements.length;
    }

    return {
      platform: 'tiktok',
      username,
      displayName,
      bio,
      profilePicUrl,
      isVerified,
      isPrivate,
      stats,
      extractedAt: new Date().toISOString(),
      sourceUrl: window.location.href,
      extractionMethod: 'dom_scraping'
    };
  } catch (error) {
    console.error('[Elena Bridge] Error extracting from DOM:', error);
    return null;
  }
}

// Extrae posts recientes del SIGI_STATE o DOM
function extractRecentPosts(sigiState: TikTokSigiState | null): ExtractedPost[] {
  const posts: ExtractedPost[] = [];

  try {
    // Intentar desde SIGI_STATE
    if (sigiState?.ItemModule) {
      const videoIds = sigiState.ItemList?.user?.list || Object.keys(sigiState.ItemModule);

      videoIds.slice(0, 12).forEach(id => {
        const video = sigiState.ItemModule?.[id];
        if (!video) return;

        posts.push({
          id: video.id || id,
          type: 'video',
          thumbnailUrl: video.video?.cover || video.video?.dynamicCover || null,
          caption: video.desc || null,
          likes: video.stats?.diggCount || null,
          comments: video.stats?.commentCount || null,
          shares: video.stats?.shareCount || null,
          views: video.stats?.playCount || null,
          timestamp: video.createTime
            ? new Date(video.createTime * 1000).toISOString()
            : null
        });
      });

      return posts;
    }

    // Fallback a DOM
    const videoElements = document.querySelectorAll('[data-e2e="user-post-item"]');

    videoElements.forEach((el, index) => {
      if (index >= 12) return;

      const link = el.querySelector('a') as HTMLAnchorElement;
      const img = el.querySelector('img') as HTMLImageElement;

      const videoId = link?.href?.match(/\/video\/(\d+)/)?.[1] || `video_${index}`;

      // Buscar stats en el elemento
      const viewsText = el.querySelector('[data-e2e="video-views"]')?.textContent;

      posts.push({
        id: videoId,
        type: 'video',
        thumbnailUrl: img?.src || null,
        caption: null,
        likes: null,
        comments: null,
        shares: null,
        views: parseFormattedNumber(viewsText),
        timestamp: null
      });
    });
  } catch (error) {
    console.warn('[Elena Bridge] Error extracting posts:', error);
  }

  return posts;
}

// Función principal de extracción para TikTok
export function extractTikTokProfile(): ExtractionResult {
  console.log('[Elena Bridge] Starting TikTok extraction...');

  const sigiState = getSigiState();

  if (sigiState?.UserModule?.users) {
    const users = sigiState.UserModule.users;
    const username = Object.keys(users)[0];
    const user = users[username];

    if (user) {
      console.log('[Elena Bridge] Using SIGI_STATE for:', user.uniqueId);

      return {
        success: true,
        data: {
          platform: 'tiktok',
          username: user.uniqueId || username,
          displayName: user.nickname || null,
          bio: user.signature || null,
          profilePicUrl: user.avatarLarger || user.avatarMedium || null,
          isVerified: user.verified || false,
          isPrivate: user.privateAccount || false,
          stats: {
            followers: user.followerCount || null,
            following: user.followingCount || null,
            likes: user.heartCount || null,
            posts: user.videoCount || null
          },
          extractedAt: new Date().toISOString(),
          sourceUrl: window.location.href,
          extractionMethod: 'hydrated_data'
        },
        recentPosts: extractRecentPosts(sigiState)
      };
    }
  }

  // Fallback a scraping de DOM
  console.log('[Elena Bridge] Falling back to DOM scraping');
  const domData = extractFromDOM();

  if (domData) {
    return {
      success: true,
      data: domData,
      recentPosts: extractRecentPosts(null)
    };
  }

  return {
    success: false,
    data: null,
    error: 'No se pudo extraer datos del perfil. Asegúrate de estar en una página de perfil de TikTok.'
  };
}

// Verifica si estamos en una URL de perfil de TikTok
export function isTikTokProfilePage(): boolean {
  const url = window.location.href;

  if (!url.includes('tiktok.com')) return false;

  // Patrón: tiktok.com/@username
  const pathMatch = window.location.pathname.match(/^\/@([^/]+)\/?$/);

  return !!pathMatch;
}
