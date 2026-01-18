import type { ProfileData, ProfileStats, ExtractionResult, ExtractedPost } from '../types';

/**
 * Instagram Profile Extractor
 * Prioriza datos hidratados del servidor, fallback a scraping de DOM
 * Diseñado para comportamiento "stealth" - simula lectura humana
 */

interface InstagramHydratedData {
  user?: {
    username?: string;
    full_name?: string;
    biography?: string;
    profile_pic_url_hd?: string;
    profile_pic_url?: string;
    is_verified?: boolean;
    is_private?: boolean;
    edge_followed_by?: { count: number };
    edge_follow?: { count: number };
    edge_owner_to_timeline_media?: {
      count: number;
      edges?: Array<{
        node: {
          id: string;
          __typename: string;
          display_url: string;
          edge_media_to_caption?: { edges: Array<{ node: { text: string } }> };
          edge_liked_by?: { count: number };
          edge_media_to_comment?: { count: number };
          taken_at_timestamp?: number;
          video_view_count?: number;
        };
      }>;
    };
  };
}

// Intenta obtener datos hidratados de Instagram (window._sharedData o __additionalDataLoaded)
function getHydratedData(): InstagramHydratedData | null {
  try {
    // Método 1: _sharedData (legacy, pero aún funciona en algunos casos)
    const win = window as Window & {
      _sharedData?: { entry_data?: { ProfilePage?: Array<{ graphql?: InstagramHydratedData }> } };
      __additionalDataLoaded?: (path: string, data: InstagramHydratedData) => void;
    };

    if (win._sharedData?.entry_data?.ProfilePage?.[0]?.graphql) {
      console.log('[Elena Bridge] Hydrated data found via _sharedData');
      return win._sharedData.entry_data.ProfilePage[0].graphql;
    }

    // Método 2: Buscar en scripts con type="application/json"
    const scripts = document.querySelectorAll('script[type="application/json"]');
    for (const script of scripts) {
      try {
        const data = JSON.parse(script.textContent || '');
        if (data?.user?.username || data?.graphql?.user?.username) {
          console.log('[Elena Bridge] Hydrated data found via JSON script');
          return data.graphql || data;
        }
      } catch {
        // Script no es JSON válido, continuar
      }
    }

    // Método 3: Buscar en window.__PRELOADED_STATE__
    const preloadedState = (window as Window & { __PRELOADED_STATE__?: string }).__PRELOADED_STATE__;
    if (preloadedState) {
      try {
        const parsed = JSON.parse(preloadedState);
        if (parsed?.users?.users) {
          const username = Object.keys(parsed.users.users)[0];
          if (username) {
            console.log('[Elena Bridge] Hydrated data found via __PRELOADED_STATE__');
            return { user: parsed.users.users[username] };
          }
        }
      } catch {
        // No es JSON válido
      }
    }

    return null;
  } catch (error) {
    console.warn('[Elena Bridge] Error getting hydrated data:', error);
    return null;
  }
}

// Parsea números con formato (ej: "1.2M", "500K", "10,234")
function parseFormattedNumber(text: string | null | undefined): number | null {
  if (!text) return null;

  const cleaned = text.trim().toLowerCase().replace(/,/g, '');

  if (cleaned.includes('m')) {
    return Math.round(parseFloat(cleaned.replace('m', '')) * 1_000_000);
  }
  if (cleaned.includes('k')) {
    return Math.round(parseFloat(cleaned.replace('k', '')) * 1_000);
  }

  const num = parseInt(cleaned, 10);
  return isNaN(num) ? null : num;
}

// Extrae datos del DOM cuando no hay datos hidratados
function extractFromDOM(): ProfileData | null {
  try {
    // Detectar si estamos en una página de perfil
    const pathMatch = window.location.pathname.match(/^\/([^/]+)\/?$/);
    if (!pathMatch) return null;

    const username = pathMatch[1];

    // Excluir rutas que no son perfiles
    const excludedPaths = ['explore', 'reels', 'direct', 'accounts', 'stories', 'p', 'tv'];
    if (excludedPaths.includes(username)) return null;

    // Selectores actualizados para Instagram (pueden cambiar con updates de IG)
    const headerSection = document.querySelector('header section');

    // Display name (puede estar en h2 o en meta tags)
    let displayName = document.querySelector('header h2')?.textContent?.trim() ||
                      document.querySelector('meta[property="og:title"]')?.getAttribute('content')?.split('(')[0]?.trim() ||
                      null;

    // Bio
    const bioElement = document.querySelector('header section > div:last-child > span') ||
                       document.querySelector('[data-testid="user-biography"]') ||
                       document.querySelector('header section div > span > span');
    const bio = bioElement?.textContent?.trim() || null;

    // Verificado
    const isVerified = !!document.querySelector('header [aria-label="Verified"]') ||
                       !!document.querySelector('header svg[aria-label*="erified"]');

    // Privado
    const isPrivate = !!document.querySelector('[data-testid="private-account-indicator"]') ||
                      document.body.innerText.includes('This Account is Private') ||
                      document.body.innerText.includes('Esta cuenta es privada');

    // Foto de perfil
    const profilePic = document.querySelector('header img[alt*="profile"]') as HTMLImageElement ||
                       document.querySelector('header img') as HTMLImageElement;
    const profilePicUrl = profilePic?.src || null;

    // Estadísticas (followers, following, posts)
    const statsElements = headerSection?.querySelectorAll('ul li') || [];
    const stats: ProfileStats = {
      posts: null,
      followers: null,
      following: null
    };

    statsElements.forEach((el, index) => {
      const text = el.textContent || '';
      const num = parseFormattedNumber(text);

      if (index === 0 || text.toLowerCase().includes('post')) {
        stats.posts = num;
      } else if (index === 1 || text.toLowerCase().includes('follower')) {
        stats.followers = num;
      } else if (index === 2 || text.toLowerCase().includes('following')) {
        stats.following = num;
      }
    });

    // Alternativa: buscar por aria-labels específicos
    if (!stats.followers) {
      const followersEl = document.querySelector('[href$="/followers/"] span') ||
                          document.querySelector('a[href*="followers"] span');
      stats.followers = parseFormattedNumber(followersEl?.textContent);
    }

    if (!stats.following) {
      const followingEl = document.querySelector('[href$="/following/"] span') ||
                          document.querySelector('a[href*="following"] span');
      stats.following = parseFormattedNumber(followingEl?.textContent);
    }

    return {
      platform: 'instagram',
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

// Extrae posts recientes del DOM
function extractRecentPosts(): ExtractedPost[] {
  const posts: ExtractedPost[] = [];

  try {
    // Buscar grid de posts
    const postLinks = document.querySelectorAll('article a[href*="/p/"], main a[href*="/p/"]');

    postLinks.forEach((link, index) => {
      if (index >= 12) return; // Limitar a 12 posts

      const img = link.querySelector('img') as HTMLImageElement;
      const href = link.getAttribute('href') || '';
      const postId = href.match(/\/p\/([^/]+)/)?.[1] || `post_${index}`;

      // Detectar tipo (video tiene overlay de play)
      const hasVideoIndicator = !!link.querySelector('svg[aria-label*="Video"]') ||
                                !!link.querySelector('[aria-label*="Reel"]') ||
                                !!link.querySelector('[aria-label*="video"]');

      const hasCarouselIndicator = !!link.querySelector('svg[aria-label*="Carousel"]') ||
                                   !!link.querySelector('[aria-label*="carousel"]');

      let type: ExtractedPost['type'] = 'image';
      if (hasCarouselIndicator) type = 'carousel';
      else if (hasVideoIndicator) type = 'video';

      posts.push({
        id: postId,
        type,
        thumbnailUrl: img?.src || null,
        caption: null, // Requeriría navegar al post
        likes: null,
        comments: null,
        timestamp: null
      });
    });
  } catch (error) {
    console.warn('[Elena Bridge] Error extracting posts:', error);
  }

  return posts;
}

// Función principal de extracción para Instagram
export function extractInstagramProfile(): ExtractionResult {
  console.log('[Elena Bridge] Starting Instagram extraction...');

  // Intentar datos hidratados primero
  const hydratedData = getHydratedData();

  if (hydratedData?.user) {
    const user = hydratedData.user;
    console.log('[Elena Bridge] Using hydrated data for:', user.username);

    const posts: ExtractedPost[] = [];

    // Extraer posts de datos hidratados
    user.edge_owner_to_timeline_media?.edges?.forEach((edge, index) => {
      if (index >= 12) return;

      const node = edge.node;
      let type: ExtractedPost['type'] = 'image';

      if (node.__typename === 'GraphVideo') type = 'video';
      else if (node.__typename === 'GraphSidecar') type = 'carousel';

      posts.push({
        id: node.id,
        type,
        thumbnailUrl: node.display_url,
        caption: node.edge_media_to_caption?.edges?.[0]?.node?.text || null,
        likes: node.edge_liked_by?.count || null,
        comments: node.edge_media_to_comment?.count || null,
        views: node.video_view_count || null,
        timestamp: node.taken_at_timestamp
          ? new Date(node.taken_at_timestamp * 1000).toISOString()
          : null
      });
    });

    return {
      success: true,
      data: {
        platform: 'instagram',
        username: user.username || '',
        displayName: user.full_name || null,
        bio: user.biography || null,
        profilePicUrl: user.profile_pic_url_hd || user.profile_pic_url || null,
        isVerified: user.is_verified || false,
        isPrivate: user.is_private || false,
        stats: {
          followers: user.edge_followed_by?.count || null,
          following: user.edge_follow?.count || null,
          posts: user.edge_owner_to_timeline_media?.count || null
        },
        extractedAt: new Date().toISOString(),
        sourceUrl: window.location.href,
        extractionMethod: 'hydrated_data'
      },
      recentPosts: posts
    };
  }

  // Fallback a scraping de DOM
  console.log('[Elena Bridge] Falling back to DOM scraping');
  const domData = extractFromDOM();

  if (domData) {
    return {
      success: true,
      data: domData,
      recentPosts: extractRecentPosts()
    };
  }

  return {
    success: false,
    data: null,
    error: 'No se pudo extraer datos del perfil. Asegúrate de estar en una página de perfil de Instagram.'
  };
}

// Verifica si estamos en una URL de perfil de Instagram
export function isInstagramProfilePage(): boolean {
  const url = window.location.href;

  if (!url.includes('instagram.com')) return false;

  // Patrón: instagram.com/username (sin paths adicionales)
  const pathMatch = window.location.pathname.match(/^\/([^/]+)\/?$/);
  if (!pathMatch) return false;

  const username = pathMatch[1];
  const excludedPaths = [
    'explore', 'reels', 'direct', 'accounts', 'stories',
    'p', 'tv', 'reel', 'about', 'privacy', 'terms', 'help'
  ];

  return !excludedPaths.includes(username);
}
