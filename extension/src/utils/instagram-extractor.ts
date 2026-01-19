import type { ProfileData, ProfileStats, ExtractionResult, ExtractedPost } from '../types';
import { DomLayoutChangedError } from './errors';

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

interface LocationLike {
  href: string;
  pathname: string;
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

/**
 * Encuentra un elemento cuyo contenido de texto coincida con alguno de los patrones.
 * Devuelve el elemento más profundo que contiene el texto.
 */
function findByTextContent(root: Element, patterns: (string | RegExp)[], tag?: string): Element | null {
  const candidates: Element[] = [];

  function traverse(el: Element) {
    if (tag && el.tagName.toLowerCase() !== tag.toLowerCase()) {
      // Continue but don't match this element if tag mismatch (unless we only want to match children?)
      // Actually, standard traversal: check current, then children.
    }

    // Check if current element matches
    let matches = false;
    const text = el.textContent || '';

    // Optimization: if text is empty, skip
    if (!text.trim()) return;

    // Check patterns
    for (const pattern of patterns) {
        if (typeof pattern === 'string') {
            if (text.toLowerCase().includes(pattern.toLowerCase())) {
                matches = true;
                break;
            }
        } else {
            if (pattern.test(text)) {
                matches = true;
                break;
            }
        }
    }

    // If matches, checks if any children match. If NO children match, this is the deepest match.
    // Or we can collect all matches and sort by depth/length.
    if (matches && (!tag || el.tagName.toLowerCase() === tag.toLowerCase())) {
        candidates.push(el);
    }

    for (const child of Array.from(el.children)) {
      traverse(child);
    }
  }

  traverse(root);

  // Return the candidate with shortest text content (likely the most specific element)
  if (candidates.length === 0) return null;
  return candidates.reduce((prev, curr) =>
    (prev.textContent?.length || Infinity) < (curr.textContent?.length || Infinity) ? prev : curr
  );
}

/**
 * Busca una métrica numérica asociada a un label semántico (ej: "Followers").
 * Busca en el padre o hermanos del elemento que contiene el label.
 */
function findMetricByLabel(root: Element, labels: string[]): number | null {
  const labelEl = findByTextContent(root, labels);
  if (!labelEl) return null;

  // 1. Check parent text
  const parent = labelEl.parentElement;
  if (parent) {
      // Try to extract number from parent text (excluding the label text if possible, but parseFormattedNumber handles it)
      const num = parseFormattedNumber(parent.textContent);
      if (num !== null) return num;
  }

  // 2. Check previous sibling
  const prev = labelEl.previousElementSibling;
  if (prev) {
      const num = parseFormattedNumber(prev.textContent);
      if (num !== null) return num;
  }

  // 3. Check inside the label element itself (maybe number is a child or prefix)
  const numSelf = parseFormattedNumber(labelEl.textContent);
  if (numSelf !== null) return numSelf;

  return null;
}


// Extrae datos del DOM cuando no hay datos hidratados
function extractFromDOM(location: LocationLike = window.location): ProfileData | null {
  try {
    // Detectar si estamos en una página de perfil
    const pathMatch = location.pathname.match(/^\/([^/]+)\/?$/);
    if (!pathMatch) return null;

    const username = pathMatch[1];

    // Excluir rutas que no son perfiles
    const excludedPaths = ['explore', 'reels', 'direct', 'accounts', 'stories', 'p', 'tv'];
    if (excludedPaths.includes(username)) return null;

    // Scoped Semantic Traversal
    const header = document.querySelector('header');

    if (!header) {
        // If no header, maybe layout changed drastically or not fully loaded
        // But we can try legacy global selectors as last resort?
        // For now, if no header, likely not profile page or error.
        // Let's assume we need header for semantic search.
        throw new DomLayoutChangedError("Header element not found");
    }

    // Display name
    let displayName = document.querySelector('header h2')?.textContent?.trim() ||
                      document.querySelector('meta[property="og:title"]')?.getAttribute('content')?.split('(')[0]?.trim() ||
                      null;

    // --- Stats (Semantic) ---
    const stats: ProfileStats = {
      posts: null,
      followers: null,
      following: null
    };

    // Dictionary
    const labels = {
        posts: ['posts', 'publicaciones', 'publicações'],
        followers: ['followers', 'seguidores'], // 'seguidores' is same for ES/PT
        following: ['following', 'seguidos', 'seguindo', 'a seguir']
    };

    stats.posts = findMetricByLabel(header, labels.posts);
    stats.followers = findMetricByLabel(header, labels.followers);
    stats.following = findMetricByLabel(header, labels.following);

    // --- Bio (Semantic/Heuristic) ---
    // Heuristic: Longest text in header NOT username or buttons
    let bio: string | null = null;

    // Explicit selector (Primary)
    const bioExplicit = header.querySelector('[data-testid="user-biography"]');
    if (bioExplicit) {
        bio = bioExplicit.textContent?.trim() || null;
    } else {
        // Fallback Heuristic
        const candidates = Array.from(header.querySelectorAll('*'))
            .filter(el => {
                // Must be a leaf node or close to it (text node container)
                // Filter out buttons, links (unless it's the bio link container? no, bio text usually is span/div)
                if (el.closest('button') || el.closest('a')) return false;

                // Exclude stats
                const text = el.textContent || '';
                if (labels.posts.some(l => text.toLowerCase().includes(l))) return false;
                if (labels.followers.some(l => text.toLowerCase().includes(l))) return false;
                if (labels.following.some(l => text.toLowerCase().includes(l))) return false;

                // Exclude username
                if (text.includes(username)) return false;

                return true;
            });

        // Sort by length desc
        candidates.sort((a, b) => (b.textContent?.length || 0) - (a.textContent?.length || 0));

        if (candidates.length > 0) {
            bio = candidates[0].textContent?.trim() || null;
        }
    }

    // --- Verified (Semantic) ---
    const isVerified = !!header.querySelector('[aria-label="Verified"]') ||
                       !!header.querySelector('svg[aria-label*="erified"]');

    // --- Profile Pic ---
    const profilePic = header.querySelector('img[alt*="profile"]') as HTMLImageElement ||
                       header.querySelector('img') as HTMLImageElement;
    const profilePicUrl = profilePic?.src || null;

    // --- Private ---
    const isPrivate = !!document.querySelector('[data-testid="private-account-indicator"]') ||
                      (document.body.textContent || '').includes('This Account is Private') ||
                      (document.body.textContent || '').includes('Esta cuenta es privada');


    // --- Fallback Protection ---
    // If stats are missing, try rigid selectors
    if (stats.followers === null || stats.posts === null) {
         console.warn('[Elena Bridge] Semantic extraction failed for stats, trying rigid selectors');

         const statsElements = header.querySelectorAll('ul li');
         statsElements.forEach((el, index) => {
            const text = el.textContent || '';
            const num = parseFormattedNumber(text);
            if (index === 0) stats.posts = num;
            if (index === 1) stats.followers = num;
            if (index === 2) stats.following = num;
         });
    }

    // Final Check
    if (stats.followers === null && stats.posts === null && !bio) {
        throw new DomLayoutChangedError("Failed to extract essential profile data (stats/bio)");
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
      sourceUrl: location.href,
      extractionMethod: 'dom_scraping'
    };
  } catch (error) {
    if (error instanceof DomLayoutChangedError) throw error;
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
export function extractInstagramProfile(location: LocationLike = window.location): ExtractionResult {
  console.log('[Elena Bridge] Starting Instagram extraction...');

  try {
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
        pageType: 'profile',
        profile: {
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
          sourceUrl: location.href,
          extractionMethod: 'hydrated_data'
        },
        recentPosts: posts
      };
    }

    // Fallback a scraping de DOM
    console.log('[Elena Bridge] Falling back to DOM scraping');
    const domData = extractFromDOM(location);

    if (domData) {
      return {
        success: true,
        pageType: 'profile',
        profile: domData,
        recentPosts: extractRecentPosts()
      };
    }
  } catch (error) {
     if (error instanceof DomLayoutChangedError) throw error;
     // Other errors?
  }

  return {
    success: false,
    pageType: 'unknown',
    data: null,
    error: 'No se pudo extraer datos del perfil. Asegúrate de estar en una página de perfil de Instagram.'
  } as any;
}

// Verifica si estamos en una URL de perfil de Instagram
export function isInstagramProfilePage(location: LocationLike = window.location): boolean {
  const url = location.href;

  if (!url.includes('instagram.com')) return false;

  // Patrón: instagram.com/username (sin paths adicionales)
  const pathMatch = location.pathname.match(/^\/([^/]+)\/?$/);
  if (!pathMatch) return false;

  const username = pathMatch[1];
  const excludedPaths = [
    'explore', 'reels', 'direct', 'accounts', 'stories',
    'p', 'tv', 'reel', 'about', 'privacy', 'terms', 'help'
  ];

  return !excludedPaths.includes(username);
}
