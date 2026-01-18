/**
 * Instagram Content Extractor
 * Extrae datos de posts, reels y stories individuales
 * Prioriza datos hidratados, fallback a DOM scraping
 */

import type {
  ContentData,
  MediaItem,
  ContentMetrics,
  AudioInfo,
  ExtractionResult,
  PageType
} from '../types';

// Interfaces para datos hidratados de Instagram
interface IGMediaNode {
  id: string;
  __typename: string;
  display_url?: string;
  video_url?: string;
  is_video?: boolean;
  edge_media_to_caption?: { edges: Array<{ node: { text: string } }> };
  edge_media_preview_like?: { count: number };
  edge_liked_by?: { count: number };
  edge_media_to_comment?: { count: number };
  edge_media_preview_comment?: { count: number };
  video_view_count?: number;
  taken_at_timestamp?: number;
  dimensions?: { width: number; height: number };
  video_duration?: number;
  accessibility_caption?: string;
  owner?: {
    username: string;
    full_name?: string;
    profile_pic_url?: string;
    is_verified?: boolean;
  };
  edge_sidecar_to_children?: {
    edges: Array<{ node: IGMediaNode }>;
  };
  clips_music_attribution_info?: {
    song_name?: string;
    artist_name?: string;
    audio_id?: string;
  };
}

interface IGHydratedData {
  graphql?: {
    shortcode_media?: IGMediaNode;
  };
  items?: Array<{
    pk?: string;
    id?: string;
    code?: string;
    media_type?: number;
    caption?: { text: string };
    like_count?: number;
    comment_count?: number;
    view_count?: number;
    play_count?: number;
    taken_at?: number;
    image_versions2?: { candidates: Array<{ url: string; width: number; height: number }> };
    video_versions?: Array<{ url: string; width: number; height: number }>;
    user?: {
      username: string;
      full_name?: string;
      profile_pic_url?: string;
      is_verified?: boolean;
    };
    carousel_media?: Array<{
      id: string;
      media_type: number;
      image_versions2?: { candidates: Array<{ url: string }> };
      video_versions?: Array<{ url: string }>;
    }>;
    music_metadata?: {
      music_info?: {
        music_asset_info?: {
          title?: string;
          display_artist?: string;
        };
      };
    };
  }>;
}

// Parsea números con formato (ej: "1.2M", "500K")
function parseFormattedNumber(text: string | null | undefined): number | null {
  if (!text) return null;
  const cleaned = text.trim().toLowerCase().replace(/,/g, '').replace(/\s/g, '');

  if (cleaned.includes('m')) {
    return Math.round(parseFloat(cleaned.replace('m', '')) * 1_000_000);
  }
  if (cleaned.includes('k')) {
    return Math.round(parseFloat(cleaned.replace('k', '')) * 1_000);
  }

  const num = parseInt(cleaned, 10);
  return isNaN(num) ? null : num;
}

// Extrae hashtags de un caption
function extractHashtags(caption: string): string[] {
  const matches = caption.match(/#[\w\u00C0-\u024F]+/g);
  return matches ? matches.map(h => h.toLowerCase()) : [];
}

// Extrae menciones de un caption
function extractMentions(caption: string): string[] {
  const matches = caption.match(/@[\w.]+/g);
  return matches ? matches.map(m => m.toLowerCase()) : [];
}

// Busca datos hidratados en la página
function getHydratedData(): IGHydratedData | null {
  try {
    // Método 1: Buscar en scripts con type="application/json"
    const scripts = document.querySelectorAll('script[type="application/json"]');
    for (const script of scripts) {
      try {
        const data = JSON.parse(script.textContent || '');
        if (data?.graphql?.shortcode_media || data?.items) {
          console.log('[Elena Bridge] IG content hydrated data found');
          return data;
        }
        // Buscar en estructuras anidadas
        if (data?.require) {
          for (const req of data.require) {
            if (Array.isArray(req) && req[3]?.[0]?.__bbox?.require) {
              for (const innerReq of req[3][0].__bbox.require) {
                if (innerReq[3]?.[0]?.data?.xdt_shortcode_media) {
                  return { graphql: { shortcode_media: innerReq[3][0].data.xdt_shortcode_media } };
                }
              }
            }
          }
        }
      } catch {
        // Continuar con el siguiente script
      }
    }

    // Método 2: window._sharedData (legacy)
    const sharedData = (window as Window & { _sharedData?: { entry_data?: { PostPage?: Array<{ graphql?: IGHydratedData['graphql'] }> } } })._sharedData;
    if (sharedData?.entry_data?.PostPage?.[0]?.graphql) {
      console.log('[Elena Bridge] IG content found via _sharedData');
      return { graphql: sharedData.entry_data.PostPage[0].graphql };
    }

    return null;
  } catch (error) {
    console.warn('[Elena Bridge] Error getting hydrated data:', error);
    return null;
  }
}

// Extrae datos del DOM cuando no hay datos hidratados
function extractFromDOM(): ContentData | null {
  try {
    const url = window.location.href;

    // Detectar tipo de contenido y ID
    let contentType: ContentData['contentType'] = 'post';
    let contentId = '';

    const postMatch = url.match(/\/p\/([^/]+)/);
    const reelMatch = url.match(/\/reel\/([^/]+)/);

    if (reelMatch) {
      contentType = 'reel';
      contentId = reelMatch[1];
    } else if (postMatch) {
      contentId = postMatch[1];
      // Detectar si es video/reel por elementos en la página
      if (document.querySelector('video') || document.querySelector('[aria-label*="Reel"]')) {
        contentType = 'reel';
      }
    }

    if (!contentId) return null;

    // Autor
    const authorLink = document.querySelector('header a[href^="/"]') as HTMLAnchorElement;
    const authorImg = document.querySelector('header img') as HTMLImageElement;
    const authorUsername = authorLink?.href?.match(/instagram\.com\/([^/]+)/)?.[1] || '';
    const isVerified = !!document.querySelector('header [aria-label*="erified"]');

    // Caption
    const captionEl = document.querySelector('h1') ||
                      document.querySelector('[data-testid="post-comment-root"]') ||
                      document.querySelector('article span[dir="auto"]');
    const caption = captionEl?.textContent?.trim() || null;

    // Media
    const media: MediaItem[] = [];
    const videoEl = document.querySelector('article video') as HTMLVideoElement;
    const imgEl = document.querySelector('article img[srcset], article img[src*="instagram"]') as HTMLImageElement;

    if (videoEl) {
      media.push({
        type: 'video',
        url: videoEl.src || null,
        thumbnailUrl: videoEl.poster || null
      });
    } else if (imgEl) {
      media.push({
        type: 'image',
        url: imgEl.src,
        thumbnailUrl: imgEl.src,
        altText: imgEl.alt || undefined
      });
    }

    // Métricas - buscar en la página
    const likesEl = document.querySelector('[href$="/liked_by/"] span') ||
                    document.querySelector('section span[class*="like"]');
    const viewsEl = document.querySelector('span[class*="view"]');

    const metrics: ContentMetrics = {
      likes: parseFormattedNumber(likesEl?.textContent),
      comments: null,
      shares: null,
      saves: null,
      views: parseFormattedNumber(viewsEl?.textContent),
      plays: null
    };

    // Timestamp
    const timeEl = document.querySelector('time');
    const postedAt = timeEl?.getAttribute('datetime') || null;

    return {
      platform: 'instagram',
      contentType,
      contentId,
      contentUrl: url,
      author: {
        username: authorUsername,
        displayName: null,
        profilePicUrl: authorImg?.src || null,
        isVerified
      },
      media,
      caption,
      hashtags: caption ? extractHashtags(caption) : [],
      mentions: caption ? extractMentions(caption) : [],
      metrics,
      audio: null,
      postedAt,
      extractedAt: new Date().toISOString(),
      sourceUrl: url,
      extractionMethod: 'dom_scraping'
    };
  } catch (error) {
    console.error('[Elena Bridge] Error extracting from DOM:', error);
    return null;
  }
}

// Extrae desde datos hidratados
function extractFromHydrated(data: IGHydratedData): ContentData | null {
  try {
    // Formato GraphQL (posts/reels web)
    if (data.graphql?.shortcode_media) {
      const media = data.graphql.shortcode_media;
      const isVideo = media.is_video || media.__typename === 'GraphVideo';
      const isCarousel = media.__typename === 'GraphSidecar';

      let contentType: ContentData['contentType'] = 'post';
      if (media.__typename === 'XDTGraphVideo' || media.clips_music_attribution_info) {
        contentType = 'reel';
      } else if (isVideo) {
        contentType = 'reel';
      } else if (isCarousel) {
        contentType = 'carousel';
      }

      const caption = media.edge_media_to_caption?.edges?.[0]?.node?.text || null;

      // Extraer items de media
      const mediaItems: MediaItem[] = [];

      if (isCarousel && media.edge_sidecar_to_children?.edges) {
        for (const edge of media.edge_sidecar_to_children.edges) {
          const node = edge.node;
          mediaItems.push({
            type: node.is_video ? 'video' : 'image',
            url: node.is_video ? node.video_url || null : node.display_url || null,
            thumbnailUrl: node.display_url || null,
            width: node.dimensions?.width,
            height: node.dimensions?.height,
            altText: node.accessibility_caption
          });
        }
      } else {
        mediaItems.push({
          type: isVideo ? 'video' : 'image',
          url: isVideo ? media.video_url || null : media.display_url || null,
          thumbnailUrl: media.display_url || null,
          width: media.dimensions?.width,
          height: media.dimensions?.height,
          duration: media.video_duration,
          altText: media.accessibility_caption
        });
      }

      // Audio info para reels
      let audio: AudioInfo | null = null;
      if (media.clips_music_attribution_info) {
        audio = {
          title: media.clips_music_attribution_info.song_name || null,
          artist: media.clips_music_attribution_info.artist_name || null,
          isOriginal: false,
          audioUrl: null
        };
      }

      return {
        platform: 'instagram',
        contentType,
        contentId: media.id,
        contentUrl: window.location.href,
        author: {
          username: media.owner?.username || '',
          displayName: media.owner?.full_name || null,
          profilePicUrl: media.owner?.profile_pic_url || null,
          isVerified: media.owner?.is_verified || false
        },
        media: mediaItems,
        caption,
        hashtags: caption ? extractHashtags(caption) : [],
        mentions: caption ? extractMentions(caption) : [],
        metrics: {
          likes: media.edge_media_preview_like?.count || media.edge_liked_by?.count || null,
          comments: media.edge_media_to_comment?.count || media.edge_media_preview_comment?.count || null,
          shares: null,
          saves: null,
          views: media.video_view_count || null,
          plays: null
        },
        audio,
        postedAt: media.taken_at_timestamp
          ? new Date(media.taken_at_timestamp * 1000).toISOString()
          : null,
        extractedAt: new Date().toISOString(),
        sourceUrl: window.location.href,
        extractionMethod: 'hydrated_data'
      };
    }

    // Formato items (API móvil)
    if (data.items?.[0]) {
      const item = data.items[0];
      const isVideo = item.media_type === 2;
      const isCarousel = item.media_type === 8;

      let contentType: ContentData['contentType'] = 'post';
      if (item.music_metadata || item.play_count !== undefined) {
        contentType = 'reel';
      } else if (isVideo) {
        contentType = 'reel';
      } else if (isCarousel) {
        contentType = 'carousel';
      }

      const caption = item.caption?.text || null;
      const mediaItems: MediaItem[] = [];

      if (isCarousel && item.carousel_media) {
        for (const cm of item.carousel_media) {
          const isVid = cm.media_type === 2;
          mediaItems.push({
            type: isVid ? 'video' : 'image',
            url: isVid
              ? cm.video_versions?.[0]?.url || null
              : cm.image_versions2?.candidates?.[0]?.url || null,
            thumbnailUrl: cm.image_versions2?.candidates?.[0]?.url || null
          });
        }
      } else {
        mediaItems.push({
          type: isVideo ? 'video' : 'image',
          url: isVideo
            ? item.video_versions?.[0]?.url || null
            : item.image_versions2?.candidates?.[0]?.url || null,
          thumbnailUrl: item.image_versions2?.candidates?.[0]?.url || null
        });
      }

      let audio: AudioInfo | null = null;
      if (item.music_metadata?.music_info?.music_asset_info) {
        const musicInfo = item.music_metadata.music_info.music_asset_info;
        audio = {
          title: musicInfo.title || null,
          artist: musicInfo.display_artist || null,
          isOriginal: false,
          audioUrl: null
        };
      }

      return {
        platform: 'instagram',
        contentType,
        contentId: item.code || item.pk?.toString() || item.id || '',
        contentUrl: window.location.href,
        author: {
          username: item.user?.username || '',
          displayName: item.user?.full_name || null,
          profilePicUrl: item.user?.profile_pic_url || null,
          isVerified: item.user?.is_verified || false
        },
        media: mediaItems,
        caption,
        hashtags: caption ? extractHashtags(caption) : [],
        mentions: caption ? extractMentions(caption) : [],
        metrics: {
          likes: item.like_count || null,
          comments: item.comment_count || null,
          shares: null,
          saves: null,
          views: item.view_count || null,
          plays: item.play_count || null
        },
        audio,
        postedAt: item.taken_at
          ? new Date(item.taken_at * 1000).toISOString()
          : null,
        extractedAt: new Date().toISOString(),
        sourceUrl: window.location.href,
        extractionMethod: 'hydrated_data'
      };
    }

    return null;
  } catch (error) {
    console.error('[Elena Bridge] Error extracting from hydrated:', error);
    return null;
  }
}

/**
 * Función principal para extraer contenido de Instagram
 */
export function extractInstagramContent(): ExtractionResult {
  console.log('[Elena Bridge] Starting Instagram content extraction...');

  // Intentar datos hidratados primero
  const hydratedData = getHydratedData();

  if (hydratedData) {
    const content = extractFromHydrated(hydratedData);
    if (content) {
      console.log('[Elena Bridge] Extracted via hydrated data:', content.contentId);
      return {
        success: true,
        pageType: content.contentType === 'reel' ? 'reel' : 'post',
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
      pageType: content.contentType === 'reel' ? 'reel' : 'post',
      content
    };
  }

  return {
    success: false,
    pageType: 'unknown',
    error: 'No se pudo extraer el contenido. Asegúrate de estar viendo un post o reel.'
  };
}

/**
 * Detecta el tipo de página de Instagram
 */
export function detectInstagramPageType(): PageType {
  const url = window.location.href;

  if (url.includes('/reel/')) return 'reel';
  if (url.includes('/p/')) return 'post';
  if (url.includes('/stories/')) return 'story';
  if (url.match(/instagram\.com\/[^/]+\/?$/)) return 'profile';

  return 'unknown';
}

/**
 * Verifica si estamos en una página de contenido de Instagram
 */
export function isInstagramContentPage(): boolean {
  const pageType = detectInstagramPageType();
  return pageType === 'post' || pageType === 'reel';
}
