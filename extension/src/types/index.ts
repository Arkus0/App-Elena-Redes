// Tipos para datos extraídos de redes sociales

export type Platform = 'instagram' | 'tiktok' | 'unknown';

// Tipo de página/contenido que estamos viendo
export type PageType = 'profile' | 'post' | 'reel' | 'video' | 'story' | 'unknown';

// ============================================================================
// Datos de Perfil (para referencia del autor)
// ============================================================================

export interface ProfileData {
  platform: Platform;
  username: string;
  displayName: string | null;
  bio: string | null;
  profilePicUrl: string | null;
  isVerified: boolean;
  isPrivate: boolean;
  stats: ProfileStats;
  extractedAt: string;
  sourceUrl: string;
  extractionMethod: 'hydrated_data' | 'dom_scraping';
  rawHtml?: string;
}

export interface ProfileStats {
  followers: number | null;
  following: number | null;
  posts: number | null;
  likes?: number | null; // TikTok specific
}

// ============================================================================
// Datos de Contenido Individual (Posts, Reels, Videos)
// ============================================================================

export interface ContentData {
  platform: Platform;
  contentType: 'post' | 'reel' | 'video' | 'carousel' | 'story';
  contentId: string;
  contentUrl: string;

  // Autor del contenido
  author: {
    username: string;
    displayName: string | null;
    profilePicUrl: string | null;
    isVerified: boolean;
  };

  // Contenido multimedia
  media: MediaItem[];

  // Texto/Caption
  caption: string | null;
  hashtags: string[];
  mentions: string[];

  // Métricas de engagement
  metrics: ContentMetrics;

  // Audio (para reels/videos)
  audio: AudioInfo | null;

  // Metadata
  postedAt: string | null;
  extractedAt: string;
  sourceUrl: string;
  extractionMethod: 'hydrated_data' | 'dom_scraping';
}

export interface MediaItem {
  type: 'image' | 'video';
  url: string | null;
  thumbnailUrl: string | null;
  width?: number;
  height?: number;
  duration?: number; // Para videos, en segundos
  altText?: string;
}

export interface ContentMetrics {
  likes: number | null;
  comments: number | null;
  shares: number | null;
  saves: number | null;
  views: number | null;
  plays: number | null; // TikTok specific
}

export interface AudioInfo {
  title: string | null;
  artist: string | null;
  isOriginal: boolean;
  audioUrl: string | null;
}

// ============================================================================
// Resultado de Extracción Unificado
// ============================================================================

export interface ExtractionResult {
  success: boolean;
  pageType: PageType;
  // Uno de estos estará presente según el tipo de página
  profile?: ProfileData;
  content?: ContentData;
  // Para perfiles, lista de posts recientes
  recentPosts?: ExtractedPost[];
  error?: string;

  // Human-in-the-Loop: indica si el contenido es del perfil propio de la clienta
  // Cuando es true, el backend registra las métricas reales para feedback del modelo ML
  isOwnProfile?: boolean;
}

// ============================================================================
// Configuración de Usuario (chrome.storage)
// ============================================================================

// Configuración almacenada localmente en la extensión
// Permite identificar cuando el contenido extraído es del perfil propio
export interface UserConfig {
  // Username de Instagram de la clienta (sin @) - ej: "inmoalmeria"
  ownInstagramUsername?: string;
  // Username de TikTok de la clienta (sin @) - ej: "inmoalmeria"
  ownTiktokUsername?: string;
  // Timestamp de última sincronización
  lastSyncAt?: string;
}

// Para compatibilidad con lista de posts en perfiles
export interface ExtractedPost {
  id: string;
  type: 'image' | 'video' | 'carousel' | 'reel';
  thumbnailUrl: string | null;
  caption: string | null;
  likes: number | null;
  comments: number | null;
  shares?: number | null;
  views?: number | null;
  timestamp: string | null;
}

// ============================================================================
// Mensajes entre Content Script y Service Worker
// ============================================================================

export interface MessagePayload {
  action: 'EXTRACT_CONTENT' | 'SEND_TO_API' | 'GET_STATUS' | 'EXTRACTION_COMPLETE' |
          'SAVE_USER_CONFIG' | 'GET_USER_CONFIG';
  data?: ExtractionResult;
  config?: UserConfig;
  error?: string;
}

export interface ApiResponse {
  success: boolean;
  message: string;
  taskId?: string;
  error?: string;
}

// ============================================================================
// Estado del Análisis
// ============================================================================

export type AnalysisStatus = 'idle' | 'extracting' | 'sending' | 'success' | 'error';

// Estado completo para el popup
export interface ExtensionState {
  isContentPage: boolean;
  pageType: PageType;
  platform: Platform;
  status: AnalysisStatus;
  lastExtraction?: {
    contentType: string;
    author: string;
    timestamp: string;
  };
}
