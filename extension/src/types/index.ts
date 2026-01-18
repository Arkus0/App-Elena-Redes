// Tipos para datos extraídos de perfiles sociales

export type Platform = 'instagram' | 'tiktok' | 'unknown';

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

export interface ExtractionResult {
  success: boolean;
  data: ProfileData | null;
  recentPosts?: ExtractedPost[];
  error?: string;
}

// Mensajes entre content script y service worker
export interface MessagePayload {
  action: 'EXTRACT_PROFILE' | 'SEND_TO_API' | 'GET_STATUS' | 'EXTRACTION_COMPLETE';
  data?: ExtractionResult | ProfileData;
  error?: string;
}

export interface ApiResponse {
  success: boolean;
  message: string;
  taskId?: string;
  error?: string;
}

// Estado del análisis
export type AnalysisStatus = 'idle' | 'extracting' | 'sending' | 'success' | 'error';
