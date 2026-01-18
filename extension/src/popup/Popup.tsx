import { useState, useEffect } from 'react';
import type { Platform, PageType, AnalysisStatus } from '../types';

interface TabStatus {
  isContentPage: boolean;
  pageType: PageType;
  platform: Platform;
  status: AnalysisStatus;
}

interface ServerStatus {
  serverOnline: boolean;
  apiUrl: string;
}

export default function Popup() {
  const [tabStatus, setTabStatus] = useState<TabStatus | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Obtener estado de la tab actual
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const activeTab = tabs[0];

      if (activeTab?.id) {
        chrome.tabs.sendMessage(activeTab.id, { action: 'GET_STATUS' }, (response) => {
          if (chrome.runtime.lastError) {
            // El content script no está cargado (no estamos en IG/TikTok)
            setTabStatus({
              isContentPage: false,
              pageType: 'unknown',
              platform: 'unknown',
              status: 'idle'
            });
          } else {
            setTabStatus(response);
          }
        });
      }
    });

    // Obtener estado del servidor
    chrome.runtime.sendMessage({ action: 'GET_STATUS' }, (response) => {
      setServerStatus(response);
      setLoading(false);
    });
  }, []);

  const getPlatformIcon = (platform: Platform): string => {
    switch (platform) {
      case 'instagram': return '\u{1F4F7}'; // Camara
      case 'tiktok': return '\u{1F3B5}'; // Nota musical
      default: return '\u{1F310}'; // Globo
    }
  };

  const getPlatformName = (platform: Platform): string => {
    switch (platform) {
      case 'instagram': return 'Instagram';
      case 'tiktok': return 'TikTok';
      default: return 'Desconocido';
    }
  };

  const getPageTypeLabel = (pageType: PageType): string => {
    switch (pageType) {
      case 'post': return 'Post detectado';
      case 'reel': return 'Reel detectado';
      case 'video': return 'Video detectado';
      case 'profile': return 'Perfil detectado';
      case 'story': return 'Story detectado';
      default: return 'Pagina no soportada';
    }
  };

  const getPageTypeIcon = (pageType: PageType): string => {
    switch (pageType) {
      case 'post': return '\u{1F4F8}'; // Camera with flash
      case 'reel': return '\u{1F3AC}'; // Clapper board
      case 'video': return '\u{1F4F9}'; // Video camera
      case 'profile': return '\u{1F464}'; // Person silhouette
      case 'story': return '\u{23F3}'; // Hourglass
      default: return '\u{2753}'; // Question mark
    }
  };

  if (loading) {
    return (
      <div className="popup-container">
        <div className="loading">Cargando...</div>
      </div>
    );
  }

  return (
    <div className="popup-container">
      {/* Header */}
      <header className="popup-header">
        <div className="logo">
          <span className="logo-icon">{'\u{1F9E0}'}</span>
          <h1>Elena Bridge</h1>
        </div>
        <span className="version">v1.0.0</span>
      </header>

      {/* Estado del servidor */}
      <section className="status-section">
        <h2>Estado del Servidor</h2>
        <div className={`status-badge ${serverStatus?.serverOnline ? 'online' : 'offline'}`}>
          <span className="status-dot"></span>
          <span>{serverStatus?.serverOnline ? 'Conectado' : 'Desconectado'}</span>
        </div>
        {!serverStatus?.serverOnline && (
          <p className="status-hint">
            Inicia el servidor backend en localhost:8000
          </p>
        )}
      </section>

      {/* Estado de la pestaña actual */}
      <section className="tab-section">
        <h2>Pagina Actual</h2>
        {tabStatus?.isContentPage ? (
          <div className="profile-detected">
            <span className="platform-icon">{getPageTypeIcon(tabStatus.pageType)}</span>
            <div className="profile-info">
              <span className="platform-name">
                {getPlatformIcon(tabStatus.platform)} {getPlatformName(tabStatus.platform)}
              </span>
              <span className="profile-hint">{getPageTypeLabel(tabStatus.pageType)}</span>
            </div>
          </div>
        ) : (
          <div className="no-profile">
            <p>No estas en contenido de Instagram o TikTok</p>
            <p className="hint">Navega a un post, reel o video para guardar</p>
          </div>
        )}
      </section>

      {/* Instrucciones */}
      <section className="instructions-section">
        <h2>Como usar</h2>
        <ol>
          <li>Navega por Instagram o TikTok</li>
          <li>Encuentra un post, reel o video que te guste</li>
          <li>Haz clic en el boton flotante para guardarlo</li>
          <li>Los datos se envian automaticamente al backend</li>
        </ol>
      </section>

      {/* Contenido soportado */}
      <section className="supported-section">
        <h2>Contenido Soportado</h2>
        <div className="supported-list">
          <span className="supported-item">{'\u{1F4F8}'} Posts</span>
          <span className="supported-item">{'\u{1F3AC}'} Reels</span>
          <span className="supported-item">{'\u{1F4F9}'} Videos</span>
          <span className="supported-item">{'\u{1F464}'} Perfiles</span>
        </div>
      </section>

      {/* Footer */}
      <footer className="popup-footer">
        <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
          Ver API Docs
        </a>
      </footer>
    </div>
  );
}
