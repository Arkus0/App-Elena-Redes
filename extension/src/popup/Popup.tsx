import { useState, useEffect } from 'react';
import type { Platform, AnalysisStatus } from '../types';

interface TabStatus {
  isProfilePage: boolean;
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
              isProfilePage: false,
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
        <h2>Pestaña Actual</h2>
        {tabStatus?.isProfilePage ? (
          <div className="profile-detected">
            <span className="platform-icon">{getPlatformIcon(tabStatus.platform)}</span>
            <div className="profile-info">
              <span className="platform-name">{getPlatformName(tabStatus.platform)}</span>
              <span className="profile-hint">Perfil detectado</span>
            </div>
          </div>
        ) : (
          <div className="no-profile">
            <p>No estás en un perfil de Instagram o TikTok</p>
            <p className="hint">Navega a un perfil para usar Elena Bridge</p>
          </div>
        )}
      </section>

      {/* Instrucciones */}
      <section className="instructions-section">
        <h2>Como usar</h2>
        <ol>
          <li>Navega a un perfil de Instagram o TikTok</li>
          <li>Haz clic en el boton "Analizar con Elena"</li>
          <li>Los datos se enviaran automaticamente al backend</li>
        </ol>
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
