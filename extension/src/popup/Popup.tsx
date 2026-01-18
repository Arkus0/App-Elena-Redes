import { useState, useEffect } from 'react';
import type { Platform, PageType, AnalysisStatus, UserConfig } from '../types';

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

// API config for syncing
const API_BASE_URL = 'http://localhost:8000/api/v1';

export default function Popup() {
  const [tabStatus, setTabStatus] = useState<TabStatus | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [loading, setLoading] = useState(true);

  // Human-in-the-Loop config state
  const [showConfig, setShowConfig] = useState(false);
  const [userConfig, setUserConfig] = useState<UserConfig>({});
  const [instagramUsername, setInstagramUsername] = useState('');
  const [tiktokUsername, setTiktokUsername] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [configStatus, setConfigStatus] = useState<'idle' | 'saving' | 'syncing' | 'success' | 'error'>('idle');
  const [configMessage, setConfigMessage] = useState('');

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

    // Cargar configuración guardada
    chrome.runtime.sendMessage({ action: 'GET_USER_CONFIG' }, (response) => {
      if (response?.config) {
        setUserConfig(response.config);
        setInstagramUsername(response.config.ownInstagramUsername || '');
        setTiktokUsername(response.config.ownTiktokUsername || '');
      }
    });

    // Cargar API key si existe
    chrome.storage.sync.get(['extensionApiKey'], (result) => {
      if (result.extensionApiKey) {
        setApiKey(result.extensionApiKey);
      }
    });
  }, []);

  // Guardar configuración manualmente
  const handleSaveConfig = () => {
    setConfigStatus('saving');
    const newConfig: UserConfig = {
      ownInstagramUsername: instagramUsername.replace('@', '').toLowerCase() || undefined,
      ownTiktokUsername: tiktokUsername.replace('@', '').toLowerCase() || undefined,
      lastSyncAt: new Date().toISOString()
    };

    chrome.runtime.sendMessage(
      { action: 'SAVE_USER_CONFIG', config: newConfig },
      (response) => {
        if (response?.success) {
          setUserConfig(newConfig);
          setConfigStatus('success');
          setConfigMessage('Configuración guardada');
          setTimeout(() => setConfigStatus('idle'), 2000);
        } else {
          setConfigStatus('error');
          setConfigMessage('Error al guardar');
        }
      }
    );
  };

  // Sincronizar con API usando API key
  const handleSyncWithApi = async () => {
    if (!apiKey) {
      setConfigStatus('error');
      setConfigMessage('Introduce tu API key primero');
      return;
    }

    setConfigStatus('syncing');
    setConfigMessage('Sincronizando...');

    try {
      const response = await fetch(`${API_BASE_URL}/extension/config`, {
        method: 'GET',
        headers: {
          'X-Extension-API-Key': apiKey,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('API key inválida');
      }

      const data = await response.json();

      // Guardar config localmente
      const newConfig: UserConfig = {
        ownInstagramUsername: data.own_instagram_username || undefined,
        ownTiktokUsername: data.own_tiktok_username || undefined,
        lastSyncAt: new Date().toISOString()
      };

      chrome.runtime.sendMessage(
        { action: 'SAVE_USER_CONFIG', config: newConfig },
        (res) => {
          if (res?.success) {
            setUserConfig(newConfig);
            setInstagramUsername(data.own_instagram_username || '');
            setTiktokUsername(data.own_tiktok_username || '');
            setConfigStatus('success');
            setConfigMessage(`Sincronizado: ${data.business_name}`);

            // Guardar API key para futuras sincronizaciones
            chrome.storage.sync.set({ extensionApiKey: apiKey });
          }
        }
      );
    } catch (error) {
      setConfigStatus('error');
      setConfigMessage(error instanceof Error ? error.message : 'Error de sincronización');
    }

    setTimeout(() => setConfigStatus('idle'), 3000);
  };

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

      {/* Human-in-the-Loop: Config Section */}
      <section className="config-section">
        <button
          className="config-toggle"
          onClick={() => setShowConfig(!showConfig)}
        >
          {'\u{2699}'} Mi Perfil IG/TikTok
          <span className={`toggle-arrow ${showConfig ? 'open' : ''}`}>
            {showConfig ? '\u{25B2}' : '\u{25BC}'}
          </span>
        </button>

        {showConfig && (
          <div className="config-panel">
            {/* Feedback Loop Status */}
            <div className={`feedback-status ${userConfig.ownInstagramUsername || userConfig.ownTiktokUsername ? 'active' : 'inactive'}`}>
              {userConfig.ownInstagramUsername || userConfig.ownTiktokUsername ? (
                <>
                  <span className="status-icon">{'\u{2705}'}</span>
                  <span>Feedback Loop Activo</span>
                </>
              ) : (
                <>
                  <span className="status-icon">{'\u{26A0}'}</span>
                  <span>Configura tu username</span>
                </>
              )}
            </div>

            {/* Current Config */}
            {(userConfig.ownInstagramUsername || userConfig.ownTiktokUsername) && (
              <div className="current-config">
                {userConfig.ownInstagramUsername && (
                  <div className="config-item">
                    <span>{'\u{1F4F7}'}</span>
                    <span>@{userConfig.ownInstagramUsername}</span>
                  </div>
                )}
                {userConfig.ownTiktokUsername && (
                  <div className="config-item">
                    <span>{'\u{1F3B5}'}</span>
                    <span>@{userConfig.ownTiktokUsername}</span>
                  </div>
                )}
              </div>
            )}

            {/* Manual Config */}
            <div className="config-inputs">
              <div className="input-group">
                <label>Instagram Username</label>
                <input
                  type="text"
                  placeholder="@miusuario"
                  value={instagramUsername}
                  onChange={(e) => setInstagramUsername(e.target.value)}
                />
              </div>
              <div className="input-group">
                <label>TikTok Username</label>
                <input
                  type="text"
                  placeholder="@miusuario"
                  value={tiktokUsername}
                  onChange={(e) => setTiktokUsername(e.target.value)}
                />
              </div>
              <button
                className="save-btn"
                onClick={handleSaveConfig}
                disabled={configStatus === 'saving'}
              >
                {configStatus === 'saving' ? 'Guardando...' : 'Guardar Manual'}
              </button>
            </div>

            {/* API Sync */}
            <div className="api-sync">
              <div className="divider">
                <span>o sincronizar con dashboard</span>
              </div>
              <div className="input-group">
                <label>API Key (del dashboard)</label>
                <input
                  type="password"
                  placeholder="elena_ext_xxxx..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
              </div>
              <button
                className="sync-btn"
                onClick={handleSyncWithApi}
                disabled={configStatus === 'syncing' || !apiKey}
              >
                {configStatus === 'syncing' ? 'Sincronizando...' : '\u{1F504} Sincronizar'}
              </button>
            </div>

            {/* Status Message */}
            {configStatus !== 'idle' && configMessage && (
              <div className={`config-message ${configStatus}`}>
                {configMessage}
              </div>
            )}

            <p className="config-hint">
              Cuando guardes tu propio contenido, las metricas reales se enviaran al modelo ML para mejorar las predicciones.
            </p>
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="popup-footer">
        <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
          Ver API Docs
        </a>
        <a href="http://localhost:5173/my-profile" target="_blank" rel="noopener noreferrer">
          Dashboard Config
        </a>
      </footer>
    </div>
  );
}
