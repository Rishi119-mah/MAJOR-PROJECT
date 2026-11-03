import React, { createContext, useContext, useState, useEffect } from 'react';
import { Language, AppMode } from '../types';
import { translations } from '../i18n/translations';
import { checkApiHealth } from '../services/api';

interface AppContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  mode: AppMode;
  setMode: (mode: AppMode) => void;
  isOnline: boolean;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  t: (key: string) => string;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguage] = useState<Language>(() => {
    return (localStorage.getItem('fraudshield_lang') as Language) || 'en';
  });

  const [mode, setMode] = useState<AppMode>('research');
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [selectedModel, setSelectedModel] = useState<string>('calibrated_review_network_image_xgboost');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

  // Sync language with localStorage
  useEffect(() => {
    localStorage.setItem('fraudshield_lang', language);
  }, [language]);

  // Periodic health check
  useEffect(() => {
    let isMounted = true;
    const verifyHealth = async () => {
      try {
        await checkApiHealth();
        if (isMounted) setIsOnline(true);
      } catch {
        if (isMounted) setIsOnline(false);
      }
    };

    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const toggleSidebar = () => setSidebarCollapsed((prev) => !prev);

  const t = (key: string): string => {
    return translations[language]?.[key] || translations.en[key] || key;
  };

  return (
    <AppContext.Provider
      value={{
        language,
        setLanguage,
        mode,
        setMode,
        isOnline,
        selectedModel,
        setSelectedModel,
        sidebarCollapsed,
        setSidebarCollapsed,
        toggleSidebar,
        t,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

