import React, { createContext, useContext, useState, useEffect } from 'react';
import { checkApiHealth } from '../services/api';

interface AppContextType {
  isOnline: boolean;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [selectedModel, setSelectedModel] = useState<string>('calibrated_review_network_image_xgboost');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

  // Periodic health check with FastAPI backend
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

  return (
    <AppContext.Provider
      value={{
        isOnline,
        selectedModel,
        setSelectedModel,
        sidebarCollapsed,
        setSidebarCollapsed,
        toggleSidebar,
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
