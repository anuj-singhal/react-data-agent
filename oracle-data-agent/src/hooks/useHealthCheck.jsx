// hooks/useHealthCheck.js
import { useState, useEffect } from 'react';
import apiService from '../services/api';
import { HEALTH_CHECK_INTERVAL } from '../utils/constants';

export const useHealthCheck = () => {
  const [healthStatus, setHealthStatus] = useState(null);

  const checkHealth = async () => {
    const status = await apiService.checkHealth();
    setHealthStatus(status);
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, HEALTH_CHECK_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  return { healthStatus, checkHealth };
};