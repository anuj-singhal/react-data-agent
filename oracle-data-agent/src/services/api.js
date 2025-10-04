// services/api.js
import { API_URL } from '../utils/constants';

class ApiService {
  async checkHealth() {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Health check failed:', error);
      return { 
        status: 'error', 
        mcp_connected: false, 
        available_tools: [] 
      };
    }
  }

  async executeQuery(query) {
    if (!query || !query.trim()) {
      throw new Error('Query cannot be empty');
    }

    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: query.trim() }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  }

  async getTables() {
    const response = await fetch(`${API_URL}/tables`);
    if (!response.ok) {
      throw new Error('Failed to fetch tables');
    }
    return response.json();
  }

  async getTableInfo(tableName) {
    const response = await fetch(`${API_URL}/table/${tableName}`);
    if (!response.ok) {
      throw new Error('Failed to fetch table info');
    }
    return response.json();
  }
}

export default new ApiService();