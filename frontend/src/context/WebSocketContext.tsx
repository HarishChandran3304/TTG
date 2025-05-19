import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import { nanoid } from 'nanoid';
import config from '../config';

interface WebSocketContextType {
  connect: (owner: string, repo: string, subName?: string) => Promise<void>;
  disconnect: () => void;
  sendMessage: (message: string) => void;
  isConnected: boolean;
  isProcessing: boolean;
  lastMessage: string | null;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const connectingRef = useRef(false);

  const disconnect = () => {
    if (socket) {
      socket.close();
      setSocket(null);
      setIsConnected(false);
      setIsProcessing(false);
      setLastMessage(null); // Clear last message on disconnect
      connectingRef.current = false;
    }
  };

  const connect = async (owner: string, repo: string, subName?: string) => {
    // Prevent multiple simultaneous connection attempts
    if (connectingRef.current || isConnected) {
      return;
    }

    return new Promise<void>((resolve, reject) => {
      try {
        connectingRef.current = true;

        // Clean up any existing connection first
        if (socket) {
          socket.close();
          setSocket(null);
          setIsConnected(false);
          setIsProcessing(false);
        }

        const clientId = nanoid(10);
        const url = subName?
            `${config.API_URL}/${owner}/${repo}/${subName}/${clientId}`
            : `${config.API_URL}/${owner}/${repo}/${clientId}`

        const ws = new WebSocket(url);

        const timeout = setTimeout(() => {
          connectingRef.current = false;
          reject(new Error('Connection timeout'));
          ws.close();
        }, 5000); // 5 second timeout

        ws.onerror = (error) => {
          clearTimeout(timeout);
          connectingRef.current = false;
          setIsConnected(false);
          setIsProcessing(false);
          reject(error);
        };

        ws.onopen = () => {
          clearTimeout(timeout);
          setIsConnected(true);
          setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send('ping');
            }
          }, 240000); // Ping every 4 mins
          setIsProcessing(true);
          resolve();
        };

        ws.onmessage = (event) => {
          if (event.data === 'pong') {
            return;
          }
          if (event.data === 'repo_processed') {
            setIsProcessing(false);
            connectingRef.current = false;
            setLastMessage('repo_processed');
          } else if (event.data === 'error:repo_too_large') {
            setLastMessage('error:repo_too_large');
            connectingRef.current = false;
            setIsProcessing(false);
            ws.close();
          } else if (event.data === 'error:repo_not_found') {
            setLastMessage('error:repo_not_found');
            connectingRef.current = false;
            setIsProcessing(false);
            ws.close();
          } else if (event.data === 'error:repo_private') {
            setLastMessage('error:repo_private');
            connectingRef.current = false;
            setIsProcessing(false);
            ws.close();
          } else {
            setLastMessage(event.data);
          }
        };

        ws.onclose = () => {
          setIsConnected(false);
          setIsProcessing(false);
          setSocket(null);
          connectingRef.current = false;
        };

        setSocket(ws);
      } catch (error) {
        connectingRef.current = false;
        reject(error);
      }
    });
  };

  const sendMessage = (message: string) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(message);
    }
  };

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  return (
    <WebSocketContext.Provider
      value={{
        connect,
        disconnect,
        sendMessage,
        isConnected,
        isProcessing,
        lastMessage,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
}
