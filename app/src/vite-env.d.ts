/// <reference types="vite/client" />

interface Window {
  api: {
    openVideo: () => Promise<{ path: string } | null>;
    saveVideo: (defaultName: string) => Promise<{ path: string } | null>;
    showInFolder: (filePath: string) => Promise<{ ok: boolean }>;
    submitTask: (payload: object) => Promise<{ task_id: string; status: string }>;
    getTask: (taskId: string) => Promise<{ task_id: string; status: string; progress: number; message: string }>;
    startEngine: () => Promise<{ ok: boolean }>;
  };
}
