import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('api', {
  openVideo: () => ipcRenderer.invoke('open-video'),
  saveVideo: (defaultName: string) => ipcRenderer.invoke('save-video', defaultName),
  showInFolder: (filePath: string) => ipcRenderer.invoke('show-in-folder', filePath),
  submitTask: (payload: object) => ipcRenderer.invoke('engine:submit-task', payload),
  getTask: (taskId: string) => ipcRenderer.invoke('engine:get-task', taskId),
  startEngine: () => ipcRenderer.invoke('engine:start'),
});
