import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('api', {
  openVideo: () => ipcRenderer.invoke('open-video'),
  openAudio: () => ipcRenderer.invoke('open-audio'),
  openAnyMedia: () => ipcRenderer.invoke('open-any-media'),
  saveVideo: (defaultName: string) => ipcRenderer.invoke('save-video', defaultName),
  showInFolder: (filePath: string) => ipcRenderer.invoke('show-in-folder', filePath),
  submitTask: (payload: object) => ipcRenderer.invoke('engine:submit-task', payload),
  getTask: (taskId: string) => ipcRenderer.invoke('engine:get-task', taskId),
  submitAudioTask: (payload: object) => ipcRenderer.invoke('engine:submit-audio-task', payload),
  getAudioTask: (taskId: string) => ipcRenderer.invoke('engine:get-audio-task', taskId),
  renderAudioTask: (taskId: string, payload: object) => ipcRenderer.invoke('engine:render-audio-task', taskId, payload),
  startEngine: () => ipcRenderer.invoke('engine:start'),
});
