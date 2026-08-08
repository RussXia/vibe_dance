/// <reference types="vite/client" />

export interface AudioAlignResult {
  offset_seconds: number;
  tempo_ratio: number;
  confidence: string;
  method: string;
}

export interface AudioPreview {
  video_a_path: string;
  audio_a_path: string;
  audio_b_path: string;
  waveform_a: number[];
  waveform_b: number[];
}

export interface AudioTaskInfo {
  task_id: string;
  status: string;
  progress: number;
  message: string;
  align_result?: AudioAlignResult;
  preview?: AudioPreview;
}

declare global {
  interface Window {
    api: {
      openVideo: () => Promise<{ path: string } | null>;
      openAudio: () => Promise<{ path: string } | null>;
      openAnyMedia: () => Promise<{ path: string } | null>;
      saveVideo: (defaultName: string) => Promise<{ path: string } | null>;
      showInFolder: (filePath: string) => Promise<{ ok: boolean }>;
      submitTask: (payload: object) => Promise<{ task_id: string; status: string }>;
      getTask: (taskId: string) => Promise<{ task_id: string; status: string; progress: number; message: string }>;
      submitAudioTask: (payload: object) => Promise<{ task_id: string; status: string }>;
      getAudioTask: (taskId: string) => Promise<AudioTaskInfo>;
      renderAudioTask: (taskId: string, payload: object) => Promise<{ ok: boolean }>;
      startEngine: () => Promise<{ ok: boolean }>;
    };
  }
}

export {};
