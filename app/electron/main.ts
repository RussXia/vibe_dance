import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import path from 'node:path';
import { spawn } from 'node:child_process';

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 960,
    height: 720,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, '../../dist/index.html'));
  }
}

// 启动 Python 引擎子进程
let engineProc: ReturnType<typeof spawn> | null = null;

// venv 中 Python 可执行文件路径：macOS/Linux 为 bin/python，Windows 为 Scripts/python.exe
function pythonExecutable(venvDir: string): string {
  return process.platform === 'win32'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');
}

// PyInstaller 冻结后的引擎可执行文件名（engine/dist/engine_bundle/ 下）：
// macOS/Linux 为 vibe_engine，Windows 为 vibe_engine.exe
function engineBinaryName(): string {
  return process.platform === 'win32' ? 'vibe_engine.exe' : 'vibe_engine';
}

function startEngine() {
  let engineCmd: string;
  let engineArgs: string[];
  let engineCwd: string;
  if (app.isPackaged) {
    // 打包后：使用 PyInstaller 冻结的引擎二进制
    // （electron-builder extraResources 拷入 Resources/engine_bundle/vibe_engine）
    const bundleDir = path.join(process.resourcesPath, 'engine_bundle');
    engineCmd = path.join(bundleDir, engineBinaryName());
    engineArgs = ['8787'];
    engineCwd = bundleDir;
  } else {
    // 开发时：引擎路径指向仓库根的 engine/.venv。运行时代码位于
    // dist-electron/electron/main.js，__dirname 为 dist-electron/electron/，
    // 上溯 3 级（dist-electron/electron -> dist-electron -> app -> 仓库根）到仓库根 engine/。
    const engineDir = path.join(__dirname, '../../../engine');
    engineCmd = pythonExecutable(path.join(engineDir, '.venv'));
    engineArgs = ['-m', 'engine', '8787'];
    engineCwd = engineDir;
  }
  engineProc = spawn(engineCmd, engineArgs, { cwd: engineCwd });
  engineProc.on('exit', (code: number | null) => {
    console.log(`engine exited with code ${code}`);
    engineProc = null;
  });
}

const ENGINE_BASE = 'http://127.0.0.1:8787';

async function engineFetch(pathname: string, init?: RequestInit) {
  const res = await fetch(`${ENGINE_BASE}${pathname}`, init);
  if (!res.ok) {
    throw new Error(`engine error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

ipcMain.handle('engine:submit-task', async (_e, payload) => {
  return engineFetch('/task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
});

ipcMain.handle('engine:get-task', async (_e, taskId: string) => {
  return engineFetch(`/task/${taskId}`);
});

ipcMain.handle('engine:submit-audio-task', async (_e, payload) => {
  return engineFetch('/audio-task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
});

ipcMain.handle('engine:get-audio-task', async (_e, taskId: string) => {
  return engineFetch(`/audio-task/${taskId}`);
});

ipcMain.handle('engine:render-audio-task', async (_e, taskId: string, payload: { offset_seconds: number; tempo_ratio?: number }) => {
  return engineFetch(`/audio-task/${taskId}/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
});

ipcMain.handle('open-audio', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    filters: [{ name: '音频/视频', extensions: ['mp3', 'wav', 'flac', 'm4a', 'aac', 'mp4', 'mov'] }],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return { path: result.filePaths[0] };
});

ipcMain.handle('open-any-media', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    filters: [
      { name: '视频', extensions: ['mp4', 'mov'] },
      { name: '音频', extensions: ['mp3', 'wav', 'flac', 'm4a', 'aac'] },
    ],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return { path: result.filePaths[0] };
});

ipcMain.handle('open-video', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    filters: [{ name: '视频', extensions: ['mp4', 'mov'] }],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return { path: result.filePaths[0] };
});

ipcMain.handle('save-video', async (_e, defaultName: string) => {
  const result = await dialog.showSaveDialog(mainWindow!, {
    title: '保存导出视频',
    defaultPath: defaultName,
    filters: [{ name: 'MP4 视频', extensions: ['mp4'] }],
  });
  if (result.canceled || !result.filePath) {
    return null;
  }
  return { path: result.filePath };
});

ipcMain.handle('show-in-folder', async (_e, filePath: string) => {
  shell.showItemInFolder(filePath);
  return { ok: true };
});

app.whenReady().then(() => {
  createWindow();
  ipcMain.handle('engine:start', async () => {
    startEngine();
    return { ok: true };
  });
  // 自动拉起引擎子进程（引擎 HTTP 服务监听 8787，供 submitTask 调用）；
  // engine:start IPC 仍保留作为引擎崩溃/重启的入口。
  startEngine();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// app 真正退出前杀掉引擎子进程，避免其成为孤儿进程
//（孤儿引擎会一直占用 8787 端口，且让已卸载 app 的进程/图标残留）。
// will-quit 在 Cmd+Q（macOS）与 window-all-closed->quit（Windows/Linux）
// 两条退出路径都会触发，统一在此清理。
app.on('will-quit', () => {
  if (engineProc) {
    engineProc.kill();
    engineProc = null;
  }
});
