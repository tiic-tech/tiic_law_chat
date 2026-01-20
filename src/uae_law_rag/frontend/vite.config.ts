//docstring
// 职责: 前端构建与开发服务器配置入口（含 alias 与 proxy）。
// 边界: 不包含业务代码或运行时逻辑。
// 上游关系: package.json scripts 调用。
// 下游关系: Vite dev/build 流程读取本配置。
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  // 关键：让 Vite 在读取 config 时也加载 .env.*（否则 process.env 拿不到）
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  const BACKEND_TARGET = env.VITE_BACKEND_TARGET ?? 'http://127.0.0.1:18000'

  return {
    plugins: [react()],

    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },

    server: {
      host: '127.0.0.1',
      proxy: {
        '/api': {
          target: BACKEND_TARGET,
          changeOrigin: true,
          secure: false,
        },
      },
    },

    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  }
})
