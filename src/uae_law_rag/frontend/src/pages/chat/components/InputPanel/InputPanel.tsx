// src/pages/chat/components/InputPanel/InputPanel.tsx
//docstring
// 职责: 输入区展示组件，采集用户 query 并通过 props 回调上抛。
// 边界: 不读写 stores；不调用 services/api；不依赖 HTTP DTO。
// 上游关系: ChatPage（container 传入 onSend）。
// 下游关系: 无（仅通过 props 回调上抛给上游容器）。
import { useState } from 'react'

type InputPanelProps = {
  onSend: (query: string) => Promise<void>
  disabled?: boolean
}

const InputPanel = ({ onSend, disabled }: InputPanelProps) => {
  const [value, setValue] = useState('')

  const handleClick = async () => {
    const query = value.trim()
    if (!query) return
    await onSend(query)
    setValue('')
  }

  return (
    <div className="input-panel">
      <input
        className="input-panel__input"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(event) => {
          if (event.key !== 'Enter') return
          event.preventDefault()
          if (disabled) return
          void handleClick()
        }}
        placeholder="Ask a question..."
        disabled={disabled}
      />
      <button
        className="input-panel__send"
        onClick={handleClick}
        disabled={disabled || !value.trim()}
        type="button"
      >
        Send
      </button>
    </div>
  )
}

export default InputPanel
