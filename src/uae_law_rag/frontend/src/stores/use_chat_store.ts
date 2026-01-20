import { useSyncExternalStore } from 'react'
import { chatStore } from '@/stores/chat_store'

export const useChatStore = () => {
  return useSyncExternalStore(chatStore.subscribe, chatStore.getState, chatStore.getState)
}
