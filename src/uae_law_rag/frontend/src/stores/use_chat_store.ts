import { useSyncExternalStore } from 'react'
import type { ChatStore } from '@/stores/chat_store'

export const useChatStore = (store: ChatStore) => {
  return useSyncExternalStore(store.subscribe, store.getState, store.getState)
}
