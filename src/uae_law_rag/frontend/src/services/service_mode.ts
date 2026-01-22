import { env } from '@/config/env'

export type ServiceMode = 'mock' | 'live'

let override: ServiceMode | undefined

const normalizeMode = (value: string | undefined): ServiceMode => {
  if (value === 'live') return 'live'
  return 'mock'
}

export const getServiceMode = (): ServiceMode => {
  if (override) return override
  return normalizeMode(env.serviceMode)
}

export const setServiceMode = (mode: ServiceMode) => {
  override = mode
}
