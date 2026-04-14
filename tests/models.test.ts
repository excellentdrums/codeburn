import { describe, it, expect, beforeAll } from 'vitest'
import { getModelCosts, getShortModelName, loadPricing } from '../src/models.js'

describe('models', () => {
  beforeAll(async () => {
    await loadPricing()
  })

  it('returns correct costs for Gemini 3.1 Pro', () => {
    const costs = getModelCosts('gemini-3.1-pro')
    expect(costs).toBeDefined()
    expect(costs?.inputCostPerToken).toBe(2e-6)
    expect(costs?.outputCostPerToken).toBe(12e-6)
  })

  it('returns correct costs for Gemini 2.5 Flash', () => {
    const costs = getModelCosts('gemini-2.5-flash')
    expect(costs).toBeDefined()
    expect(costs?.inputCostPerToken).toBe(0.3e-6)
    expect(costs?.outputCostPerToken).toBe(2.5e-6)
  })

  it('returns correct short name for Gemini models', () => {
    expect(getShortModelName('gemini-3.1-pro-001')).toBe('Gemini 3.1 Pro')
    expect(getShortModelName('gemini-2.5-flash-lite-latest')).toBe('Gemini 2.5 Flash-Lite')
  })

  it('handles canonicalization of Gemini models', () => {
    expect(getShortModelName('gemini-3.1-pro@20260414')).toBe('Gemini 3.1 Pro')
  })
})
