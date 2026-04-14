import { describe, it, expect, vi, beforeAll } from 'vitest'
import { parseGeminiSessionFile } from '../src/parser.js'
import { loadPricing } from '../src/models.js'
import { readFile } from 'fs/promises'

vi.mock('fs/promises', async () => {
  const actual = await vi.importActual('fs/promises')
  return {
    ...actual,
    readFile: vi.fn(),
  }
})

describe('parseGeminiSessionFile', () => {
  beforeAll(async () => {
    await loadPricing()
  })

  it('correctly parses a Gemini session JSON', async () => {
    const mockSession = {
      sessionId: 'test-session-123',
      messages: [
        {
          id: 'msg-1',
          timestamp: '2026-04-14T10:00:00Z',
          type: 'user',
          content: 'Hello gemini'
        },
        {
          id: 'msg-2',
          timestamp: '2026-04-14T10:00:05Z',
          type: 'gemini',
          model: 'gemini-3.1-pro-preview',
          content: 'Hello user, I will help you.',
          tokens: {
            input: 1000,
            output: 500,
            cached: 200,
            thoughts: 100
          },
          toolCalls: [
            {
              id: 'call-1',
              name: 'read_file',
              args: { file_path: 'README.md' }
            }
          ]
        }
      ]
    }

    vi.mocked(readFile).mockResolvedValue(JSON.stringify(mockSession))

    const summary = await parseGeminiSessionFile('fake-path.json', 'test-project', new Set())

    expect(summary).toBeDefined()
    expect(summary?.sessionId).toBe('test-session-123')
    expect(summary?.totalInputTokens).toBe(1000)
    expect(summary?.totalOutputTokens).toBe(600)
    expect(summary?.totalCacheReadTokens).toBe(200)
    expect(summary?.modelBreakdown['Gemini 3.1 Pro']).toBeDefined()
    expect(summary?.turns.length).toBe(1)
    expect(summary?.turns[0].assistantCalls[0].tools).toContain('read_file')
  })

  it('correctly detects and groups Linear MCP calls', async () => {
    const mockSession = {
      sessionId: 'test-session-linear',
      messages: [
        {
          id: 'msg-linear-1',
          timestamp: '2026-04-14T12:00:00Z',
          type: 'gemini',
          model: 'gemini-3-flash',
          toolCalls: [
            { id: 'linear-1', name: 'mcp_linear_list_issues' },
            { id: 'linear-2', name: 'mcp_linear_create_issue' }
          ]
        }
      ]
    }
    vi.mocked(readFile).mockResolvedValue(JSON.stringify(mockSession))

    const summary = await parseGeminiSessionFile('linear-path.json', 'test-project', new Set())
    
    expect(summary?.mcpBreakdown['Linear']).toBeDefined()
    expect(summary?.mcpBreakdown['Linear'].calls).toBe(2)
  })
})
