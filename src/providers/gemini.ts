import { join, basename } from 'path'
import { readFile } from 'fs/promises'
import { homedir } from 'os'
import { getShortModelName, calculateCost } from '../models.js'
import { findGeminiProjectDirs } from '../parser.js'
import { BASH_TOOLS } from '../classifier.js'
import { extractBashCommands } from '../bash-utils.js'
import type { Provider, SessionParser, SessionSource, ParsedProviderCall } from './types.js'

function getGeminiDir(): string {
  return process.env['GEMINI_CONFIG_DIR'] || join(homedir(), '.gemini')
}

function getGeminiTmpDir(): string {
  return join(getGeminiDir(), 'tmp')
}

export const gemini: Provider = {
  name: 'gemini',
  displayName: 'Gemini',
  modelDisplayName: (model: string) => getShortModelName(model),
  toolDisplayName: (rawTool: string) => rawTool,
  async discoverSessions(): Promise<SessionSource[]> {
    const dirs = await findGeminiProjectDirs(getGeminiTmpDir())
    const sources: SessionSource[] = []
    for (const dir of dirs) {
      const files = await import('fs/promises').then(fs => fs.readdir(dir.path)).catch(() => [])
      for (const file of files.filter(f => f.endsWith('.json') && f.startsWith('session-'))) {
        sources.push({ path: join(dir.path, file), project: dir.name, provider: 'gemini' })
      }
    }
    return sources
  },
  createSessionParser(source: SessionSource, seenKeys: Set<string>): SessionParser {
    return {
      async *parse() {
        let raw: string
        try {
          raw = await readFile(source.path, 'utf-8')
        } catch { return }

        let data: any
        try {
          data = JSON.parse(raw)
        } catch { return }

        if (!data.messages || !Array.isArray(data.messages)) return

        for (const m of data.messages) {
          if (m.type === 'gemini') {
            const id = m.id ?? `gemini:${m.timestamp}`
            if (seenKeys.has(id)) continue
            seenKeys.add(id)

            const tools = m.toolCalls?.map((tc: any) => tc.name) ?? []
            const bashCommands: string[] = []
            for (const tc of m.toolCalls ?? []) {
              if (BASH_TOOLS.has(tc.name)) {
                const cmd = tc.args?.command ?? (typeof tc.args === 'string' ? tc.args : '')
                if (typeof cmd === 'string') bashCommands.push(...extractBashCommands(cmd))
              }
            }

            const costUSD = calculateCost(
              m.model || 'gemini',
              m.tokens?.input ?? 0,
              (m.tokens?.output ?? 0) + (m.tokens?.thoughts ?? 0),
              0,
              m.tokens?.cached ?? 0,
              0,
              'standard'
            )

            yield {
              provider: 'gemini',
              model: m.model || 'gemini',
              inputTokens: m.tokens?.input ?? 0,
              outputTokens: (m.tokens?.output ?? 0) + (m.tokens?.thoughts ?? 0),
              cacheCreationInputTokens: 0,
              cacheReadInputTokens: m.tokens?.cached ?? 0,
              cachedInputTokens: 0,
              reasoningTokens: m.tokens?.thoughts ?? 0,
              webSearchRequests: 0,
              costUSD,
              tools,
              timestamp: m.timestamp ?? '',
              speed: 'standard',
              deduplicationKey: id,
              userMessage: '',
              sessionId: data.sessionId ?? 'unknown',
              bashCommands
            }
          }
        }
      }
    }
  }
}
