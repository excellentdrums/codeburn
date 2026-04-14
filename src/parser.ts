import { readdir, readFile } from 'fs/promises'
import { basename, join } from 'path'
import { homedir } from 'os'
import { calculateCost, getShortModelName } from './models.js'
import { BASH_TOOLS, hasMcpTools, classifyTurn } from './classifier.js'
import { extractBashCommands } from './bash-utils.js'
import { discoverAllSessions, getProvider } from './providers/index.js'
import type { ParsedProviderCall } from './providers/types.js'
import type {
  AssistantMessageContent,
  ClassifiedTurn,
  ContentBlock,
  DateRange,
  JournalEntry,
  ParsedApiCall,
  ParsedTurn,
  ProjectSummary,
  SessionSummary,
  TokenUsage,
  ToolUseBlock,
} from './types.js'

function getClaudeDir(): string {
  return process.env['CLAUDE_CONFIG_DIR'] || join(homedir(), '.claude')
}

function getProjectsDir(): string {
  return join(getClaudeDir(), 'projects')
}

function getGeminiDir(): string {
  return process.env['GEMINI_CONFIG_DIR'] || join(homedir(), '.gemini')
}

function getGeminiTmpDir(): string {
  return join(getGeminiDir(), 'tmp')
}

function getDesktopSessionsDir(): string {
  if (process.platform === 'darwin') return join(homedir(), 'Library', 'Application Support', 'Claude', 'local-agent-mode-sessions')
  if (process.platform === 'win32') return join(homedir(), 'AppData', 'Roaming', 'Claude', 'local-agent-mode-sessions')
  return join(homedir(), '.config', 'Claude', 'local-agent-mode-sessions')
}

async function findDesktopProjectDirs(base: string): Promise<string[]> {
  const results: string[] = []
  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > 8) return
    const entries = await readdir(dir).catch(() => [])
    for (const entry of entries) {
      if (entry === 'node_modules' || entry === '.git') continue
      const full = join(dir, entry)
      const s = await stat(full).catch(() => null)
      if (!s?.isDirectory()) continue
      if (entry === 'projects') {
        const projectDirs = await readdir(full).catch(() => [])
        for (const pd of projectDirs) {
          const pdFull = join(full, pd)
          const pdStat = await stat(pdFull).catch(() => null)
          if (pdStat?.isDirectory()) results.push(pdFull)
        }
      } else {
        await walk(full, depth + 1)
      }
    }
  }
  await walk(base, 0)
  return results
}

export async function findGeminiProjectDirs(base: string): Promise<Array<{ path: string; name: string }>> {
  const results: Array<{ path: string; name: string }> = []
  const entries = await readdir(base).catch(() => [])
  for (const entry of entries) {
    const full = join(base, entry)
    const s = await stat(full).catch(() => null)
    if (s?.isDirectory()) {
      const chatsDir = join(full, 'chats')
      const cs = await stat(chatsDir).catch(() => null)
      if (cs?.isDirectory()) {
        results.push({ path: chatsDir, name: entry })
      }
    }
  }
  return results
}

function unsanitizePath(dirName: string): string {
  return dirName.replace(/-/g, '/')
}

function parseJsonlLine(line: string): JournalEntry | null {
  try {
    return JSON.parse(line) as JournalEntry
  } catch {
    return null
  }
}

function extractToolNames(content: ContentBlock[]): string[] {
  return content
    .filter((b): b is ToolUseBlock => b.type === 'tool_use')
    .map(b => b.name)
}

function extractMcpTools(tools: string[]): string[] {
  return tools
    .filter(t => t.startsWith('mcp_'))
    .map(t => {
      const parts = t.split('_')
      if (parts.length >= 2) {
        return parts[1].charAt(0).toUpperCase() + parts[1].slice(1)
      }
      return t
    })
}

function extractCoreTools(tools: string[]): string[] {
  return tools.filter(t => !t.startsWith('mcp_'))
}

function extractBashCommandsFromContent(content: ContentBlock[]): string[] {
  return content
    .filter((b): b is ToolUseBlock => b.type === 'tool_use' && BASH_TOOLS.has(b.name))
    .flatMap(b => {
      const input = (b.input as any)
      const command = input?.command ?? (typeof input === 'string' ? input : '')
      return typeof command === 'string' ? extractBashCommands(command) : []
    })
}

function getUserMessageText(entry: JournalEntry): string {
  if (!entry.message || entry.message.role !== 'user') return ''
  const content = entry.message.content
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
      .map(b => b.text)
      .join(' ')
  }
  return ''
}

function getMessageId(entry: JournalEntry): string | null {
  if (entry.type !== 'assistant') return null
  const msg = entry.message as AssistantMessageContent | undefined
  return msg?.id ?? null
}

function parseApiCall(entry: JournalEntry): ParsedApiCall | null {
  if (entry.type !== 'assistant') return null
  const msg = entry.message as AssistantMessageContent | undefined
  if (!msg?.usage || !msg?.model) return null

  const usage = msg.usage
  const tokens: TokenUsage = {
    inputTokens: usage.input_tokens ?? 0,
    outputTokens: usage.output_tokens ?? 0,
    cacheCreationInputTokens: usage.cache_creation_input_tokens ?? 0,
    cacheReadInputTokens: usage.cache_read_input_tokens ?? 0,
    cachedInputTokens: 0,
    reasoningTokens: 0,
    webSearchRequests: usage.server_tool_use?.web_search_requests ?? 0,
  }

  const tools = extractToolNames(msg.content ?? [])
  const costUSD = calculateCost(
    msg.model,
    tokens.inputTokens,
    tokens.outputTokens,
    tokens.cacheCreationInputTokens,
    tokens.cacheReadInputTokens,
    tokens.webSearchRequests,
    usage.speed ?? 'standard',
  )

  const bashCmds = extractBashCommandsFromContent(msg.content ?? [])

  return {
    provider: 'claude',
    model: msg.model,
    usage: tokens,
    costUSD,
    tools,
    mcpTools: extractMcpTools(tools),
    hasAgentSpawn: tools.includes('Agent'),
    hasPlanMode: tools.includes('EnterPlanMode'),
    speed: usage.speed ?? 'standard',
    timestamp: entry.timestamp ?? '',
    bashCommands: bashCmds,
    deduplicationKey: msg.id ?? `claude:${entry.timestamp}`,
  }
}

function groupIntoTurns(entries: JournalEntry[], seenMsgIds: Set<string>): ParsedTurn[] {
  const turns: ParsedTurn[] = []
  let currentUserMessage = ''
  let currentCalls: ParsedApiCall[] = []
  let currentTimestamp = ''
  let currentSessionId = ''

  for (const entry of entries) {
    if (entry.type === 'user') {
      const text = getUserMessageText(entry)
      if (text.trim()) {
        if (currentCalls.length > 0) {
          turns.push({
            userMessage: currentUserMessage,
            assistantCalls: currentCalls,
            timestamp: currentTimestamp,
            sessionId: currentSessionId,
          })
        }
        currentUserMessage = text
        currentCalls = []
        currentTimestamp = entry.timestamp ?? ''
        currentSessionId = entry.sessionId ?? ''
      }
    } else if (entry.type === 'assistant') {
      const msgId = getMessageId(entry)
      if (msgId && seenMsgIds.has(msgId)) continue
      if (msgId) seenMsgIds.add(msgId)
      const call = parseApiCall(entry)
      if (call) currentCalls.push(call)
    }
  }

  if (currentCalls.length > 0) {
    turns.push({
      userMessage: currentUserMessage,
      assistantCalls: currentCalls,
      timestamp: currentTimestamp,
      sessionId: currentSessionId,
    })
  }

  return turns
}

function buildSessionSummary(
  sessionId: string,
  project: string,
  turns: ClassifiedTurn[],
): SessionSummary {
  const modelBreakdown: SessionSummary['modelBreakdown'] = {}
  const toolBreakdown: SessionSummary['toolBreakdown'] = {}
  const mcpBreakdown: SessionSummary['mcpBreakdown'] = {}
  const bashBreakdown: SessionSummary['bashBreakdown'] = {}
  const categoryBreakdown: SessionSummary['categoryBreakdown'] = {} as SessionSummary['categoryBreakdown']

  let totalCost = 0
  let totalInput = 0
  let totalOutput = 0
  let totalCacheRead = 0
  let totalCacheWrite = 0
  let apiCalls = 0
  let firstTs = ''
  let lastTs = ''

  for (const turn of turns) {
    const turnCost = turn.assistantCalls.reduce((s, c) => s + c.costUSD, 0)

    if (!categoryBreakdown[turn.category]) {
      categoryBreakdown[turn.category] = { turns: 0, costUSD: 0, retries: 0, editTurns: 0, oneShotTurns: 0 }
    }
    categoryBreakdown[turn.category].turns++
    categoryBreakdown[turn.category].costUSD += turnCost
    if (turn.hasEdits) {
      categoryBreakdown[turn.category].editTurns++
      categoryBreakdown[turn.category].retries += turn.retries
      if (turn.retries === 0) categoryBreakdown[turn.category].oneShotTurns++
    }

    for (const call of turn.assistantCalls) {
      totalCost += call.costUSD
      totalInput += call.usage.inputTokens
      totalOutput += call.usage.outputTokens
      totalCacheRead += call.usage.cacheReadInputTokens
      totalCacheWrite += call.usage.cacheCreationInputTokens
      apiCalls++

      const modelKey = getShortModelName(call.model)
      if (!modelBreakdown[modelKey]) {
        modelBreakdown[modelKey] = {
          calls: 0,
          costUSD: 0,
          tokens: { inputTokens: 0, outputTokens: 0, cacheCreationInputTokens: 0, cacheReadInputTokens: 0, cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0 },
        }
      }
      modelBreakdown[modelKey].calls++
      modelBreakdown[modelKey].costUSD += call.costUSD
      modelBreakdown[modelKey].tokens.inputTokens += call.usage.inputTokens
      modelBreakdown[modelKey].tokens.outputTokens += call.usage.outputTokens
      modelBreakdown[modelKey].tokens.cacheReadInputTokens += call.usage.cacheReadInputTokens
      modelBreakdown[modelKey].tokens.cacheCreationInputTokens += call.usage.cacheCreationInputTokens

      for (const tool of extractCoreTools(call.tools)) {
        toolBreakdown[tool] = toolBreakdown[tool] ?? { calls: 0 }
        toolBreakdown[tool].calls++
      }
      for (const mcp of call.mcpTools) {
        mcpBreakdown[mcp] = mcpBreakdown[mcp] ?? { calls: 0 }
        mcpBreakdown[mcp].calls++
      }
      for (const cmd of call.bashCommands) {
        bashBreakdown[cmd] = bashBreakdown[cmd] ?? { calls: 0 }
        bashBreakdown[cmd].calls++
      }

      if (!firstTs || call.timestamp < firstTs) firstTs = call.timestamp
      if (!lastTs || call.timestamp > lastTs) lastTs = call.timestamp
    }
  }

  return {
    sessionId,
    project,
    firstTimestamp: firstTs || turns[0]?.timestamp || '',
    lastTimestamp: lastTs || turns[turns.length - 1]?.timestamp || '',
    totalCostUSD: totalCost,
    totalInputTokens: totalInput,
    totalOutputTokens: totalOutput,
    totalCacheReadTokens: totalCacheRead,
    totalCacheWriteTokens: totalCacheWrite,
    apiCalls,
    turns,
    modelBreakdown,
    toolBreakdown,
    mcpBreakdown,
    bashBreakdown,
    categoryBreakdown,
  }
}

async function parseSessionFile(
  filePath: string,
  project: string,
  seenMsgIds: Set<string>,
  dateRange?: DateRange,
): Promise<SessionSummary | null> {
  let content: string
  try {
    content = await readFile(filePath, 'utf-8')
  } catch {
    return null
  }
  const lines = content.split('\n').filter(l => l.trim())
  const entries: JournalEntry[] = []

  for (const line of lines) {
    const entry = parseJsonlLine(line)
    if (entry) entries.push(entry)
  }

  if (entries.length === 0) return null

  let filteredEntries = entries
  if (dateRange) {
    filteredEntries = entries.filter(e => {
      if (!e.timestamp) return e.type === 'user'
      const ts = new Date(e.timestamp)
      return ts >= dateRange.start && ts <= dateRange.end
    })
    if (filteredEntries.length === 0) return null
  }

  const sessionId = basename(filePath, '.jsonl')
  const turns = groupIntoTurns(filteredEntries, seenMsgIds)
  const classified = turns.map(classifyTurn)

  return buildSessionSummary(sessionId, project, classified)
}

<<<<<<< HEAD
async function collectJsonlFiles(dirPath: string): Promise<string[]> {
  const files = await readdir(dirPath).catch(() => [])
  const jsonlFiles = files.filter(f => f.endsWith('.jsonl')).map(f => join(dirPath, f))

  for (const entry of files) {
    if (entry.endsWith('.jsonl')) continue
    const subagentsPath = join(dirPath, entry, 'subagents')
    const subFiles = await readdir(subagentsPath).catch(() => [])
    for (const sf of subFiles) {
      if (sf.endsWith('.jsonl')) jsonlFiles.push(join(subagentsPath, sf))
    }
  }

  return jsonlFiles
=======
export async function parseGeminiSessionFile(
  filePath: string,
  project: string,
  seenMsgIds: Set<string>,
  dateRange?: DateRange,
): Promise<SessionSummary | null> {
  let raw: string
  try {
    raw = await readFile(filePath, 'utf-8')
  } catch {
    return null
  }

  let data: any
  try {
    data = JSON.parse(raw)
  } catch {
    return null
  }

  if (!data.messages || !Array.isArray(data.messages)) return null

  const entries: JournalEntry[] = data.messages.map((m: any): JournalEntry => {
    const timestamp = m.timestamp || ''
    const sessionId = data.sessionId || ''
    
    if (m.type === 'user') {
      const text = typeof m.content === 'string' ? m.content : (m.content?.[0]?.text ?? '')
      return { type: 'user', timestamp, sessionId, message: { role: 'user', content: text } }
    }
    
    if (m.type === 'gemini') {
      const content: ContentBlock[] = [
        { type: 'text', text: m.content ?? '' },
        ...(m.toolCalls?.map((tc: any) => ({
          type: 'tool_use',
          id: tc.id,
          name: tc.name,
          input: tc.args,
        })) ?? [])
      ]
      
      return {
        type: 'assistant',
        timestamp,
        sessionId,
        message: {
          id: m.id,
          model: m.model || 'gemini',
          type: 'message',
          role: 'assistant',
          content,
          usage: {
            input_tokens: m.tokens?.input ?? 0,
            output_tokens: (m.tokens?.output ?? 0) + (m.tokens?.thoughts ?? 0),
            cache_read_input_tokens: m.tokens?.cached ?? 0,
            cache_creation_input_tokens: 0,
          }
        }
      }
    }
    
    return { type: m.type, timestamp, sessionId }
  })

  let filteredEntries = entries
  if (dateRange) {
    filteredEntries = entries.filter(e => {
      if (!e.timestamp) return e.type === 'user'
      const ts = new Date(e.timestamp)
      return ts >= dateRange.start && ts <= dateRange.end
    })
    if (filteredEntries.length === 0) return null
  }

  const turns = groupIntoTurns(filteredEntries, seenMsgIds)
  const classified = turns.map(classifyTurn)

  return buildSessionSummary(data.sessionId || basename(filePath, '.json'), project, classified)
>>>>>>> fe31aed (feat: implement Gemini session support)
}

async function scanProjectDirs(dirs: Array<{ path: string; name: string }>, seenMsgIds: Set<string>, dateRange?: DateRange): Promise<ProjectSummary[]> {
  const projectMap = new Map<string, SessionSummary[]>()

  for (const { path: dirPath, name: dirName } of dirs) {
    const jsonlFiles = await collectJsonlFiles(dirPath)

    for (const filePath of jsonlFiles) {
      const session = await parseSessionFile(filePath, dirName, seenMsgIds, dateRange)
      if (session && session.apiCalls > 0) {
        const existing = projectMap.get(dirName) ?? []
        existing.push(session)
        projectMap.set(dirName, existing)
      }
    }
  }

  return Array.from(projectMap.entries()).map(([dirName, sessions]) => ({
    project: dirName,
    projectPath: unsanitizePath(dirName),
    sessions,
    totalCostUSD: sessions.reduce((s, sess) => s + sess.totalCostUSD, 0),
    totalApiCalls: sessions.reduce((s, sess) => s + sess.apiCalls, 0),
  }))
}

export async function scanGeminiProjectDirs(dirs: Array<{ path: string; name: string }>, seenMsgIds: Set<string>, dateRange?: DateRange): Promise<ProjectSummary[]> {
  const projectMap = new Map<string, SessionSummary[]>()

  for (const { path: dirPath, name: dirName } of dirs) {
    const files = await readdir(dirPath).catch(() => [])
    const jsonFiles = files.filter(f => f.endsWith('.json') && f.startsWith('session-'))

    for (const file of jsonFiles) {
      const session = await parseGeminiSessionFile(join(dirPath, file), dirName, seenMsgIds, dateRange)
      if (session && session.apiCalls > 0) {
        const existing = projectMap.get(dirName) ?? []
        existing.push(session)
        projectMap.set(dirName, existing)
      }
    }
  }

  return Array.from(projectMap.entries()).map(([dirName, sessions]) => ({
    project: dirName,
    projectPath: dirName,
    sessions,
    totalCostUSD: sessions.reduce((s, sess) => s + sess.totalCostUSD, 0),
    totalApiCalls: sessions.reduce((s, sess) => s + sess.apiCalls, 0),
  }))
}

function providerCallToTurn(call: ParsedProviderCall): ParsedTurn {
  const tools = call.tools
  const usage: TokenUsage = {
    inputTokens: call.inputTokens,
    outputTokens: call.outputTokens,
    cacheCreationInputTokens: call.cacheCreationInputTokens,
    cacheReadInputTokens: call.cacheReadInputTokens,
    cachedInputTokens: call.cachedInputTokens,
    reasoningTokens: call.reasoningTokens,
    webSearchRequests: call.webSearchRequests,
  }

  const apiCall: ParsedApiCall = {
    provider: call.provider,
    model: call.model,
    usage,
    costUSD: call.costUSD,
    tools,
    mcpTools: extractMcpTools(tools),
    hasAgentSpawn: tools.includes('Agent'),
    hasPlanMode: tools.includes('EnterPlanMode'),
    speed: call.speed,
    timestamp: call.timestamp,
    bashCommands: [],
    deduplicationKey: call.deduplicationKey,
  }

  return {
    userMessage: call.userMessage,
    assistantCalls: [apiCall],
    timestamp: call.timestamp,
    sessionId: call.sessionId,
  }
}

async function parseProviderSources(
  providerName: string,
  sources: Array<{ path: string; project: string }>,
  seenKeys: Set<string>,
  dateRange?: DateRange,
): Promise<ProjectSummary[]> {
  const provider = getProvider(providerName)
  if (!provider) return []

  const sessionMap = new Map<string, { project: string; turns: ClassifiedTurn[] }>()

  for (const source of sources) {
    const parser = provider.createSessionParser(
      { path: source.path, project: source.project, provider: providerName },
      seenKeys,
    )

    for await (const call of parser.parse()) {
      if (dateRange) {
        if (!call.timestamp) continue
        const ts = new Date(call.timestamp)
        if (ts < dateRange.start || ts > dateRange.end) continue
      }

      const turn = providerCallToTurn(call)
      const classified = classifyTurn(turn)
      const key = `${providerName}:${call.sessionId}:${source.project}`

      const existing = sessionMap.get(key)
      if (existing) {
        existing.turns.push(classified)
      } else {
        sessionMap.set(key, { project: source.project, turns: [classified] })
      }
    }
  }

  const projectMap = new Map<string, SessionSummary[]>()
  for (const [key, { project, turns }] of sessionMap) {
    const sessionId = key.split(':')[1] ?? key
    const session = buildSessionSummary(sessionId, project, turns)
    if (session.apiCalls > 0) {
      const existing = projectMap.get(project) ?? []
      existing.push(session)
      projectMap.set(project, existing)
    }
  }

  const projects: ProjectSummary[] = []
  for (const [dirName, sessions] of projectMap) {
    projects.push({
      project: dirName,
      projectPath: unsanitizePath(dirName),
      sessions,
      totalCostUSD: sessions.reduce((s, sess) => s + sess.totalCostUSD, 0),
      totalApiCalls: sessions.reduce((s, sess) => s + sess.apiCalls, 0),
    })
  }

  return projects
}

const CACHE_TTL_MS = 60_000
const MAX_CACHE_ENTRIES = 10
const sessionCache = new Map<string, { data: ProjectSummary[]; ts: number }>()

function cacheKey(dateRange?: DateRange, providerFilter?: string): string {
  const s = dateRange ? `${dateRange.start.getTime()}:${dateRange.end.getTime()}` : 'none'
  return `${s}:${providerFilter ?? 'all'}`
}

function cachePut(key: string, data: ProjectSummary[]) {
  const now = Date.now()
  for (const [k, v] of sessionCache) {
    if (now - v.ts > CACHE_TTL_MS) sessionCache.delete(k)
  }
  if (sessionCache.size >= MAX_CACHE_ENTRIES) {
    const oldest = [...sessionCache.entries()].sort((a, b) => a[1].ts - b[1].ts)[0]
    if (oldest) sessionCache.delete(oldest[0])
  }
  sessionCache.set(key, { data, ts: now })
}

export async function parseAllSessions(dateRange?: DateRange, providerFilter?: string): Promise<ProjectSummary[]> {
  const key = cacheKey(dateRange, providerFilter)
  const cached = sessionCache.get(key)
  if (cached && Date.now() - cached.ts < CACHE_TTL_MS) return cached.data

  const seenMsgIds = new Set<string>()
  const seenKeys = new Set<string>()
  const allSources = await discoverAllSessions(providerFilter)

  const claudeSources = allSources.filter(s => s.provider === 'claude')
  const nonClaudeSources = allSources.filter(s => s.provider !== 'claude')

  const claudeDirs = claudeSources.map(s => ({ path: s.path, name: s.project }))
  const claudeProjects = await scanProjectDirs(claudeDirs, seenMsgIds, dateRange)

  const providerGroups = new Map<string, Array<{ path: string; project: string }>>()
  for (const source of nonClaudeSources) {
    const existing = providerGroups.get(source.provider) ?? []
    existing.push({ path: source.path, project: source.project })
    providerGroups.set(source.provider, existing)
  }

  const otherProjects: ProjectSummary[] = []
  for (const [providerName, sources] of providerGroups) {
    const projects = await parseProviderSources(providerName, sources, seenKeys, dateRange)
    otherProjects.push(...projects)
  }

  const geminiTmpDir = getGeminiTmpDir()
  const geminiDirs = await findGeminiProjectDirs(geminiTmpDir)
  const geminiProjects = (!providerFilter || providerFilter === 'gemini') 
    ? await scanGeminiProjectDirs(geminiDirs, seenMsgIds, dateRange) 
    : []

  const mergedMap = new Map<string, ProjectSummary>()
  
  const addProject = (p: ProjectSummary) => {
    const existing = mergedMap.get(p.project)
    if (existing) {
      existing.sessions.push(...p.sessions)
      existing.totalCostUSD += p.totalCostUSD
      existing.totalApiCalls += p.totalApiCalls
    } else {
      mergedMap.set(p.project, { ...p })
    }
  }

  claudeProjects.forEach(addProject)
  otherProjects.forEach(addProject)
  geminiProjects.forEach(addProject)

  return Array.from(mergedMap.values()).sort((a, b) => b.totalCostUSD - a.totalCostUSD)
}
