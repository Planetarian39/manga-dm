#!/usr/bin/env node

import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, extname, join, relative, resolve, sep } from 'node:path'

const rootFlag = process.argv.indexOf('--root')
const baseFlag = process.argv.indexOf('--base')
const root = resolve(rootFlag >= 0 ? process.argv[rootFlag + 1] : 'docs/.vitepress/dist')
const base = baseFlag >= 0 ? process.argv[baseFlag + 1] : '/manga-dm/'
const errors = []

function walk(directory) {
  if (!existsSync(directory)) return []
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = join(directory, entry.name)
    return entry.isDirectory() ? walk(target) : [target]
  })
}

function relativeToRoot(file) {
  return relative(root, file).split(sep).join('/')
}

function resolveTarget(sourceFile, rawUrl) {
  let url = rawUrl.trim()
  if (!url || /^(?:https?:|mailto:|tel:|javascript:|data:|blob:|\/\/)/i.test(url)) return null

  let decoded
  try {
    decoded = decodeURIComponent(url)
  } catch {
    errors.push(`invalid encoded built URL in ${relativeToRoot(sourceFile)}: ${url}`)
    return null
  }

  const hashIndex = decoded.indexOf('#')
  const fragment = hashIndex >= 0 ? decoded.slice(hashIndex + 1) : ''
  const withoutHash = hashIndex >= 0 ? decoded.slice(0, hashIndex) : decoded
  const path = withoutHash.split('?')[0]

  let target
  if (!path) {
    target = sourceFile
  } else if (path.startsWith(base)) {
    target = resolve(root, path.slice(base.length))
  } else if (path.startsWith('/')) {
    errors.push(`built URL escapes configured base in ${relativeToRoot(sourceFile)}: ${url}`)
    return null
  } else {
    target = resolve(dirname(sourceFile), path)
  }

  if (path.endsWith('/')) target = resolve(target, 'index.html')
  return { target, fragment, rawUrl: url }
}

function hasFragment(file, fragment) {
  if (!fragment || extname(file).toLowerCase() !== '.html') return true
  const html = readFileSync(file, 'utf8')
  const escaped = fragment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return new RegExp(`(?:id|name)=["']${escaped}["']`).test(html)
}

if (!existsSync(root)) {
  console.error(`Built documentation root is missing: ${root}`)
  process.exit(1)
}

const htmlFiles = walk(root).filter((file) => extname(file).toLowerCase() === '.html')
for (const file of htmlFiles) {
  const html = readFileSync(file, 'utf8')
  for (const match of html.matchAll(/\b(?:href|src)=["']([^"']+)["']/gi)) {
    const resolved = resolveTarget(file, match[1])
    if (!resolved) continue
    if (!existsSync(resolved.target)) {
      errors.push(`missing built target in ${relativeToRoot(file)}: ${resolved.rawUrl}`)
      continue
    }
    if (!hasFragment(resolved.target, resolved.fragment)) {
      errors.push(`missing built fragment in ${relativeToRoot(file)}: ${resolved.rawUrl}`)
    }
  }
}

if (errors.length) {
  console.error(`Built documentation checks failed (${errors.length}):`)
  for (const error of errors) console.error(`- ${error}`)
  process.exit(1)
}

console.log(`Built documentation checks passed (${htmlFiles.length} HTML files)`)
