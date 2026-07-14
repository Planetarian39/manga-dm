import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import test from 'node:test'

const repositoryRoot = resolve(import.meta.dirname, '..')
const checker = join(repositoryRoot, 'scripts', 'check-built-docs.mjs')

function write(root, relativePath, content = '') {
  const target = join(root, relativePath)
  mkdirSync(dirname(target), { recursive: true })
  writeFileSync(target, content, 'utf8')
}

function createBuiltFixture() {
  const root = mkdtempSync(join(tmpdir(), 'manga-built-docs-'))
  write(
    root,
    'index.html',
    '<main id="home"><a href="/manga-dm/methods/model.html#nfw">Model</a><link href="/manga-dm/assets/style.css"></main>',
  )
  write(root, 'methods/model.html', '<h1 id="nfw">NFW</h1>')
  write(root, 'assets/style.css', 'body{}')
  return root
}

function runCheck(root) {
  return spawnSync(process.execPath, [checker, '--root', root, '--base', '/manga-dm/'], {
    encoding: 'utf8',
  })
}

test('accepts base-prefixed built routes, fragments, and assets', () => {
  const root = createBuiltFixture()
  try {
    const result = runCheck(root)
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects extensionless direct routes when only an HTML file exists', () => {
  const root = createBuiltFixture()
  try {
    write(root, 'index.html', '<a href="/manga-dm/methods/model">Broken direct route</a>')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /missing built target/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects missing fragments in existing HTML pages', () => {
  const root = createBuiltFixture()
  try {
    write(root, 'index.html', '<a href="/manga-dm/methods/model.html#missing">Broken fragment</a>')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /missing built fragment/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
