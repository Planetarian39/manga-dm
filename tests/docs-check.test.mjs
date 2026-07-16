import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import test from 'node:test'

const repositoryRoot = resolve(import.meta.dirname, '..')
const checker = join(repositoryRoot, 'scripts', 'check-docs.mjs')

function write(root, relativePath, content = '') {
  const target = join(root, relativePath)
  mkdirSync(dirname(target), { recursive: true })
  writeFileSync(target, content, 'utf8')
}

function createFixture() {
  const root = mkdtempSync(join(tmpdir(), 'manga-docs-check-'))
  write(
    root,
    'docs/public-boundary.json',
    JSON.stringify({
      allowedGalaxyIds: ['11743-9102'],
      requiredPages: ['index.md', 'methods/data-and-selection.md'],
      forbiddenExtensions: ['.tex', '.bib'],
      forbiddenPhrases: ['620 out of 1234'],
      forbiddenPatterns: ['\\b\\d+ galaxies\\b'],
      mcmcManifest: 'public/meta/mcmc-migration.json',
      minimumMcmcHeadings: 2,
      provenanceManifest: 'public/meta/case-study-provenance.json',
      additionalPublicAssets: ['assets/home/declared-hero.png'],
    }),
  )
  write(root, 'docs/index.md', '# Home\n\n[Method](/methods/data-and-selection)\n')
  write(root, 'docs/methods/data-and-selection.md', '# Data and selection\n')
  write(
    root,
    'docs/public/meta/mcmc-migration.json',
    JSON.stringify({ headings: [{ source: 'A', destination: '/a' }, { source: 'B', destination: '/b' }] }),
  )
  write(
    root,
    'docs/public/meta/case-study-provenance.json',
    JSON.stringify({
      artifacts: [
        {
          galaxyId: '11743-9102',
          publicPath: 'downloads/posteriors/11743-9102.nc',
          sha256: createHash('sha256').update('data').digest('hex'),
          bytes: 4,
        },
      ],
    }),
  )
  write(root, 'docs/public/downloads/posteriors/11743-9102.nc', 'data')
  write(root, 'docs/public/assets/home/declared-hero.png', 'hero')
  return root
}

function runCheck(root) {
  return spawnSync(process.execPath, [checker, '--root', root], { encoding: 'utf8' })
}

test('accepts a complete allowlisted documentation fixture', () => {
  const root = createFixture()
  try {
    const result = runCheck(root)
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
    assert.match(result.stdout, /Documentation checks passed/)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('accepts an item-based MCMC migration manifest', () => {
  const root = createFixture()
  try {
    write(
      root,
      'docs/public/meta/mcmc-migration.json',
      JSON.stringify({
        items: [
          { type: 'heading', source: { heading: 'A' }, destination: { path: 'docs/a.md' } },
          { type: 'heading', source: { heading: 'B' }, destination: { path: 'docs/b.md' } },
        ],
      }),
    )
    const result = runCheck(root)
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects broken internal links', () => {
  const root = createFixture()
  try {
    write(root, 'docs/index.md', '# Home\n\n[Missing](/methods/not-present)\n')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /broken internal link/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects forbidden paper source and non-allowlisted posterior assets', () => {
  const root = createFixture()
  try {
    write(root, 'docs/paper.tex', '\\section{Private}')
    write(root, 'docs/public/downloads/posteriors/9999-9999.nc', 'data')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /forbidden extension/i)
    assert.match(`${result.stdout}\n${result.stderr}`, /non-allowlisted posterior/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects case-study assets that are absent from provenance', () => {
  const root = createFixture()
  try {
    write(root, 'docs/public/assets/case-studies/11743-9102/untracked.png', 'image')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /case-study asset is missing provenance/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects every unexpected file under the public asset root', () => {
  const root = createFixture()
  try {
    write(root, 'docs/public/assets/aggregate-result.png', 'private aggregate figure')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /unexpected public file/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects forbidden result language in Vue-rendered content', () => {
  const root = createFixture()
  try {
    write(root, 'docs/.vitepress/theme/LeakingPanel.vue', '<template>620 out of 1234</template>')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /forbidden phrase/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects provenance whose galaxy ID does not match its public path', () => {
  const root = createFixture()
  try {
    write(
      root,
      'docs/public/meta/case-study-provenance.json',
      JSON.stringify({
        artifacts: [
          {
            galaxyId: '11743-9102',
            publicPath: 'downloads/posteriors/9999-9999.nc',
            sha256: createHash('sha256').update('data').digest('hex'),
            bytes: 4,
          },
        ],
      }),
    )
    write(root, 'docs/public/downloads/posteriors/9999-9999.nc', 'data')
    rmSync(join(root, 'docs/public/downloads/posteriors/11743-9102.nc'))
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /galaxy ID does not match public path/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects unpublished aggregate-count patterns in Markdown', () => {
  const root = createFixture()
  try {
    write(root, 'docs/index.md', '# Home\n\nThe retained set contains 42 galaxies.\n')
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /forbidden content pattern/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects raw inline LaTeX delimiters outside fenced code', () => {
  const root = createFixture()
  try {
    write(
      root,
      'docs/index.md',
      '# Home\n\nRaw inline math: \\(x + y\\).\n\n```text\nLiteral example: \\(x + y\\)\n```\n',
    )
    const rawResult = runCheck(root)
    assert.notEqual(rawResult.status, 0)
    assert.match(`${rawResult.stdout}\n${rawResult.stderr}`, /raw inline LaTeX delimiter/i)

    write(root, 'docs/index.md', '# Home\n\n```text\nLiteral example: \\(x + y\\)\n```\n')
    const fencedResult = runCheck(root)
    assert.equal(fencedResult.status, 0, `${fencedResult.stdout}\n${fencedResult.stderr}`)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('rejects incomplete MCMC migration coverage and invalid provenance hashes', () => {
  const root = createFixture()
  try {
    write(
      root,
      'docs/public/meta/mcmc-migration.json',
      JSON.stringify({ headings: [{ source: 'A', destination: '/a' }] }),
    )
    write(
      root,
      'docs/public/meta/case-study-provenance.json',
      JSON.stringify({ artifacts: [{ galaxyId: '11743-9102', publicPath: 'x.nc', sha256: 'bad', bytes: 1 }] }),
    )
    const result = runCheck(root)
    assert.notEqual(result.status, 0)
    assert.match(`${result.stdout}\n${result.stderr}`, /MCMC heading coverage/i)
    assert.match(`${result.stdout}\n${result.stderr}`, /invalid SHA-256/i)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('mobile home hero reserves space below the fixed navigation for its eyebrow', () => {
  const styles = readFileSync(join(repositoryRoot, 'docs', '.vitepress', 'theme', 'style.css'), 'utf8')
  const mobileRules = styles.match(/@media \(max-width: 640px\) \{([\s\S]*?)\n\}/)?.[1] ?? ''

  assert.match(mobileRules, /\.VPHomeHero\s*\{[\s\S]*?padding-top:\s*132px\s*!important;/)
  assert.match(mobileRules, /\.VPHomeHero::before\s*\{[\s\S]*?top:\s*76px;/)
})

test('public posterior downloads are not ignored by Git', () => {
  const path = 'docs/public/downloads/posteriors/11743-9102_nfw_param_cm200_samples.nc'
  const result = spawnSync('git', ['check-ignore', '-q', path], {
    cwd: repositoryRoot,
    encoding: 'utf8',
  })
  assert.equal(result.status, 1, `${path} is still ignored by Git`)
})

test('Pages build job grants configure-pages permission', () => {
  const workflow = readFileSync(join(repositoryRoot, '.github', 'workflows', 'deploy-docs.yml'), 'utf8')
  const buildJob = workflow.match(/\n  build:\n([\s\S]*?)\n  deploy:/)?.[1] ?? ''
  assert.match(buildJob, /permissions:\s*\n\s+contents:\s*read\s*\n\s+pages:\s*write/)
})

test('Pages workflow runs and watches Python documentation checks', () => {
  const workflow = readFileSync(join(repositoryRoot, '.github', 'workflows', 'deploy-docs.yml'), 'utf8')
  assert.match(workflow, /scripts\/docs\/\*\*/)
  assert.match(workflow, /tests\/test_\*\.py/)
  assert.match(workflow, /pip install[^\n]*h5py/)
  assert.match(workflow, /python -m unittest tests\.test_extract_case_summaries/)
  assert.match(workflow, /python -m unittest tests\.test_docs_mcmc_figures/)
})

test('postdeploy checks representative assets and exact posterior size', () => {
  const workflow = readFileSync(join(repositoryRoot, '.github', 'workflows', 'deploy-docs.yml'), 'utf8')
  const postdeploy = workflow.match(/\n  postdeploy:\n([\s\S]*)$/)?.[1] ?? ''
  assert.match(postdeploy, /\.css/)
  assert.match(postdeploy, /\.js/)
  assert.match(postdeploy, /assets\/case-studies\/11743-9102\/nfw-fit\.png/)
  assert.match(postdeploy, /88196/)
})

test('docs check runs both Node regression suites', () => {
  const packageJson = JSON.parse(readFileSync(join(repositoryRoot, 'package.json'), 'utf8'))
  assert.match(packageJson.scripts['docs:check'], /tests\/docs-check\.test\.mjs/)
  assert.match(packageJson.scripts['docs:check'], /tests\/built-docs-check\.test\.mjs/)
})

test('homepage workflow component uses direct-loadable HTML routes', () => {
  const component = readFileSync(
    join(repositoryRoot, 'docs', '.vitepress', 'theme', 'components', 'WorkflowMap.vue'),
    'utf8',
  )
  const hrefs = [...component.matchAll(/href:\s*'([^']+)'/g)].map((match) => match[1])
  assert.equal(hrefs.length, 4)
  for (const href of hrefs) assert.match(href, /\.html$/)
})

test('documentation toolchain ownership and Node runtime are recorded', () => {
  const agents = readFileSync(join(repositoryRoot, 'AGENTS.md'), 'utf8')
  const packageJson = JSON.parse(readFileSync(join(repositoryRoot, 'package.json'), 'utf8'))
  assert.match(agents, /VitePress public site/)
  assert.match(agents, /package\.json/)
  assert.equal(packageJson.engines?.node, '>=20')
  assert.equal(packageJson.scripts?.['docs:dev'], 'vitepress dev docs')
})

test('MCMC project-lessons link targets the existing priors anchor', () => {
  const page = readFileSync(join(repositoryRoot, 'docs', 'background', 'mcmc', 'project-lessons.md'), 'utf8')
  assert.match(page, /single-galaxy-nfw#priors\)/)
  assert.doesNotMatch(page, /single-galaxy-nfw#parameters-and-priors\)/)
})

test('deep case covers inputs, decomposition, and finalized posterior summary', () => {
  const page = readFileSync(join(repositoryRoot, 'docs', 'case-studies', '11743-9102.md'), 'utf8')
  assert.match(page, /## Input and velocity-field context/)
  assert.match(page, /## Rotation curve and component decomposition/)
  assert.match(page, /nfw-fit\.png/)
  assert.match(page, /## Finalized posterior artifact summary/)
  assert.match(page, /12\.4232/)
  assert.match(page, /-0\.6888/)
  assert.match(page, /## Legacy extended summary/)
})

test('case provenance pins the private scientific source inputs', () => {
  const provenance = JSON.parse(
    readFileSync(join(repositoryRoot, 'docs', 'public', 'meta', 'case-study-provenance.json'), 'utf8'),
  )
  assert.match(provenance.sourceRepository?.commit ?? '', /^[a-f\d]{40}$/)
  assert.equal(provenance.sourceRepository?.name, 'manga-dev')
  assert.equal(provenance.sourceRepository?.path, undefined)
  assert.match(provenance.sourceArchiveSha256 ?? '', /^[a-f\d]{64}$/)
  assert.match(provenance.sourceThesisSha256 ?? '', /^[a-f\d]{64}$/)
})

test('paper and implementation status use explicit paired component states', () => {
  const component = readFileSync(
    join(repositoryRoot, 'docs', '.vitepress', 'theme', 'components', 'MethodStatus.vue'),
    'utf8',
  )
  assert.match(component, /status:\s*'paper'\s*\|\s*'implementation'/)
  assert.match(component, /Implementation status/)

  const pages = [
    'docs/methods/index.md',
    'docs/methods/data-and-selection.md',
    'docs/methods/empirical-rotation-curves.md',
    'docs/methods/single-galaxy-nfw.md',
    'docs/methods/population-model.md',
    'docs/methods/diagnostics-and-quality-gates.md',
    'docs/run/index.md',
    'docs/run/configuration.md',
    'docs/run/cli-workflow.md',
  ]
  for (const page of pages) {
    const content = readFileSync(join(repositoryRoot, page), 'utf8')
    assert.match(content, /<MethodStatus status="paper">/, `${page} needs a Paper method state`)
    assert.match(content, /<MethodStatus status="implementation">/, `${page} needs a Current implementation state`)
    for (const block of content.matchAll(/<MethodStatus[\s\S]*?<\/MethodStatus>/g)) {
      assert.doesNotMatch(block[0], /\\\(|\\\)/, `${page} status cards must not expose raw math delimiters`)
    }
  }
})

test('small workflow numerals use the accessible gold text token', () => {
  const styles = readFileSync(join(repositoryRoot, 'docs', '.vitepress', 'theme', 'style.css'), 'utf8')
  const rule = styles.match(/\.workflow-map__number\s*\{([\s\S]*?)\}/)?.[1] ?? ''
  assert.match(rule, /color:\s*var\(--ledger-gold-text\)/)
})

test('installation guide records the h5py documentation-tool dependency', () => {
  const installation = readFileSync(join(repositoryRoot, 'docs', 'run', 'installation.md'), 'utf8')
  assert.match(installation, /h5py/)
})

test('project architecture records legacy documentation route migration', () => {
  const page = readFileSync(join(repositoryRoot, 'docs', 'project', 'architecture.md'), 'utf8')
  assert.match(page, /Data-Processing-Pipeline\.md/)
  assert.match(page, /future\/manga-dm-rc-shapes\.md/)
})
