import { defineConfig } from 'vitepress'

const repository = 'https://github.com/Planetarian39/manga-dm'
const canonicalRoot = 'https://planetarian39.github.io/manga-dm/'

export default defineConfig({
  title: 'MaNGA Dark Matter',
  titleTemplate: ':title | MaNGA Dark Matter',
  description: 'Hongyi Xu\'s computational astrophysics project: Bayesian dark-matter halo inference from MaNGA galaxy kinematics.',
  lang: 'en-US',
  base: '/manga-dm/',
  srcExclude: ['superpowers/**'],
  cleanUrls: false,
  lastUpdated: true,
  sitemap: {
    hostname: canonicalRoot,
  },
  transformHead({ pageData }) {
    const relativePath = pageData.relativePath.replaceAll('\\', '/')
    const publicPath = relativePath === 'index.md'
      ? ''
      : relativePath.endsWith('/index.md')
        ? relativePath.slice(0, -'index.md'.length)
        : relativePath.replace(/\.md$/, '.html')
    return [['link', { rel: 'canonical', href: new URL(publicPath, canonicalRoot).href }]]
  },
  head: [
    ['meta', { name: 'theme-color', content: '#102a43' }],
    ['meta', { name: 'color-scheme', content: 'light dark' }],
  ],
  markdown: {
    math: true,
    lineNumbers: true,
    image: {
      lazyLoading: true,
    },
  },
  themeConfig: {
    siteTitle: 'MaNGA / Dark Matter',
    nav: [
      { text: 'Overview', link: '/' },
      { text: 'Manuscript Preview', link: '/paper-preview/' },
      {
        text: 'Research',
        items: [
          {
            text: 'Start here',
            items: [
              { text: 'Two-minute overview', link: '/overview/project-overview.html' },
              { text: 'Scientific methods', link: '/methods/' },
            ],
          },
          {
            text: 'Evidence and interpretation',
            items: [
              { text: 'Worked example: 11743-9102', link: '/case-studies/11743-9102.html' },
              { text: 'Diagnostics and quality', link: '/methods/diagnostics-and-quality-gates.html' },
              { text: 'Limitations', link: '/project/limitations.html' },
            ],
          },
          {
            text: 'Background',
            items: [
              { text: 'MCMC and Bayesian inference', link: '/background/mcmc/' },
            ],
          },
        ],
      },
      {
        text: 'Reproducibility',
        items: [
          {
            text: 'Run and inspect',
            items: [
              { text: 'Run the pipeline', link: '/run/' },
              { text: 'Downloads and provenance', link: '/case-studies/downloads.html' },
            ],
          },
          {
            text: 'Implementation',
            items: [
              { text: 'Architecture', link: '/project/architecture.html' },
              { text: 'Method-to-code map', link: '/project/code-map.html' },
              { text: 'Implementation status', link: '/project/implementation-status.html' },
            ],
          },
          {
            text: 'Release record',
            items: [
              { text: 'Application snapshot', link: '/project/application-snapshot.html' },
            ],
          },
        ],
      },
      { text: 'About', link: '/about/' },
    ],
    sidebar: {
      '/methods/': [
        {
          text: 'Methods',
          items: [
            { text: 'Method map', link: '/methods/' },
            { text: 'Data and selection', link: '/methods/data-and-selection' },
            { text: 'Empirical rotation curves', link: '/methods/empirical-rotation-curves' },
            { text: 'Single-galaxy NFW model', link: '/methods/single-galaxy-nfw' },
            { text: 'Population model', link: '/methods/population-model' },
            { text: 'Diagnostics and quality gates', link: '/methods/diagnostics-and-quality-gates' },
          ],
        },
      ],
      '/case-studies/': [
        {
          text: 'Case studies',
          items: [
            { text: 'Case-study guide', link: '/case-studies/' },
            { text: '11743-9102 deep dive', link: '/case-studies/11743-9102' },
            { text: 'Supporting cases', link: '/case-studies/supporting-cases' },
            { text: 'Downloads and provenance', link: '/case-studies/downloads' },
          ],
        },
      ],
      '/run/': [
        {
          text: 'Run the pipeline',
          items: [
            { text: 'Workflow at a glance', link: '/run/' },
            { text: 'Installation', link: '/run/installation' },
            { text: 'CLI workflow', link: '/run/cli-workflow' },
            { text: 'Configuration', link: '/run/configuration' },
            { text: 'Inputs and outputs', link: '/run/inputs-and-outputs' },
          ],
        },
      ],
      '/background/': [
        {
          text: 'MCMC background',
          items: [
            { text: 'MCMC overview', link: '/background/mcmc/' },
            { text: 'Bayesian foundations', link: '/background/mcmc/bayesian-foundations' },
            { text: 'Priors and data', link: '/background/mcmc/priors-and-data' },
            { text: 'Sampling and diagnostics', link: '/background/mcmc/sampling-and-diagnostics' },
            { text: 'Optimization and MCMC', link: '/background/mcmc/optimization-vs-mcmc' },
            { text: 'Project lessons', link: '/background/mcmc/project-lessons' },
          ],
        },
      ],
      '/project/': [
        {
          text: 'Project',
          items: [
            { text: 'Architecture', link: '/project/architecture' },
            { text: 'Method-to-code map', link: '/project/code-map' },
            { text: 'Implementation status', link: '/project/implementation-status' },
            { text: 'Limitations', link: '/project/limitations' },
            { text: 'Future research', link: '/project/future-research' },
            { text: 'Application snapshot', link: '/project/application-snapshot' },
          ],
        },
      ],
      '/about/': [
        {
          text: 'Researcher',
          items: [
            { text: 'About Hongyi Xu', link: '/about/' },
            { text: 'Two-minute overview', link: '/overview/project-overview' },
            { text: 'Application snapshot', link: '/project/application-snapshot' },
          ],
        },
      ],
    },
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        miniSearch: {
          searchOptions: {
            boost: { title: 4, text: 1.5, titles: 2 },
            fuzzy: 0.15,
            prefix: true,
          },
        },
      },
    },
    outline: {
      level: [2, 3],
      label: 'On this page',
    },
    socialLinks: [{ icon: 'github', link: repository }],
    editLink: {
      pattern: `${repository}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub',
    },
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'medium',
      },
    },
    docFooter: {
      prev: 'Previous',
      next: 'Next',
    },
    footer: {
      message: 'A computational astrophysics research project by Hongyi Xu, Department of Physics, University of Toronto.',
      copyright: 'Sample-size milestones, methods, software, and allowlisted case artifacts are public; aggregate scientific findings remain unpublished.',
    },
  },
})
