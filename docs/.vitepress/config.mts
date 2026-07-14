import { defineConfig } from 'vitepress'

const repository = 'https://github.com/Planetarian39/manga-dm'

export default defineConfig({
  title: 'MaNGA Dark Matter',
  titleTemplate: ':title | MaNGA Dark Matter',
  description: 'Methods, implementation notes, and reproducible workflows for the MaNGA dark-matter analysis pipeline.',
  lang: 'en-US',
  base: '/manga-dm/',
  srcExclude: ['superpowers/**'],
  cleanUrls: false,
  lastUpdated: true,
  sitemap: {
    hostname: 'https://planetarian39.github.io/manga-dm/',
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
      { text: 'Methods', link: '/methods/' },
      { text: 'Case studies', link: '/case-studies/' },
      { text: 'Run the pipeline', link: '/run/' },
      { text: 'Background', link: '/background/mcmc/' },
      { text: 'Project', link: '/project/architecture' },
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
      message: 'Methods documentation for the MaNGA dark-matter analysis pipeline.',
      copyright: 'Released with the project source; unpublished aggregate findings are intentionally excluded.',
    },
  },
})
