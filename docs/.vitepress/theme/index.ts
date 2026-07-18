import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import Layout from './Layout.vue'
import ApplicationHome from './components/ApplicationHome.vue'
import CaseDownload from './components/CaseDownload.vue'
import MethodStatus from './components/MethodStatus.vue'
import ManuscriptPreview from './components/ManuscriptPreview.vue'
import ManuscriptPreviewGallery from './components/ManuscriptPreviewGallery.vue'
import ScienceFigure from './components/ScienceFigure.vue'
import WorkflowMap from './components/WorkflowMap.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('ApplicationHome', ApplicationHome)
    app.component('CaseDownload', CaseDownload)
    app.component('MethodStatus', MethodStatus)
    app.component('ManuscriptPreview', ManuscriptPreview)
    app.component('ManuscriptPreviewGallery', ManuscriptPreviewGallery)
    app.component('ScienceFigure', ScienceFigure)
    app.component('WorkflowMap', WorkflowMap)
  },
} satisfies Theme
