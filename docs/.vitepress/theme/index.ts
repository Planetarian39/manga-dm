import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import ApplicationHome from './components/ApplicationHome.vue'
import CaseDownload from './components/CaseDownload.vue'
import MethodStatus from './components/MethodStatus.vue'
import ScienceFigure from './components/ScienceFigure.vue'
import WorkflowMap from './components/WorkflowMap.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('ApplicationHome', ApplicationHome)
    app.component('CaseDownload', CaseDownload)
    app.component('MethodStatus', MethodStatus)
    app.component('ScienceFigure', ScienceFigure)
    app.component('WorkflowMap', WorkflowMap)
  },
} satisfies Theme
