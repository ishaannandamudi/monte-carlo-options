import { QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './app/App'
import { createQueryClient } from './app/queryClient'
import './styles/tokens.css'
import './styles/global.css'
import './styles/components.css'
import './styles/layout.css'

const rootElement = document.getElementById('root')
if (!rootElement) throw new Error('Missing application root element')

const queryClient = createQueryClient()

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
