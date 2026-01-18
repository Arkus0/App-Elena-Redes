import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './stores/authStore'
import Layout from './components/Layout'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import Onboarding from './pages/Onboarding'
import Competitors from './pages/Competitors'
import CompetitorAnalysis from './pages/CompetitorAnalysis'
import ContentCalendar from './pages/ContentCalendar'
import ContentGenerator from './pages/ContentGenerator'
import ViralScanner from './pages/ViralScanner'
import ContentDetail from './pages/ContentDetail'
import MyProfileSettings from './pages/MyProfileSettings'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />
}

function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      {/* Protected routes */}
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="onboarding" element={<Onboarding />} />
        <Route path="competitors" element={<Competitors />} />
        <Route path="competitors/:competitorId" element={<CompetitorAnalysis />} />
        <Route path="calendar" element={<ContentCalendar />} />
        <Route path="calendar/new" element={<ContentGenerator />} />
        <Route path="content/:contentId" element={<ContentDetail />} />
        <Route path="viral" element={<ViralScanner />} />
        <Route path="my-profile" element={<MyProfileSettings />} />
      </Route>

      {/* Catch all */}
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  )
}

export default App
