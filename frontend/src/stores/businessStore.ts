import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Business, BusinessStatus, Competitor, ContentCalendar } from '../types'

interface BusinessState {
  currentBusiness: Business | null
  businesses: Business[]
  competitors: Competitor[]
  calendars: ContentCalendar[]
  status: BusinessStatus | null
  isLoading: boolean

  setCurrentBusiness: (business: Business | null) => void
  setBusinesses: (businesses: Business[]) => void
  setCompetitors: (competitors: Competitor[]) => void
  setCalendars: (calendars: ContentCalendar[]) => void
  setStatus: (status: BusinessStatus | null) => void
  setLoading: (loading: boolean) => void
  addCompetitor: (competitor: Competitor) => void
  removeCompetitor: (competitorId: number) => void
  updateCompetitor: (competitor: Competitor) => void
  reset: () => void
}

export const useBusinessStore = create<BusinessState>()(
  persist(
    (set) => ({
      currentBusiness: null,
      businesses: [],
      competitors: [],
      calendars: [],
      status: null,
      isLoading: false,

      setCurrentBusiness: (business) => set({ currentBusiness: business }),
      setBusinesses: (businesses) => set({ businesses }),
      setCompetitors: (competitors) => set({ competitors }),
      setCalendars: (calendars) => set({ calendars }),
      setStatus: (status) => set({ status }),
      setLoading: (isLoading) => set({ isLoading }),

      addCompetitor: (competitor) =>
        set((state) => ({
          competitors: [...state.competitors, competitor],
        })),

      removeCompetitor: (competitorId) =>
        set((state) => ({
          competitors: state.competitors.filter((c) => c.id !== competitorId),
        })),

      updateCompetitor: (competitor) =>
        set((state) => ({
          competitors: state.competitors.map((c) =>
            c.id === competitor.id ? competitor : c
          ),
        })),

      reset: () =>
        set({
          currentBusiness: null,
          businesses: [],
          competitors: [],
          calendars: [],
          status: null,
          isLoading: false,
        }),
    }),
    {
      name: 'brandpulse-business',
    }
  )
)
