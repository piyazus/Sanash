import { createSlice, PayloadAction, createAsyncThunk } from '@reduxjs/toolkit'
import { api } from '../services/api'

interface Job {
    id: number
    videoId: number
    status: string
    progress: number
    currentFrame: number
    totalFrames: number | null
    createdAt: string
}

interface JobsState {
    items: Job[]
    currentJob: Job | null
    loading: boolean
    error: string | null
}

const initialState: JobsState = {
    items: [],
    currentJob: null,
    loading: false,
    error: null,
}

export const fetchJobs = createAsyncThunk('jobs/fetchAll', async () => {
    const response = await api.get('/api/v1/jobs')
    return response.data
})

export const fetchJob = createAsyncThunk('jobs/fetchOne', async (id: number) => {
    const response = await api.get(`/api/v1/jobs/${id}`)
    return response.data
})

const jobsSlice = createSlice({
    name: 'jobs',
    initialState,
    reducers: {
        updateJobProgress: (state, action: PayloadAction<{ id: number; progress: number; currentFrame: number }>) => {
            const job = state.items.find(j => j.id === action.payload.id)
            if (job) {
                job.progress = action.payload.progress
                job.currentFrame = action.payload.currentFrame
            }
            if (state.currentJob?.id === action.payload.id) {
                state.currentJob.progress = action.payload.progress
                state.currentJob.currentFrame = action.payload.currentFrame
            }
        },
        updateJobStatus: (state, action: PayloadAction<{ id: number; status: string }>) => {
            const job = state.items.find(j => j.id === action.payload.id)
            if (job) {
                job.status = action.payload.status
            }
            if (state.currentJob?.id === action.payload.id) {
                state.currentJob.status = action.payload.status
            }
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(fetchJobs.pending, (state) => {
                state.loading = true
            })
            .addCase(fetchJobs.fulfilled, (state, action) => {
                state.loading = false
                state.items = action.payload
            })
            .addCase(fetchJobs.rejected, (state, action) => {
                state.loading = false
                state.error = action.error.message || 'Failed to fetch jobs'
            })
            .addCase(fetchJob.fulfilled, (state, action) => {
                state.currentJob = action.payload
            })
    },
})

export const { updateJobProgress, updateJobStatus } = jobsSlice.actions
export default jobsSlice.reducer
