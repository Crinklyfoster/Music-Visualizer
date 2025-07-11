# 🎵 Enhanced Audio Spectrum Visualizer - Production Ready
# Advanced real-time audio analysis with comprehensive time and frequency domain features

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
import gc
import psutil
import os
import time
from functools import wraps

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎵 Enhanced Audio Spectrum Visualizer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for styling with dark mode support
st.markdown("""
<style>
/* Dark mode compatible styles */
.stApp {
    color: var(--text-color);
    background-color: var(--background-color);
}

/* Custom styling for better visibility */
.metric-container {
    background-color: var(--secondary-background-color);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border-color);
    margin: 0.5rem 0;
}

.feature-card {
    background-color: var(--secondary-background-color);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}

/* Ensure text visibility in both themes */
.stMarkdown, .stText, div[data-testid="stMarkdownContainer"] {
    color: var(--text-color) !important;
}

/* Progress bar styling */
.stProgress > div > div > div > div {
    background-color: #667eea;
}

/* Custom metric styling */
.custom-metric {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

/* Feature importance styling */
.feature-importance {
    background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    padding: 0.8rem;
    border-radius: 0.5rem;
    color: white;
    margin: 0.3rem 0;
}

/* Audio quality indicator */
.audio-quality {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def memory_monitor(func):
    """Decorator to monitor memory usage of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        memory_before = get_memory_usage()
        result = func(*args, **kwargs)
        memory_after = get_memory_usage()
        memory_diff = memory_after - memory_before
        if memory_diff > 50:  # If function used more than 50MB
            st.sidebar.warning(f"⚠️ {func.__name__} used {memory_diff:.1f}MB")
        return result
    return wrapper

def cleanup_memory():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(700, 10, 10)

def downsample_for_visualization(data, max_points=5000):
    """Downsample data for visualization to prevent memory issues"""
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

def adaptive_hop_length(audio_length, sr):
    """Calculate adaptive hop length based on audio duration"""
    duration = audio_length / sr
    if duration > 120:  # > 2 minutes
        return 2048
    elif duration > 60:  # > 1 minute
        return 1024
    else:
        return 512

@st.cache_data(show_spinner=False)
def load_audio_cached(file_bytes, file_name):
    """Cached audio loading function"""
    try:
        audio_buffer = io.BytesIO(file_bytes)
        audio_data, sr = librosa.load(audio_buffer, sr=None)
        return audio_data, sr, None
    except Exception as e:
        return None, None, str(e)

def load_audio(file):
    """Load audio file with error handling"""
    try:
        file_bytes = file.read()
        file.seek(0)  # Reset file pointer
        return load_audio_cached(file_bytes, file.name)
    except Exception as e:
        return None, None, str(e)

def optimize_audio_for_processing(audio_data, sr, max_duration=180):
    """Optimize audio data for processing"""
    original_duration = len(audio_data) / sr
    if original_duration > max_duration:
        st.warning(f"⚠️ Audio duration ({original_duration:.1f}s) exceeds recommended limit ({max_duration}s). Trimming for performance.")
        max_samples = int(max_duration * sr)
        audio_data = audio_data[:max_samples]
    
    # Normalize audio
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data

@st.cache_data(show_spinner=False)
def extract_basic_features(audio_data, sr):
    """Extract basic features with memory optimization"""
    features = {}
    try:
        # Basic info
        features['Duration'] = float(len(audio_data) / sr)
        features['Sample_Rate'] = int(sr)
        features['Total_Samples'] = int(len(audio_data))
        
        # Adaptive parameters
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Temporal features
        zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=hop_length)[0]
        features['Zero_Crossing_Rate'] = float(np.mean(zcr))
        features['Zero_Crossing_Rate_Std'] = float(np.std(zcr))
        
        # Energy features
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        features['RMS_Energy'] = float(np.mean(rms))
        features['RMS_Energy_Std'] = float(np.std(rms))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
        features['Spectral_Centroid_Mean'] = float(np.mean(spectral_centroids))
        features['Spectral_Centroid_Std'] = float(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=hop_length)[0]
        features['Spectral_Rolloff_Mean'] = float(np.mean(spectral_rolloff))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=hop_length)[0]
        features['Spectral_Bandwidth_Mean'] = float(np.mean(spectral_bandwidth))
        
        # Clean up intermediate variables
        del zcr, rms, spectral_centroids, spectral_rolloff, spectral_bandwidth
        cleanup_memory()
        
    except Exception as e:
        st.error(f"Error in basic feature extraction: {str(e)}")
        features['Error'] = str(e)
    
    return features

@st.cache_data(show_spinner=False)
def extract_rhythm_features(audio_data, sr):
    """Extract rhythm features separately - FIXED VERSION"""
    features = {}
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Improved tempo and beat tracking with error handling
        try:
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, 
                sr=sr, 
                hop_length=hop_length,
                start_bpm=120,  # Better starting point
                tightness=100   # More stable tracking
            )
            features['Tempo'] = float(tempo)
            features['Beat_Count'] = int(len(beats))
            features['Beat_Density'] = float(len(beats) / (len(audio_data) / sr))
            
            # Beat consistency analysis
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
                beat_intervals = np.diff(beat_times)
                features['Beat_Consistency'] = float(1.0 - np.std(beat_intervals) / np.mean(beat_intervals))
            else:
                features['Beat_Consistency'] = 0.0
                
        except Exception as e:
            st.warning(f"Beat tracking failed: {str(e)}")
            features['Tempo'] = 0.0
            features['Beat_Count'] = 0
            features['Beat_Density'] = 0.0
            features['Beat_Consistency'] = 0.0
        
        # Improved onset detection
        try:
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, 
                sr=sr, 
                hop_length=hop_length,
                backtrack=True,
                pre_max=20,
                post_max=20,
                pre_avg=100,
                post_avg=100,
                delta=0.07,
                wait=10
            )
            features['Onset_Count'] = int(len(onset_frames))
            features['Onset_Density'] = float(len(onset_frames) / (len(audio_data) / sr))
            
            # Onset strength
            onset_strength = librosa.onset.onset_strength(
                y=audio_data, 
                sr=sr, 
                hop_length=hop_length
            )
            features['Onset_Strength_Mean'] = float(np.mean(onset_strength))
            features['Onset_Strength_Std'] = float(np.std(onset_strength))
            
        except Exception as e:
            st.warning(f"Onset detection failed: {str(e)}")
            features['Onset_Count'] = 0
            features['Onset_Density'] = 0.0
            features['Onset_Strength_Mean'] = 0.0
            features['Onset_Strength_Std'] = 0.0
        
        # Rhythm regularity
        try:
            tempogram = librosa.feature.tempogram(
                y=audio_data, 
                sr=sr, 
                hop_length=hop_length
            )
            features['Rhythm_Regularity'] = float(np.mean(tempogram))
            features['Rhythm_Complexity'] = float(np.std(tempogram))
        except Exception as e:
            features['Rhythm_Regularity'] = 0.0
            features['Rhythm_Complexity'] = 0.0
        
        cleanup_memory()
        
    except Exception as e:
        st.error(f"Error in rhythm feature extraction: {str(e)}")
        # Set default values
        features.update({
            'Tempo': 0.0,
            'Beat_Count': 0,
            'Beat_Density': 0.0,
            'Beat_Consistency': 0.0,
            'Onset_Count': 0,
            'Onset_Density': 0.0,
            'Onset_Strength_Mean': 0.0,
            'Onset_Strength_Std': 0.0,
            'Rhythm_Regularity': 0.0,
            'Rhythm_Complexity': 0.0
        })
    
    return features

@st.cache_data(show_spinner=False)
def extract_mfcc_features(audio_data, sr, n_mfcc=13):
    """Extract MFCC features separately"""
    features = {}
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Reduce MFCC count for long audio
        if len(audio_data) / sr > 60:
            n_mfcc = min(n_mfcc, 8)
        
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        
        for i in range(n_mfcc):
            features[f'MFCC_{i+1}_Mean'] = float(np.mean(mfccs[i]))
            features[f'MFCC_{i+1}_Std'] = float(np.std(mfccs[i]))
        
        del mfccs
        cleanup_memory()
        
    except Exception as e:
        st.error(f"Error in MFCC extraction: {str(e)}")
    
    return features

@st.cache_data(show_spinner=False)
def extract_chroma_features(audio_data, sr):
    """Extract chroma features separately"""
    features = {}
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hop_length)
        
        features['Chroma_Mean'] = float(np.mean(chroma))
        features['Chroma_Std'] = float(np.std(chroma))
        
        # Individual chroma features
        chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i, label in enumerate(chroma_labels):
            features[f'Chroma_{label}_Mean'] = float(np.mean(chroma[i]))
        
        del chroma
        cleanup_memory()
        
    except Exception as e:
        st.error(f"Error in chroma extraction: {str(e)}")
    
    return features

@st.cache_data(show_spinner=False)
def extract_statistical_features(audio_data):
    """Extract statistical features"""
    features = {}
    try:
        features['Audio_Mean'] = float(np.mean(audio_data))
        features['Audio_Std'] = float(np.std(audio_data))
        features['Audio_Skewness'] = float(skew(audio_data))
        features['Audio_Kurtosis'] = float(kurtosis(audio_data))
        features['Audio_Min'] = float(np.min(audio_data))
        features['Audio_Max'] = float(np.max(audio_data))
        features['Audio_Range'] = float(np.max(audio_data) - np.min(audio_data))
    except Exception as e:
        st.error(f"Error in statistical feature extraction: {str(e)}")
    
    return features

def extract_comprehensive_features(audio_data, sr):
    """Extract all features with progress tracking"""
    all_features = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Basic features
        status_text.text("Extracting basic features...")
        progress_bar.progress(20)
        basic_features = extract_basic_features(audio_data, sr)
        all_features.update(basic_features)
        
        # Rhythm features
        status_text.text("Analyzing rhythm...")
        progress_bar.progress(40)
        rhythm_features = extract_rhythm_features(audio_data, sr)
        all_features.update(rhythm_features)
        
        # MFCC features
        status_text.text("Computing MFCC...")
        progress_bar.progress(60)
        mfcc_features = extract_mfcc_features(audio_data, sr)
        all_features.update(mfcc_features)
        
        # Chroma features
        status_text.text("Analyzing harmony...")
        progress_bar.progress(80)
        chroma_features = extract_chroma_features(audio_data, sr)
        all_features.update(chroma_features)
        
        # Statistical features
        status_text.text("Computing statistics...")
        progress_bar.progress(90)
        stat_features = extract_statistical_features(audio_data)
        all_features.update(stat_features)
        
        progress_bar.progress(100)
        status_text.text("Feature extraction complete!")
        
        # Clean up progress indicators
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error in feature extraction: {str(e)}")
    
    return all_features

@memory_monitor
def create_optimized_waveform_plot(audio_data, sr):
    """Create memory-optimized waveform plot"""
    try:
        # Downsample for visualization
        audio_viz = downsample_for_visualization(audio_data, max_points=8000)
        time_viz = np.linspace(0, len(audio_data) / sr, len(audio_viz))
        
        # Adaptive hop length
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Waveform', 'RMS Energy', 'Spectral Centroid'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Waveform
        fig.add_trace(go.Scatter(
            x=time_viz, y=audio_viz,
            mode='lines', name='Waveform',
            line=dict(color='#667eea', width=1),
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ), row=1, col=1)
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        rms_viz = downsample_for_visualization(rms, max_points=2000)
        time_rms = np.linspace(0, len(audio_data) / sr, len(rms_viz))
        
        fig.add_trace(go.Scatter(
            x=time_rms, y=rms_viz,
            mode='lines', name='RMS Energy',
            line=dict(color='#f093fb', width=1),
            hovertemplate='Time: %{x:.3f}s<br>RMS: %{y:.4f}<extra></extra>'
        ), row=2, col=1)
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
        centroids_viz = downsample_for_visualization(spectral_centroids, max_points=2000)
        time_centroids = np.linspace(0, len(audio_data) / sr, len(centroids_viz))
        
        fig.add_trace(go.Scatter(
            x=time_centroids, y=centroids_viz,
            mode='lines', name='Spectral Centroid',
            line=dict(color='#f6d365', width=1),
            hovertemplate='Time: %{x:.3f}s<br>Centroid: %{y:.1f}Hz<extra></extra>'
        ), row=3, col=1)
        
        # Update layout with theme detection
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig.update_layout(
            height=600,
            title_text="Audio Analysis Overview",
            showlegend=False,
            template=template
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="RMS Energy", row=2, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
        
        # Clean up
        del rms, spectral_centroids
        cleanup_memory()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waveform plot: {str(e)}")
        return None

@memory_monitor
def create_spectrum_plot(audio_data, sr):
    """Create frequency spectrum plot"""
    try:
        # Compute FFT
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(fft), 1/sr)
        
        # Take only positive frequencies
        positive_freq_idx = frequency > 0
        frequency = frequency[positive_freq_idx]
        magnitude = magnitude[positive_freq_idx]
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Downsample for visualization
        freq_viz = downsample_for_visualization(frequency, max_points=5000)
        mag_viz = downsample_for_visualization(magnitude_db, max_points=5000)
        
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freq_viz, y=mag_viz,
            mode='lines', name='Frequency Spectrum',
            line=dict(color='#667eea', width=1),
            hovertemplate='Frequency: %{x:.0f}Hz<br>Magnitude: %{y:.1f}dB<extra></extra>'
        ))
        
        fig.update_layout(
            title="Frequency Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dB)",
            template=template,
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating spectrum plot: {str(e)}")
        return None

@memory_monitor
def create_spectrogram(audio_data, sr):
    """Create spectrogram plot"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute spectrogram
        D = librosa.stft(audio_data, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Downsample for visualization if needed
        if S_db.shape[1] > 1000:
            step = S_db.shape[1] // 1000
            S_db = S_db[:, ::step]
        
        # Create time and frequency axes
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        fig = go.Figure(data=go.Heatmap(
            z=S_db,
            x=times,
            y=freqs,
            colorscale='Viridis',
            hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.0f}Hz<br>Power: %{z:.1f}dB<extra></extra>'
        ))
        
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig.update_layout(
            title="Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template=template,
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating spectrogram: {str(e)}")
        return None

@memory_monitor
def create_mel_spectrogram(audio_data, sr):
    """Create mel spectrogram plot"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Downsample for visualization if needed
        if S_db.shape[1] > 1000:
            step = S_db.shape[1] // 1000
            S_db = S_db[:, ::step]
        
        # Create time and mel frequency axes
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
        mel_freqs = librosa.mel_frequencies(n_mels=S_db.shape[0])
        
        fig = go.Figure(data=go.Heatmap(
            z=S_db,
            x=times,
            y=mel_freqs,
            colorscale='Plasma',
            hovertemplate='Time: %{x:.3f}s<br>Mel Band: %{y}<br>Power: %{z:.1f}dB<extra></extra>'
        ))
        
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig.update_layout(
            title="Mel Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Mel Frequency",
            template=template,
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating mel spectrogram: {str(e)}")
        return None

@memory_monitor
def create_chroma_plot(audio_data, sr):
    """Create chromagram plot"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute chromagram
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Downsample for visualization if needed
        if chroma.shape[1] > 1000:
            step = chroma.shape[1] // 1000
            chroma = chroma[:, ::step]
        
        # Create time axis
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
        chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        fig = go.Figure(data=go.Heatmap(
            z=chroma,
            x=times,
            y=chroma_labels,
            colorscale='Blues',
            hovertemplate='Time: %{x:.3f}s<br>Note: %{y}<br>Strength: %{z:.3f}<extra></extra>'
        ))
        
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig.update_layout(
            title="Chromagram",
            xaxis_title="Time (s)",
            yaxis_title="Pitch Class",
            template=template,
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chromagram: {str(e)}")
        return None

def create_rhythm_analysis_plot(audio_data, sr):
    """Create comprehensive rhythm analysis plot"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Create subplots for rhythm analysis
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Onset Strength', 'Tempogram', 'Beat Tracking'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Onset strength
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=hop_length)
        times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=sr, hop_length=hop_length)
        
        fig.add_trace(go.Scatter(
            x=times, y=onset_strength,
            mode='lines', name='Onset Strength',
            line=dict(color='#ff6b6b', width=1),
            hovertemplate='Time: %{x:.3f}s<br>Strength: %{y:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Tempogram
        tempogram = librosa.feature.tempogram(y=audio_data, sr=sr, hop_length=hop_length)
        tempogram_times = librosa.frames_to_time(np.arange(tempogram.shape[1]), sr=sr, hop_length=hop_length)
        tempo_axis = librosa.tempo_frequencies(len(tempogram), hop_length=hop_length, sr=sr)
        
        fig.add_trace(go.Heatmap(
            z=tempogram,
            x=tempogram_times,
            y=tempo_axis,
            colorscale='Hot',
            showscale=False,
            hovertemplate='Time: %{x:.3f}s<br>Tempo: %{y:.1f}BPM<br>Strength: %{z:.3f}<extra></extra>'
        ), row=2, col=1)
        
        # Beat tracking
        try:
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=hop_length)
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            
            # Create beat visualization
            beat_strength = np.zeros_like(times)
            for beat_time in beat_times:
                closest_idx = np.argmin(np.abs(times - beat_time))
                if closest_idx < len(beat_strength):
                    beat_strength[closest_idx] = 1.0
            
            fig.add_trace(go.Scatter(
                x=times, y=beat_strength,
                mode='markers+lines', name='Beats',
                line=dict(color='#4ecdc4', width=2),
                marker=dict(size=8, color='#4ecdc4'),
                hovertemplate='Time: %{x:.3f}s<br>Beat: %{y}<extra></extra>'
            ), row=3, col=1)
            
        except Exception as e:
            st.warning(f"Beat tracking visualization failed: {str(e)}")
        
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig.update_layout(
            height=700,
            title_text="Rhythm Analysis",
            showlegend=False,
            template=template
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Strength", row=1, col=1)
        fig.update_yaxes(title_text="Tempo (BPM)", row=2, col=1)
        fig.update_yaxes(title_text="Beat", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating rhythm analysis plot: {str(e)}")
        return None

def create_feature_summary_plot(features):
    """Create feature summary visualization"""
    try:
        # Select key features for visualization
        key_features = {
            'Tempo': features.get('Tempo', 0),
            'Spectral Centroid': features.get('Spectral_Centroid_Mean', 0),
            'RMS Energy': features.get('RMS_Energy', 0),
            'Zero Crossing Rate': features.get('Zero_Crossing_Rate', 0),
            'Spectral Rolloff': features.get('Spectral_Rolloff_Mean', 0),
            'Spectral Bandwidth': features.get('Spectral_Bandwidth_Mean', 0)
        }
        
        # Normalize features for better visualization
        normalized_features = {}
        for key, value in key_features.items():
            if value > 0:
                if key == 'Tempo':
                    normalized_features[key] = min(value / 200, 1.0)  # Normalize tempo
                elif key == 'Spectral Centroid':
                    normalized_features[key] = min(value / 4000, 1.0)  # Normalize centroid
                elif key == 'Spectral Rolloff':
                    normalized_features[key] = min(value / 8000, 1.0)  # Normalize rolloff
                elif key == 'Spectral Bandwidth':
                    normalized_features[key] = min(value / 2000, 1.0)  # Normalize bandwidth
                else:
                    normalized_features[key] = min(value, 1.0)
            else:
                normalized_features[key] = 0
        
        # Create radar chart
        categories = list(normalized_features.keys())
        values = list(normalized_features.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Audio Features',
            line=dict(color='#667eea'),
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        
        template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Audio Feature Summary",
            template=template,
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating feature summary plot: {str(e)}")
        return None

def create_advanced_feature_plots(features):
    """Create advanced feature visualization plots"""
    try:
        # MFCC features plot
        mfcc_features = {k: v for k, v in features.items() if k.startswith('MFCC') and k.endswith('_Mean')}
        
        if mfcc_features:
            fig_mfcc = go.Figure()
            mfcc_names = list(mfcc_features.keys())
            mfcc_values = list(mfcc_features.values())
            
            fig_mfcc.add_trace(go.Bar(
                x=mfcc_names,
                y=mfcc_values,
                marker_color='#667eea',
                hovertemplate='MFCC: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ))
            
            template = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
            fig_mfcc.update_layout(
                title="MFCC Features",
                xaxis_title="MFCC Coefficient",
                yaxis_title="Mean Value",
                template=template,
                height=400
            )
            
            return fig_mfcc
        
    except Exception as e:
        st.error(f"Error creating advanced feature plots: {str(e)}")
        return None

def display_feature_metrics(features):
    """Display key metrics in a nice format"""
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Duration",
                value=f"{features.get('Duration', 0):.2f}s",
                delta=None
            )
            st.metric(
                label="Sample Rate",
                value=f"{features.get('Sample_Rate', 0):,} Hz",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Tempo",
                value=f"{features.get('Tempo', 0):.1f} BPM",
                delta=None
            )
            st.metric(
                label="Beat Count",
                value=f"{features.get('Beat_Count', 0)}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="RMS Energy",
                value=f"{features.get('RMS_Energy', 0):.4f}",
                delta=None
            )
            st.metric(
                label="Zero Crossing Rate",
                value=f"{features.get('Zero_Crossing_Rate', 0):.4f}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Spectral Centroid",
                value=f"{features.get('Spectral_Centroid_Mean', 0):.1f} Hz",
                delta=None
            )
            st.metric(
                label="Spectral Rolloff",
                value=f"{features.get('Spectral_Rolloff_Mean', 0):.1f} Hz",
                delta=None
            )
            
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def main():
    """Main application function"""
    st.title("🎵 Enhanced Audio Spectrum Visualizer")
    st.markdown("Upload an audio file to analyze its spectral characteristics and extract comprehensive features.")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Controls")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        st.sidebar.info(f"📁 File size: {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        # Load audio
        with st.spinner("Loading audio file..."):
            audio_data, sr, error = load_audio(uploaded_file)
        
        if error:
            st.error(f"❌ Error loading audio: {error}")
            return
        
        if audio_data is None:
            st.error("❌ Failed to load audio file")
            return
        
        # Optimize audio for processing
        audio_data = optimize_audio_for_processing(audio_data, sr)
        
        # Display basic audio info
        st.success("✅ Audio loaded successfully!")
        
        # Audio player
        st.subheader("🎧 Audio Player")
        uploaded_file.seek(0)  # Reset file pointer
        st.audio(uploaded_file, format='audio/wav')
        
        # Feature extraction
        st.subheader("🔍 Feature Extraction")
        
        if st.button("🚀 Analyze Audio", type="primary"):
            with st.spinner("Extracting features..."):
                features = extract_comprehensive_features(audio_data, sr)
            
            if features:
                # Display key metrics
                st.subheader("📊 Key Metrics")
                display_feature_metrics(features)
                
                # Create visualizations
                st.subheader("📈 Visualizations")
                
                # Tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "🌊 Waveform", "📊 Spectrum", "🔥 Spectrogram", 
                    "🎵 Mel Spectrogram", "🎹 Chromagram", "🥁 Rhythm", "📋 Summary"
                ])
                
                with tab1:
                    st.subheader("Waveform Analysis")
                    waveform_fig = create_optimized_waveform_plot(audio_data, sr)
                    if waveform_fig:
                        st.plotly_chart(waveform_fig, use_container_width=True)
                
                with tab2:
                    st.subheader("Frequency Spectrum")
                    spectrum_fig = create_spectrum_plot(audio_data, sr)
                    if spectrum_fig:
                        st.plotly_chart(spectrum_fig, use_container_width=True)
                
                with tab3:
                    st.subheader("Spectrogram")
                    spectrogram_fig = create_spectrogram(audio_data, sr)
                    if spectrogram_fig:
                        st.plotly_chart(spectrogram_fig, use_container_width=True)
                
                with tab4:
                    st.subheader("Mel Spectrogram")
                    mel_fig = create_mel_spectrogram(audio_data, sr)
                    if mel_fig:
                        st.plotly_chart(mel_fig, use_container_width=True)
                
                with tab5:
                    st.subheader("Chromagram")
                    chroma_fig = create_chroma_plot(audio_data, sr)
                    if chroma_fig:
                        st.plotly_chart(chroma_fig, use_container_width=True)
                
                with tab6:
                    st.subheader("Rhythm Analysis")
                    rhythm_fig = create_rhythm_analysis_plot(audio_data, sr)
                    if rhythm_fig:
                        st.plotly_chart(rhythm_fig, use_container_width=True)
                    
                    # Display rhythm metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Beat Consistency", f"{features.get('Beat_Consistency', 0):.3f}")
                    with col2:
                        st.metric("Onset Density", f"{features.get('Onset_Density', 0):.3f}")
                    with col3:
                        st.metric("Rhythm Regularity", f"{features.get('Rhythm_Regularity', 0):.3f}")
                
                with tab7:
                    st.subheader("Feature Summary")
                    summary_fig = create_feature_summary_plot(features)
                    if summary_fig:
                        st.plotly_chart(summary_fig, use_container_width=True)
                    
                    # MFCC visualization
                    mfcc_fig = create_advanced_feature_plots(features)
                    if mfcc_fig:
                        st.plotly_chart(mfcc_fig, use_container_width=True)
                    
                    # Feature table
                    st.subheader("📋 All Features")
                    feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                    feature_df['Value'] = feature_df['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
                    st.dataframe(feature_df, use_container_width=True)
                    
                    # Download features as CSV
                    csv = feature_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Features as CSV",
                        data=csv,
                        file_name=f"audio_features_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                # Memory usage info
                current_memory = get_memory_usage()
                st.sidebar.info(f"💾 Memory usage: {current_memory:.1f} MB")
        
        # Cleanup button
        if st.sidebar.button("🧹 Clean Memory"):
            cleanup_memory()
            st.sidebar.success("✅ Memory cleaned!")
    
    else:
        st.info("👆 Please upload an audio file to begin analysis")
        st.markdown("""
        ### 📝 Instructions:
        1. **Upload** an audio file using the file uploader above
        2. **Play** the audio to preview it
        3. **Click** the "Analyze Audio" button to extract features
        4. **Explore** different visualizations in the tabs
        5. **Download** the extracted features as CSV
        
        ### 🎵 Supported Features:
        - **Temporal Features**: RMS Energy, Zero Crossing Rate
        - **Spectral Features**: Spectral Centroid, Rolloff, Bandwidth
        - **Rhythmic Features**: Tempo, Beat Tracking, Onset Detection
        - **Harmonic Features**: Chromagram, Pitch Class Profiles
        - **Perceptual Features**: MFCC, Mel Spectrogram
        - **Statistical Features**: Mean, Std, Skewness, Kurtosis
        
        ### 📊 Visualizations:
        - **Waveform**: Time-domain representation
        - **Spectrum**: Frequency-domain analysis
        - **Spectrogram**: Time-frequency representation
        - **Mel Spectrogram**: Perceptually-weighted spectrogram
        - **Chromagram**: Pitch class energy distribution
        - **Rhythm Analysis**: Beat tracking and onset detection
        - **Feature Summary**: Radar chart of key features
        """)

if __name__ == "__main__":
    main()
