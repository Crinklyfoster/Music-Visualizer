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

# Enhanced custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .memory-info {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.5rem 0;
        border: 1px solid #bbdefb;
    }
    .performance-warning {
        background: #ffebee;
        border: 1px solid #ffcdd2;
        color: #c62828;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .optimization-tip {
        background: #f3e5f5;
        border: 1px solid #e1bee7;
        color: #7b1fa2;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
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
    """Extract rhythm features separately"""
    features = {}
    
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=hop_length)
        
        features['Tempo'] = float(tempo)
        features['Beat_Count'] = int(len(beats))
        features['Beat_Density'] = float(len(beats) / (len(audio_data) / sr))
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr, hop_length=hop_length)
        features['Onset_Count'] = int(len(onset_frames))
        
        del beats, onset_frames
        cleanup_memory()
        
    except Exception as e:
        features['Tempo'] = 0.0
        features['Beat_Count'] = 0
        features['Beat_Density'] = 0.0
        features['Onset_Count'] = 0
        
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
        time_frames = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Downsample RMS if needed
        if len(rms) > 1000:
            rms = downsample_for_visualization(rms, max_points=1000)
            time_frames = downsample_for_visualization(time_frames, max_points=1000)
        
        fig.add_trace(go.Scatter(
            x=time_frames, y=rms,
            mode='lines', name='RMS',
            line=dict(color='#f5576c', width=2),
            hovertemplate='Time: %{x:.3f}s<br>RMS: %{y:.4f}<extra></extra>'
        ), row=2, col=1)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
        
        # Downsample spectral centroid if needed
        if len(spectral_centroid) > 1000:
            spectral_centroid = downsample_for_visualization(spectral_centroid, max_points=1000)
        
        fig.add_trace(go.Scatter(
            x=time_frames[:len(spectral_centroid)], y=spectral_centroid,
            mode='lines', name='Spectral Centroid',
            line=dict(color='#4facfe', width=2),
            hovertemplate='Time: %{x:.3f}s<br>Centroid: %{y:.1f}Hz<extra></extra>'
        ), row=3, col=1)
        
        fig.update_layout(
            height=700,
            title_text="Waveform Analysis",
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="RMS", row=2, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
        
        # Clean up
        del rms, spectral_centroid, time_frames, audio_viz, time_viz
        cleanup_memory()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waveform plot: {str(e)}")
        return None

@memory_monitor
def create_optimized_spectrum_plot(audio_data, sr):
    """Create memory-optimized spectrum plot"""
    try:
        # Adaptive parameters
        hop_length = adaptive_hop_length(len(audio_data), sr)
        n_fft = min(2048, len(audio_data) // 4)
        
        # Compute spectrograms with reduced resolution
        stft = librosa.stft(audio_data, hop_length=hop_length, n_fft=n_fft)
        magnitude = np.abs(stft)
        
        # Downsample for visualization
        if magnitude.shape[1] > 500:
            step = magnitude.shape[1] // 500
            magnitude = magnitude[:, ::step]
        
        # Mel spectrogram with reduced resolution
        n_mels = 64 if len(audio_data) / sr > 60 else 128
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Downsample mel spectrogram
        if mel_spec_db.shape[1] > 500:
            step = mel_spec_db.shape[1] // 500
            mel_spec_db = mel_spec_db[:, ::step]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Magnitude Spectrogram', 'Mel Spectrogram')
        )
        
        # Time and frequency axes
        time_frames = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Magnitude spectrogram
        fig.add_trace(go.Heatmap(
            z=librosa.amplitude_to_db(magnitude, ref=np.max),
            x=time_frames,
            y=freqs[:magnitude.shape[0]],
            colorscale='Viridis',
            name='Magnitude',
            hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.0f}Hz<br>Magnitude: %{z:.1f}dB<extra></extra>'
        ), row=1, col=1)
        
        # Mel spectrogram
        mel_time_frames = librosa.frames_to_time(np.arange(mel_spec_db.shape[1]), sr=sr, hop_length=hop_length)
        
        fig.add_trace(go.Heatmap(
            z=mel_spec_db,
            x=mel_time_frames,
            y=np.arange(mel_spec_db.shape[0]),
            colorscale='Plasma',
            name='Mel',
            hovertemplate='Time: %{x:.3f}s<br>Mel Band: %{y}<br>Power: %{z:.1f}dB<extra></extra>'
        ), row=1, col=2)
        
        fig.update_layout(
            height=500,
            title_text="Spectral Analysis",
            showlegend=False,
            template="plotly_white"
        )
        
        # Clean up
        del stft, magnitude, mel_spec, mel_spec_db, time_frames, freqs, mel_time_frames
        cleanup_memory()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating spectrum plot: {str(e)}")
        return None

@memory_monitor
def create_chroma_analysis_plot(audio_data, sr):
    """Create optimized chroma analysis plot"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute chroma with reduced resolution
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Downsample if needed
        if chroma.shape[1] > 500:
            step = chroma.shape[1] // 500
            chroma = chroma[:, ::step]
        
        time_frames = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
        chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Chromagram', 'Average Chroma Profile'),
            specs=[[{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # Chromagram
        fig.add_trace(go.Heatmap(
            z=chroma,
            x=time_frames,
            y=chroma_labels,
            colorscale='Blues',
            name='Chroma',
            hovertemplate='Time: %{x:.3f}s<br>Note: %{y}<br>Strength: %{z:.3f}<extra></extra>'
        ), row=1, col=1)
        
        # Average chroma profile
        chroma_mean = np.mean(chroma, axis=1)
        fig.add_trace(go.Bar(
            x=chroma_labels,
            y=chroma_mean,
            name='Average Chroma',
            marker_color='#667eea',
            hovertemplate='Note: %{x}<br>Average Strength: %{y:.3f}<extra></extra>'
        ), row=1, col=2)
        
        fig.update_layout(
            height=400,
            title_text="Chroma Analysis",
            showlegend=False,
            template="plotly_white"
        )
        
        # Clean up
        del chroma, time_frames, chroma_mean
        cleanup_memory()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chroma plot: {str(e)}")
        return None

@memory_monitor
def create_rhythm_analysis_plot(audio_data, sr):
    """Create optimized rhythm analysis plot"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr, hop_length=hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Beats and Onsets', 'Beat Intervals'),
            vertical_spacing=0.15
        )
        
        # Downsample audio for visualization
        audio_viz = downsample_for_visualization(audio_data, max_points=5000)
        time_viz = np.linspace(0, len(audio_data) / sr, len(audio_viz))
        
        # Waveform with beats and onsets
        fig.add_trace(go.Scatter(
            x=time_viz, y=audio_viz,
            mode='lines', name='Waveform',
            line=dict(color='lightgray', width=1),
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ), row=1, col=1)
        
        # Add beat markers (limit to first 50 for performance)
        for i, beat_time in enumerate(beat_times[:50]):
            fig.add_vline(
                x=beat_time, 
                line_dash="dash", 
                line_color="red", 
                line_width=1,
                row=1, col=1
            )
        
        # Beat intervals histogram
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            fig.add_trace(go.Histogram(
                x=beat_intervals,
                nbinsx=20,
                name='Beat Intervals',
                marker_color='#4facfe',
                hovertemplate='Interval: %{x:.3f}s<br>Count: %{y}<extra></extra>'
            ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            title_text=f"Rhythm Analysis (Tempo: {tempo:.1f} BPM)",
            showlegend=False,
            template="plotly_white"
        )
        
        # Clean up
        del beats, beat_times, onset_frames, onset_times, audio_viz, time_viz
        cleanup_memory()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating rhythm plot: {str(e)}")
        return None

def create_feature_comparison_plot(features_dict):
    """Create optimized feature comparison plot"""
    try:
        # Select key features for visualization
        key_features = {
            'Tempo': features_dict.get('Tempo', 0),
            'RMS_Energy': features_dict.get('RMS_Energy', 0),
            'Spectral_Centroid_Mean': features_dict.get('Spectral_Centroid_Mean', 0),
            'Zero_Crossing_Rate': features_dict.get('Zero_Crossing_Rate', 0),
            'Chroma_Mean': features_dict.get('Chroma_Mean', 0),
            'Spectral_Rolloff_Mean': features_dict.get('Spectral_Rolloff_Mean', 0)
        }
        
        # Normalize values for better visualization
        values = list(key_features.values())
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(key_features.keys()),
                y=normalized_values,
                text=[f'{v:.4f}' for v in values],
                textposition='auto',
                marker_color='#667eea',
                hovertemplate='Feature: %{x}<br>Value: %{text}<br>Normalized: %{y:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            height=400,
            title_text="Key Audio Features (Normalized)",
            template="plotly_white",
            xaxis_tickangle=-45
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating feature plot: {str(e)}")
        return None

def create_3d_feature_plot(features_dict):
    """Create optimized 3D feature visualization"""
    try:
        # Select three key features for 3D plot
        x_feature = features_dict.get('Spectral_Centroid_Mean', 0)
        y_feature = features_dict.get('RMS_Energy', 0)
        z_feature = features_dict.get('Tempo', 0)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=[x_feature],
            y=[y_feature],
            z=[z_feature],
            mode='markers',
            marker=dict(
                size=15,
                color='#667eea',
                opacity=0.8
            ),
            text=['Audio Sample'],
            hovertemplate='Spectral Centroid: %{x:.1f}Hz<br>RMS Energy: %{y:.4f}<br>Tempo: %{z:.1f}BPM<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Audio Feature Space',
            scene=dict(
                xaxis_title='Spectral Centroid (Hz)',
                yaxis_title='RMS Energy',
                zaxis_title='Tempo (BPM)'
            ),
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 3D plot: {str(e)}")
        return None

def main():
    # Main title
    st.markdown('<h1 class="main-header">🎵 Enhanced Audio Spectrum Visualizer</h1>', unsafe_allow_html=True)
    
    # Memory monitoring
    initial_memory = get_memory_usage()
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Memory display
    current_memory = get_memory_usage()
    memory_color = "red" if current_memory > 300 else "orange" if current_memory > 200 else "green"
    st.sidebar.markdown(f'''
    <div class="memory-info">
        💾 Memory Usage: <span style="color: {memory_color}"><strong>{current_memory:.1f} MB</strong></span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Performance settings
    st.sidebar.subheader("⚙️ Performance Settings")
    max_duration = st.sidebar.slider("Max Audio Duration (seconds)", 30, 300, 120)
    enable_advanced_features = st.sidebar.checkbox("Enable Advanced Features", value=True)
    
    if current_memory > 250:
        st.sidebar.markdown('''
        <div class="performance-warning">
            ⚠️ High memory usage detected. Consider refreshing the page or reducing analysis options.
        </div>
        ''', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC"
    )
    
    if uploaded_file is not None:
        # Load audio
        with st.spinner("🔄 Loading audio file..."):
            audio_data, sr, error = load_audio(uploaded_file)
        
        if audio_data is not None and sr is not None:
            # Optimize audio
            audio_data = optimize_audio_for_processing(audio_data, sr, max_duration)
            
            # Success message
            st.markdown("""
            <div class="success-box">
                <h3>✅ Audio loaded successfully!</h3>
                <p>Your audio file has been processed and optimized for analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Basic audio information
            st.header("📊 Audio Information")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                duration = len(audio_data) / sr
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Duration</h3>
                    <h2>{duration:.2f}s</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Sample Rate</h3>
                    <h2>{sr:,} Hz</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Samples</h3>
                    <h2>{len(audio_data):,}</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Channels</h3>
                    <h2>{channels}</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            with col5:
                file_size_mb = len(audio_data) * 4 / 1024 / 1024
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Size</h3>
                    <h2>{file_size_mb:.1f} MB</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            # Audio player
            st.subheader("🎵 Audio Player")
            st.audio(uploaded_file, format='audio/wav')
            
            # Extract features
            with st.spinner("🔍 Extracting audio features..."):
                features = extract_comprehensive_features(audio_data, sr)
            
            # Display key metrics
            st.header("🎯 Key Audio Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                tempo_value = features.get('Tempo', 0)
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Tempo</h3>
                    <h2>{tempo_value:.1f} BPM</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                centroid_value = features.get('Spectral_Centroid_Mean', 0)
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Spectral Centroid</h3>
                    <h2>{centroid_value:.0f} Hz</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                rms_value = features.get('RMS_Energy', 0)
                st.markdown(f'''
                <div class="metric-card">
                    <h3>RMS Energy</h3>
                    <h2>{rms_value:.4f}</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                zcr_value = features.get('Zero_Crossing_Rate', 0)
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Zero Crossing Rate</h3>
                    <h2>{zcr_value:.4f}</h2>
                </div>
                ''', unsafe_allow_html=True)
            
            # Analysis options
            st.sidebar.subheader("📊 Analysis Options")
            
            # Smart recommendations based on memory
            if current_memory > 200:
                st.sidebar.markdown('''
                <div class="optimization-tip">
                    💡 <strong>Tip:</strong> High memory usage detected. Consider selecting fewer options simultaneously.
                </div>
                ''', unsafe_allow_html=True)
                max_concurrent = 2
            else:
                max_concurrent = 4
            
            show_waveform = st.sidebar.checkbox("🌊 Waveform Analysis", value=True)
            show_spectrum = st.sidebar.checkbox("🌈 Spectral Analysis", value=True)
            show_chroma = st.sidebar.checkbox("🎼 Chroma Analysis", value=False)
            show_rhythm = st.sidebar.checkbox("🥁 Rhythm Analysis", value=False)
            show_features = st.sidebar.checkbox("📊 Feature Analysis", value=True)
            show_3d = st.sidebar.checkbox("🎨 3D Visualization", value=False)
            
            # Count enabled options
            enabled_options = sum([show_waveform, show_spectrum, show_chroma, show_rhythm, show_features, show_3d])
            
            if enabled_options > max_concurrent:
                st.sidebar.warning(f"⚠️ Too many options selected ({enabled_options}). Recommended: {max_concurrent} or fewer.")
            
            # Analysis sections
            if show_waveform:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🌊 Waveform Analysis")
                
                with st.spinner("Creating waveform visualization..."):
                    waveform_fig = create_optimized_waveform_plot(audio_data, sr)
                    if waveform_fig:
                        st.plotly_chart(waveform_fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            if show_spectrum:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🌈 Spectral Analysis")
                
                with st.spinner("Creating spectral visualization..."):
                    spectrum_fig = create_optimized_spectrum_plot(audio_data, sr)
                    if spectrum_fig:
                        st.plotly_chart(spectrum_fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            if show_chroma:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🎼 Chroma Analysis")
                
                with st.spinner("Creating chroma visualization..."):
                    chroma_fig = create_chroma_analysis_plot(audio_data, sr)
                    if chroma_fig:
                        st.plotly_chart(chroma_fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            if show_rhythm:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🥁 Rhythm Analysis")
                
                with st.spinner("Creating rhythm visualization..."):
                    rhythm_fig = create_rhythm_analysis_plot(audio_data, sr)
                    if rhythm_fig:
                        st.plotly_chart(rhythm_fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            if show_features:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("📊 Feature Analysis")
                
                # Feature summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Key Features")
                    key_features_display = {
                        'Spectral Centroid': f"{features.get('Spectral_Centroid_Mean', 0):.1f} Hz",
                        'RMS Energy': f"{features.get('RMS_Energy', 0):.4f}",
                        'Zero Crossing Rate': f"{features.get('Zero_Crossing_Rate', 0):.4f}",
                        'Tempo': f"{features.get('Tempo', 0):.1f} BPM",
                        'Spectral Rolloff': f"{features.get('Spectral_Rolloff_Mean', 0):.1f} Hz",
                        'Chroma Mean': f"{features.get('Chroma_Mean', 0):.4f}"
                    }
                    
                    for feature, value in key_features_display.items():
                        st.markdown(f'<div class="feature-box"><strong>{feature}:</strong> {value}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.subheader("📊 Statistical Summary")
                    stats = {
                        'Mean Amplitude': f"{features.get('Audio_Mean', 0):.4f}",
                        'Std Deviation': f"{features.get('Audio_Std', 0):.4f}",
                        'Skewness': f"{features.get('Audio_Skewness', 0):.4f}",
                        'Kurtosis': f"{features.get('Audio_Kurtosis', 0):.4f}",
                        'Dynamic Range': f"{features.get('Audio_Range', 0):.4f}",
                        'Beat Count': f"{features.get('Beat_Count', 0)}"
                    }
                    
                    for stat, value in stats.items():
                        st.markdown(f'<div class="feature-box"><strong>{stat}:</strong> {value}</div>', unsafe_allow_html=True)
                
                # Feature comparison plot
                with st.spinner("Creating feature comparison..."):
                    feature_fig = create_feature_comparison_plot(features)
                    if feature_fig:
                        st.plotly_chart(feature_fig, use_container_width=True)
                
                # Feature table
                if enable_advanced_features:
                    st.subheader("📋 Complete Feature Set")
                    feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                    feature_df['Value'] = feature_df['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
                    st.dataframe(feature_df, use_container_width=True, height=300)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            if show_3d:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🎨 3D Feature Visualization")
                
                with st.spinner("Creating 3D visualization..."):
                    fig_3d = create_3d_feature_plot(features)
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Export options
            st.sidebar.subheader("💾 Export Options")
            
            if st.sidebar.button("📊 Download Features CSV"):
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                csv = feature_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="💾 Download CSV File",
                    data=csv,
                    file_name=f"audio_features_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
            
            if st.sidebar.button("📋 Download Features JSON"):
                import json
                json_data = json.dumps(features, indent=2)
                st.sidebar.download_button(
                    label="💾 Download JSON File",
                    data=json_data,
                    file_name=f"audio_features_{uploaded_file.name.split('.')[0]}.json",
                    mime="application/json"
                )
            
            # Memory cleanup button
            if st.sidebar.button("🧹 Clean Memory"):
                cleanup_memory()
                st.sidebar.success("Memory cleaned!")
            
            # Analysis summary
            st.sidebar.subheader("📈 Analysis Summary")
            final_memory = get_memory_usage()
            memory_used = final_memory - initial_memory
            
            st.sidebar.markdown(f"""
            <div class="memory-info">
                <strong>Features Extracted:</strong> {len(features)}<br>
                <strong>Duration:</strong> {len(audio_data)/sr:.2f}s<br>
                <strong>Memory Used:</strong> {memory_used:.1f}MB<br>
                <strong>File Format:</strong> {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error(f"❌ Error loading audio file: {error}")
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="info-box">
            <h3>👆 Please upload an audio file to begin analysis</h3>
            <p>Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance information
        st.header("⚡ Performance Optimizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Speed Optimizations
            - **Adaptive Processing**: Automatically adjusts parameters based on audio length
            - **Smart Caching**: Caches expensive computations
            - **Memory Monitoring**: Real-time memory usage tracking
            - **Progressive Loading**: Shows progress during analysis
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ Stability Features
            - **Memory Cleanup**: Automatic garbage collection
            - **Error Handling**: Graceful error recovery
            - **Resource Management**: Prevents memory leaks
            - **Smart Recommendations**: Suggests optimal settings
            """)
        
        # Feature overview
        st.header("🎵 Analysis Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🌊 Waveform Analysis
            - Raw audio waveform visualization
            - RMS energy tracking over time
            - Spectral centroid evolution
            - Optimized for large files
            """)
        
        with col2:
            st.markdown("""
            ### 🌈 Spectral Analysis
            - High-resolution spectrograms
            - Mel-frequency analysis
            - Adaptive resolution based on file size
            - Memory-optimized processing
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Feature Extraction
            - 50+ comprehensive audio features
            - MFCC, chroma, and rhythm analysis
            - Statistical and spectral properties
            - Export to CSV/JSON formats
            """)

if __name__ == "__main__":
    main()
