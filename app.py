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
        font-size: 2.5rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .memory-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .success-message {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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

def clear_all_memory():
    """Comprehensive memory cleanup function"""
    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Reset garbage collection thresholds
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(700, 10, 10)
        
        # Clear matplotlib figures
        plt.close('all')
        
        return True
    except Exception as e:
        st.error(f"Error during memory cleanup: {str(e)}")
        return False

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

def safe_format_value(value, format_spec=".3f"):
    """Safely format a value that might be a NumPy array"""
    try:
        if hasattr(value, 'item'):  # NumPy scalar
            return f"{value.item():{format_spec}}"
        elif hasattr(value, '__len__') and len(value) == 1:  # Single-element array
            return f"{value[0]:{format_spec}}"
        elif isinstance(value, (np.ndarray, list)) and len(value) > 1:
            return f"{float(np.mean(value)):{format_spec}}"
        else:
            return f"{float(value):{format_spec}}"
    except:
        return str(value)

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
        
        # Ensure tempo is a scalar
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item() if tempo.size == 1 else float(tempo[0])
        
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
        rms_viz = downsample_for_visualization(rms, max_points=2000)
        time_rms = np.linspace(0, len(audio_data) / sr, len(rms_viz))
        
        fig.add_trace(go.Scatter(
            x=time_rms, y=rms_viz,
            mode='lines', name='RMS Energy',
            line=dict(color='#f093fb', width=1.5),
            hovertemplate='Time: %{x:.3f}s<br>RMS: %{y:.4f}<extra></extra>'
        ), row=2, col=1)
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
        centroids_viz = downsample_for_visualization(spectral_centroids, max_points=2000)
        time_centroids = np.linspace(0, len(audio_data) / sr, len(centroids_viz))
        
        fig.add_trace(go.Scatter(
            x=time_centroids, y=centroids_viz,
            mode='lines', name='Spectral Centroid',
            line=dict(color='#f6d365', width=1.5),
            hovertemplate='Time: %{x:.3f}s<br>Centroid: %{y:.1f}Hz<extra></extra>'
        ), row=3, col=1)
        
        fig.update_layout(
            title="Audio Analysis Overview",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="RMS Energy", row=2, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waveform plot: {str(e)}")
        return None

@memory_monitor
def create_optimized_spectrogram(audio_data, sr):
    """Create memory-optimized spectrogram"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute spectrogram
        D = librosa.stft(audio_data, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Downsample for visualization
        if S_db.shape[1] > 1000:
            step = S_db.shape[1] // 1000
            S_db = S_db[:, ::step]
        
        # Create time and frequency axes
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)[:S_db.shape[0]]
        
        fig = go.Figure(data=go.Heatmap(
            z=S_db,
            x=times,
            y=freqs,
            colorscale='Viridis',
            hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.0f}Hz<br>Magnitude: %{z:.1f}dB<extra></extra>'
        ))
        
        fig.update_layout(
            title="Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=500,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating spectrogram: {str(e)}")
        return None

@memory_monitor
def create_rhythm_plot(audio_data, sr, rhythm_features):
    """Create rhythm analysis plot with proper formatting"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Extract tempo and beats
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Ensure tempo is scalar for formatting
        tempo_scalar = float(tempo.item() if hasattr(tempo, 'item') else tempo)
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        
        # Create waveform for context
        audio_viz = downsample_for_visualization(audio_data, max_points=8000)
        time_viz = np.linspace(0, len(audio_data) / sr, len(audio_viz))
        
        fig = go.Figure()
        
        # Add waveform
        fig.add_trace(go.Scatter(
            x=time_viz, y=audio_viz,
            mode='lines', name='Waveform',
            line=dict(color='lightblue', width=1),
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ))
        
        # Add beat markers
        if len(beat_times) > 0:
            beat_amplitudes = np.interp(beat_times, time_viz, audio_viz)
            fig.add_trace(go.Scatter(
                x=beat_times, y=beat_amplitudes,
                mode='markers', name='Beats',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate=f'Beat at %{{x:.3f}}s<br>Tempo: {tempo_scalar:.1f}BPM<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Rhythm Analysis - Tempo: {tempo_scalar:.1f} BPM",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating rhythm plot: {str(e)}")
        return None

@memory_monitor
def create_mel_spectrogram(audio_data, sr):
    """Create mel spectrogram"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Downsample for visualization
        if mel_spec_db.shape[1] > 1000:
            step = mel_spec_db.shape[1] // 1000
            mel_spec_db = mel_spec_db[:, ::step]
        
        # Create time axis
        times = librosa.frames_to_time(np.arange(mel_spec_db.shape[1]), sr=sr, hop_length=hop_length)
        
        fig = go.Figure(data=go.Heatmap(
            z=mel_spec_db,
            x=times,
            y=list(range(mel_spec_db.shape[0])),
            colorscale='Plasma',
            hovertemplate='Time: %{x:.3f}s<br>Mel Band: %{y}<br>Power: %{z:.1f}dB<extra></extra>'
        ))
        
        fig.update_layout(
            title="Mel Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Mel Bands",
            height=500,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating mel spectrogram: {str(e)}")
        return None

@memory_monitor
def create_chroma_plot(audio_data, sr):
    """Create chroma plot"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute chroma
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Downsample for visualization
        if chroma.shape[1] > 1000:
            step = chroma.shape[1] // 1000
            chroma = chroma[:, ::step]
        
        # Create time axis
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
        
        # Chroma labels
        chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        fig = go.Figure(data=go.Heatmap(
            z=chroma,
            x=times,
            y=chroma_labels,
            colorscale='Blues',
            hovertemplate='Time: %{x:.3f}s<br>Note: %{y}<br>Strength: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Chromagram",
            xaxis_title="Time (s)",
            yaxis_title="Pitch Class",
            height=400,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chroma plot: {str(e)}")
        return None

def create_feature_summary_table(features):
    """Create a summary table of extracted features"""
    try:
        # Create summary data
        summary_data = []
        
        # Basic info
        if 'Duration' in features:
            summary_data.append(['Duration', safe_format_value(features['Duration'], '.2f'), 'seconds'])
        if 'Sample_Rate' in features:
            summary_data.append(['Sample Rate', f"{int(features['Sample_Rate'])}", 'Hz'])
        if 'Tempo' in features:
            summary_data.append(['Tempo', safe_format_value(features['Tempo'], '.1f'), 'BPM'])
        if 'Beat_Count' in features:
            summary_data.append(['Beat Count', f"{int(features['Beat_Count'])}", 'beats'])
        if 'RMS_Energy' in features:
            summary_data.append(['RMS Energy', safe_format_value(features['RMS_Energy'], '.4f'), 'amplitude'])
        if 'Spectral_Centroid_Mean' in features:
            summary_data.append(['Spectral Centroid', safe_format_value(features['Spectral_Centroid_Mean'], '.1f'), 'Hz'])
        
        # Create DataFrame
        df = pd.DataFrame(summary_data, columns=['Feature', 'Value', 'Unit'])
        
        return df
        
    except Exception as e:
        st.error(f"Error creating feature summary: {str(e)}")
        return pd.DataFrame()

def create_comprehensive_feature_table(features):
    """Create comprehensive feature table"""
    try:
        # Convert all features to properly formatted strings
        formatted_features = []
        
        for key, value in features.items():
            if key == 'Error':
                continue
                
            try:
                if isinstance(value, (int, np.integer)):
                    formatted_value = str(int(value))
                elif isinstance(value, (float, np.floating)):
                    formatted_value = f"{float(value):.6f}"
                elif hasattr(value, 'item'):
                    formatted_value = f"{value.item():.6f}"
                else:
                    formatted_value = str(value)
                    
                formatted_features.append([key, formatted_value])
                
            except Exception as e:
                formatted_features.append([key, str(value)])
        
        df = pd.DataFrame(formatted_features, columns=['Feature', 'Value'])
        return df
        
    except Exception as e:
        st.error(f"Error creating comprehensive feature table: {str(e)}")
        return pd.DataFrame()

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🎵 Enhanced Audio Spectrum Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Options")
    
    # Memory usage display with refresh button
    memory_usage = get_memory_usage()
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    with col2:
        if st.button("🔄", help="Refresh memory usage"):
            st.rerun()
    
    # Memory cleanup button
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Memory & Cache", 
                        help="Clear all cached data and free memory", 
                        type="primary"):
        with st.spinner("Clearing memory and cache..."):
            success = clear_all_memory()
            if success:
                st.sidebar.success("✅ Memory cleared successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error("❌ Error clearing memory")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
        help="Upload an audio file for analysis (Limit: 200MB per file)"
    )
    
    if uploaded_file is not None:
        # Load audio
        with st.spinner("Loading audio file..."):
            audio_data, sr, error = load_audio(uploaded_file)
        
        if error:
            st.error(f"Error loading audio: {error}")
            return
        
        if audio_data is None:
            st.error("Failed to load audio file")
            return
        
        # Optimize audio
        audio_data = optimize_audio_for_processing(audio_data, sr)
        
        # Display basic info
        st.markdown('<div class="success-message">✅ Audio loaded successfully!</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{len(audio_data) / sr:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("Samples", f"{len(audio_data):,}")
        
        # Analysis options
        st.sidebar.subheader("🔧 Analysis Settings")
        
        show_waveform = st.sidebar.checkbox("Waveform Analysis", value=True)
        show_spectrogram = st.sidebar.checkbox("Spectrogram", value=True)
        show_mel_spectrogram = st.sidebar.checkbox("Mel Spectrogram", value=False)
        show_rhythm = st.sidebar.checkbox("Rhythm Analysis", value=True)
        show_chroma = st.sidebar.checkbox("Chromagram", value=False)
        show_features = st.sidebar.checkbox("Feature Extraction", value=True)
        
        # Extract features if requested
        if show_features:
            st.header("📈 Feature Analysis")
            with st.spinner("Extracting comprehensive features..."):
                features = extract_comprehensive_features(audio_data, sr)
            
            if features:
                # Feature summary
                st.subheader("📊 Feature Summary")
                summary_df = create_feature_summary_table(features)
                if not summary_df.empty:
                    st.dataframe(summary_df, use_container_width=True)
                
                # Comprehensive features
                with st.expander("🔍 All Features"):
                    comprehensive_df = create_comprehensive_feature_table(features)
                    if not comprehensive_df.empty:
                        st.dataframe(comprehensive_df, use_container_width=True)
                        
                        # Download button
                        csv = comprehensive_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Features as CSV",
                            data=csv,
                            file_name=f"audio_features_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
        
        # Visualizations
        st.header("📊 Audio Visualizations")
        
        # Waveform plot
        if show_waveform:
            st.subheader("🌊 Waveform Analysis")
            with st.spinner("Creating waveform plot..."):
                waveform_fig = create_optimized_waveform_plot(audio_data, sr)
                if waveform_fig:
                    st.plotly_chart(waveform_fig, use_container_width=True)
        
        # Spectrogram
        if show_spectrogram:
            st.subheader("🎼 Spectrogram")
            with st.spinner("Computing spectrogram..."):
                spec_fig = create_optimized_spectrogram(audio_data, sr)
                if spec_fig:
                    st.plotly_chart(spec_fig, use_container_width=True)
        
        # Mel Spectrogram
        if show_mel_spectrogram:
            st.subheader("🎵 Mel Spectrogram")
            with st.spinner("Computing mel spectrogram..."):
                mel_fig = create_mel_spectrogram(audio_data, sr)
                if mel_fig:
                    st.plotly_chart(mel_fig, use_container_width=True)
        
        # Rhythm analysis
        if show_rhythm:
            st.subheader("🥁 Rhythm Analysis")
            with st.spinner("Analyzing rhythm..."):
                rhythm_features = extract_rhythm_features(audio_data, sr)
                rhythm_fig = create_rhythm_plot(audio_data, sr, rhythm_features)
                if rhythm_fig:
                    st.plotly_chart(rhythm_fig, use_container_width=True)
        
        # Chroma analysis
        if show_chroma:
            st.subheader("🎹 Chromagram")
            with st.spinner("Computing chromagram..."):
                chroma_fig = create_chroma_plot(audio_data, sr)
                if chroma_fig:
                    st.plotly_chart(chroma_fig, use_container_width=True)
        
        # Analysis completion message with memory cleanup option
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("🎉 Analysis completed successfully!")
        with col2:
            if st.button("🧹 Free Memory", 
                        help="Clear analysis data and free memory",
                        type="secondary"):
                with st.spinner("Freeing memory..."):
                    clear_all_memory()
                    st.success("Memory freed!")
                    time.sleep(1)
                    st.rerun()
        
        # Memory cleanup
        cleanup_memory()
        
    else:
        # Welcome message
        st.markdown("""
        <div class="feature-card">
            <h3>🎵 Welcome to the Enhanced Audio Spectrum Visualizer</h3>
            <p>Upload an audio file to begin comprehensive analysis including:</p>
            <ul>
                <li>🌊 Advanced waveform visualization with RMS energy and spectral centroid</li>
                <li>🎼 High-resolution spectrograms and mel spectrograms</li>
                <li>🥁 Precise rhythm and tempo analysis with beat detection</li>
                <li>🎹 Harmonic content analysis with chromagrams</li>
                <li>📊 Comprehensive feature extraction (MFCC, statistical features)</li>
                <li>💾 Memory optimization and management tools</li>
            </ul>
            <p><strong>New Features:</strong></p>
            <ul>
                <li>🧹 Memory cleanup and cache clearing</li>
                <li>📈 Real-time memory usage monitoring</li>
                <li>⚡ Adaptive processing for different audio lengths</li>
                <li>📥 Export features to CSV</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("📁 Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC (Max: 200MB)")
        st.info("🎵 Optimized for audio files up to 3 minutes for best performance")

if __name__ == "__main__":
    main()
