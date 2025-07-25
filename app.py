# 🎵 Enhanced Audio Spectrum Visualizer v3.0 - Complete with Audio Playback
# Advanced real-time audio analysis optimized for songs up to 6 minutes

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
import base64

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
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    .audio-player-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
        text-align: center;
    }
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .memory-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3);
        transition: all 0.3s ease;
    }
    .memory-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    .success-message {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 184, 148, 0.3);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .analysis-complete {
        background: linear-gradient(135deg, #00b894 0%, #55a3ff 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .optimization-info {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .audio-controls {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
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
        if memory_diff > 100:  # Increased threshold for 6-minute songs
            st.sidebar.warning(f"⚠️ {func.__name__} used {memory_diff:.1f}MB")
        return result
    return wrapper

def cleanup_memory():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(1000, 15, 15)  # Adjusted for longer audio

def clear_all_memory():
    """Enhanced memory cleanup function for 6-minute audio processing"""
    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        
        # Clear matplotlib figures
        plt.close('all')
        
        # Force multiple garbage collections
        for _ in range(7):  # Increased for longer audio
            gc.collect()
        
        # Reset garbage collection thresholds
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(1000, 15, 15)
        
        # Clear global variables (safely)
        import sys
        current_module = sys.modules[__name__]
        for name in list(dir(current_module)):
            if not name.startswith('_') and name not in ['st', 'np', 'pd', 'librosa', 'go', 'px', 'make_subplots', 'io', 'signal', 'skew', 'kurtosis', 'warnings', 'gc', 'psutil', 'os', 'time', 'wraps', 'base64']:
                if name in ['audio_data', 'features', 'fig', 'waveform_fig', 'spec_fig', 'mel_fig', 'rhythm_fig', 'chroma_fig']:
                    try:
                        delattr(current_module, name)
                    except:
                        pass
        
        return True
    except Exception as e:
        st.error(f"Error during memory cleanup: {str(e)}")
        return False

def create_audio_player(file_bytes, file_name):
    """Create HTML5 audio player for the uploaded file"""
    try:
        # Convert bytes to base64 for embedding
        audio_base64 = base64.b64encode(file_bytes).decode()
        
        # Determine MIME type based on file extension
        file_extension = file_name.lower().split('.')[-1]
        mime_types = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg',
            'm4a': 'audio/mp4',
            'aac': 'audio/aac'
        }
        mime_type = mime_types.get(file_extension, 'audio/mpeg')
        
        # Create HTML5 audio player
        audio_html = f"""
        <div class="audio-player-container">
            <h4>🎵 Now Playing: {file_name}</h4>
            <audio controls style="width: 100%; margin-top: 10px;">
                <source src="data:{mime_type};base64,{audio_base64}" type="{mime_type}">
                Your browser does not support the audio element.
            </audio>
            <p style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                Use the controls above to play, pause, and seek through your audio file
            </p>
        </div>
        """
        
        return audio_html
        
    except Exception as e:
        st.error(f"Error creating audio player: {str(e)}")
        return None

def downsample_for_visualization(data, max_points=8000):
    """Optimized downsampling for 6-minute audio visualization"""
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data

def adaptive_hop_length(audio_length, sr):
    """Optimized adaptive hop length for 6-minute songs"""
    duration = audio_length / sr
    if duration > 300:  # > 5 minutes
        return 4096  # Larger hop length for very long songs
    elif duration > 120:  # > 2 minutes
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
    """Load audio file with error handling and return file bytes for playback"""
    try:
        file_bytes = file.read()
        file.seek(0)  # Reset file pointer
        return load_audio_cached(file_bytes, file.name), file_bytes
    except Exception as e:
        return (None, None, str(e)), None

def optimize_audio_for_processing(audio_data, sr, max_duration=360):
    """Optimize audio data for processing - Extended to 6 minutes"""
    original_duration = len(audio_data) / sr
    if original_duration > max_duration:
        st.warning(f"⚠️ Audio duration ({original_duration:.1f}s) exceeds 6-minute limit ({max_duration}s). Trimming for optimal performance.")
        max_samples = int(max_duration * sr)
        audio_data = audio_data[:max_samples]
    
    # Normalize audio
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    return audio_data

@st.cache_data(show_spinner=False)
def extract_basic_features(audio_data, sr):
    """Extract basic features optimized for 6-minute audio"""
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
    """Extract rhythm features with proper tempo handling"""
    features = {}
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Ensure tempo is a scalar for formatting
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
    """Extract MFCC features optimized for 6-minute audio"""
    features = {}
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Adjust MFCC count based on duration
        duration = len(audio_data) / sr
        if duration > 240:  # > 4 minutes
            n_mfcc = min(n_mfcc, 10)
        elif duration > 120:  # > 2 minutes
            n_mfcc = min(n_mfcc, 12)
        
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
    """Extract chroma features optimized for 6-minute audio"""
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
    """Create memory-optimized waveform plot for 6-minute audio"""
    try:
        # Enhanced downsampling for longer audio
        audio_viz = downsample_for_visualization(audio_data, max_points=12000)
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
        rms_viz = downsample_for_visualization(rms, max_points=3000)
        time_rms = np.linspace(0, len(audio_data) / sr, len(rms_viz))
        
        fig.add_trace(go.Scatter(
            x=time_rms, y=rms_viz,
            mode='lines', name='RMS Energy',
            line=dict(color='#f093fb', width=1.5),
            hovertemplate='Time: %{x:.3f}s<br>RMS: %{y:.4f}<extra></extra>'
        ), row=2, col=1)
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
        centroids_viz = downsample_for_visualization(spectral_centroids, max_points=3000)
        time_centroids = np.linspace(0, len(audio_data) / sr, len(centroids_viz))
        
        fig.add_trace(go.Scatter(
            x=time_centroids, y=centroids_viz,
            mode='lines', name='Spectral Centroid',
            line=dict(color='#f6d365', width=1.5),
            hovertemplate='Time: %{x:.3f}s<br>Centroid: %{y:.1f}Hz<extra></extra>'
        ), row=3, col=1)
        
        fig.update_layout(
            title="Audio Analysis Overview (6-Minute Optimized)",
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
    """Create memory-optimized spectrogram for 6-minute audio"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute spectrogram with optimized parameters
        D = librosa.stft(audio_data, hop_length=hop_length, n_fft=2048)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Enhanced downsampling for longer audio
        if S_db.shape[1] > 1500:
            step = S_db.shape[1] // 1500
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
            title="Spectrogram (6-Minute Optimized)",
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
    """Create rhythm analysis plot optimized for 6-minute audio"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Extract tempo and beats
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Ensure tempo is scalar for formatting
        tempo_scalar = float(tempo.item() if hasattr(tempo, 'item') else tempo)
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        
        # Create waveform for context with enhanced downsampling
        audio_viz = downsample_for_visualization(audio_data, max_points=12000)
        time_viz = np.linspace(0, len(audio_data) / sr, len(audio_viz))
        
        fig = go.Figure()
        
        # Add waveform
        fig.add_trace(go.Scatter(
            x=time_viz, y=audio_viz,
            mode='lines', name='Waveform',
            line=dict(color='lightblue', width=1),
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ))
        
        # Add beat markers (sample for visualization if too many)
        if len(beat_times) > 200:  # Limit beat markers for 6-minute songs
            beat_step = len(beat_times) // 200
            beat_times_viz = beat_times[::beat_step]
        else:
            beat_times_viz = beat_times
            
        if len(beat_times_viz) > 0:
            beat_amplitudes = np.interp(beat_times_viz, time_viz, audio_viz)
            fig.add_trace(go.Scatter(
                x=beat_times_viz, y=beat_amplitudes,
                mode='markers', name='Beats',
                marker=dict(color='red', size=6, symbol='x'),
                hovertemplate=f'Beat at %{{x:.3f}}s<br>Tempo: {tempo_scalar:.1f}BPM<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Rhythm Analysis - Tempo: {tempo_scalar:.1f} BPM (6-Minute Optimized)",
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
    """Create mel spectrogram optimized for 6-minute audio"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute mel spectrogram with optimized parameters
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, hop_length=hop_length, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Enhanced downsampling for longer audio
        if mel_spec_db.shape[1] > 1500:
            step = mel_spec_db.shape[1] // 1500
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
            title="Mel Spectrogram (6-Minute Optimized)",
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
    """Create chroma plot optimized for 6-minute audio"""
    try:
        hop_length = adaptive_hop_length(len(audio_data), sr)
        
        # Compute chroma
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hop_length)
        
        # Enhanced downsampling for longer audio
        if chroma.shape[1] > 1500:
            step = chroma.shape[1] // 1500
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
            title="Chromagram (6-Minute Optimized)",
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
    """Main application function with integrated audio playback"""
    st.markdown('<h1 class="main-header">🎵 Enhanced Audio Spectrum Visualizer v3.0</h1>', unsafe_allow_html=True)
    
    # Optimization info banner
    st.markdown("""
    <div class="optimization-info">
        <h4>🚀 Optimized for 6-Minute Songs with Audio Playback</h4>
        <p>This version includes integrated audio playback functionality along with comprehensive analysis for songs up to 6 minutes in duration.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Enhanced Memory cleanup section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧹 Memory Management")
    
    # Memory cleanup button with enhanced styling
    if st.sidebar.button("🧹 Clear Memory & Cache", 
                        help="Clear all cached data and free memory", 
                        type="primary",
                        use_container_width=True):
        with st.spinner("🔄 Clearing memory and cache..."):
            success = clear_all_memory()
            if success:
                st.sidebar.success("✅ Memory cleared successfully!")
                st.balloons()  # Celebration effect
                time.sleep(2)
                st.rerun()
            else:
                st.sidebar.error("❌ Error clearing memory")
    
    # Memory usage warning (adjusted thresholds for 6-minute songs)
    if memory_usage > 800:
        st.sidebar.error(f"🚨 Very high memory usage: {memory_usage:.1f}MB")
    elif memory_usage > 600:
        st.sidebar.warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
    elif memory_usage > 400:
        st.sidebar.info(f"ℹ️ Moderate memory usage: {memory_usage:.1f}MB")
    
    # File upload
    st.markdown("### 📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file (optimized for songs up to 6 minutes)",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
        help="Upload an audio file for analysis and playback (Limit: 200MB per file, optimized for 6-minute songs)"
    )
    
    if uploaded_file is not None:
        # Load audio and get file bytes for playback
        with st.spinner("🎵 Loading audio file..."):
            (audio_data, sr, error), file_bytes = load_audio(uploaded_file)
        
        if error:
            st.error(f"❌ Error loading audio: {error}")
            return
        
        if audio_data is None:
            st.error("❌ Failed to load audio file")
            return
        
        # Create and display audio player
        if file_bytes:
            audio_player_html = create_audio_player(file_bytes, uploaded_file.name)
            if audio_player_html:
                st.markdown(audio_player_html, unsafe_allow_html=True)
        
        # Optimize audio for 6-minute processing
        audio_data = optimize_audio_for_processing(audio_data, sr, max_duration=360)
        
        # Display basic info with enhanced styling
        st.markdown('<div class="success-message">✅ Audio loaded successfully!</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Duration", f"{len(audio_data) / sr:.2f}s")
        with col2:
            st.metric("🎵 Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("📊 Samples", f"{len(audio_data):,}")
        with col4:
            # Show optimization status
            duration = len(audio_data) / sr
            if duration <= 360:
                st.metric("🚀 Status", "Optimized")
            else:
                st.metric("⚠️ Status", "Trimmed")
        
        # Audio playback controls info
        st.markdown("""
        <div class="audio-controls">
            <h4>🎮 Audio Playback Controls</h4>
            <p>• Use the audio player above to listen to your uploaded file</p>
            <p>• The player supports play, pause, volume control, and seeking</p>
            <p>• You can listen while viewing the analysis results below</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis options
        st.sidebar.subheader("🔧 Analysis Settings")
        
        show_waveform = st.sidebar.checkbox("🌊 Waveform Analysis", value=True)
        show_spectrogram = st.sidebar.checkbox("🎼 Spectrogram", value=True)
        show_mel_spectrogram = st.sidebar.checkbox("🎵 Mel Spectrogram", value=False)
        show_rhythm = st.sidebar.checkbox("🥁 Rhythm Analysis", value=True)
        show_chroma = st.sidebar.checkbox("🎹 Chromagram", value=False)
        show_features = st.sidebar.checkbox("📈 Feature Extraction", value=True)
        
        # Extract features if requested
        if show_features:
            st.header("📈 Feature Analysis")
            with st.spinner("🔍 Extracting comprehensive features..."):
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
            with st.spinner("🎨 Creating waveform plot..."):
                waveform_fig = create_optimized_waveform_plot(audio_data, sr)
                if waveform_fig:
                    st.plotly_chart(waveform_fig, use_container_width=True)
        
        # Spectrogram
        if show_spectrogram:
            st.subheader("🎼 Spectrogram")
            with st.spinner("🎨 Computing spectrogram..."):
                spec_fig = create_optimized_spectrogram(audio_data, sr)
                if spec_fig:
                    st.plotly_chart(spec_fig, use_container_width=True)
        
        # Mel Spectrogram
        if show_mel_spectrogram:
            st.subheader("🎵 Mel Spectrogram")
            with st.spinner("🎨 Computing mel spectrogram..."):
                mel_fig = create_mel_spectrogram(audio_data, sr)
                if mel_fig:
                    st.plotly_chart(mel_fig, use_container_width=True)
        
        # Rhythm analysis
        if show_rhythm:
            st.subheader("🥁 Rhythm Analysis")
            with st.spinner("🎨 Analyzing rhythm..."):
                rhythm_features = extract_rhythm_features(audio_data, sr)
                rhythm_fig = create_rhythm_plot(audio_data, sr, rhythm_features)
                if rhythm_fig:
                    st.plotly_chart(rhythm_fig, use_container_width=True)
        
        # Chroma analysis
        if show_chroma:
            st.subheader("🎹 Chromagram")
            with st.spinner("🎨 Computing chromagram..."):
                chroma_fig = create_chroma_plot(audio_data, sr)
                if chroma_fig:
                    st.plotly_chart(chroma_fig, use_container_width=True)
        
        # Analysis completion message with memory cleanup option
        st.markdown("---")
        st.markdown('<div class="analysis-complete">🎉 Analysis completed successfully!</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("🧹 Free Memory", 
                        help="Clear analysis data and free memory",
                        type="secondary",
                        use_container_width=True):
                with st.spinner("🔄 Freeing memory..."):
                    clear_all_memory()
                    st.success("✅ Memory freed!")
                    time.sleep(1)
                    st.rerun()
        
        with col3:
            current_memory = get_memory_usage()
            st.metric("Current Memory", f"{current_memory:.1f} MB")
        
        # Memory cleanup
        cleanup_memory()
        
    else:
        # Enhanced welcome message
        st.markdown("""
        <div class="feature-card">
            <h3>🎵 Welcome to the Enhanced Audio Spectrum Visualizer v3.0</h3>
            <p>This version includes <strong>integrated audio playback functionality</strong> along with comprehensive analysis for songs up to 6 minutes. Upload an audio file to:</p>
            <ul>
                <li>🎵 <strong>Play your audio file</strong> with built-in HTML5 audio player</li>
                <li>🌊 <strong>Analyze waveforms</strong> with adaptive downsampling for longer songs</li>
                <li>🎼 <strong>View spectrograms</strong> with intelligent hop length selection</li>
                <li>🥁 <strong>Detect rhythm and tempo</strong> with beat marker optimization</li>
                <li>🎹 <strong>Analyze harmonic content</strong> with memory-conscious processing</li>
                <li>📊 <strong>Extract comprehensive features</strong> that scale with song duration</li>
                <li>💾 <strong>Manage memory efficiently</strong> for 6-minute audio processing</li>
            </ul>
            <p><strong>🎮 Integrated Audio Playback Features:</strong></p>
            <ul>
                <li>🎵 <strong>HTML5 Audio Player:</strong> Play, pause, and seek through your audio</li>
                <li>🔊 <strong>Volume Control:</strong> Adjust playback volume</li>
                <li>⏯️ <strong>Playback Controls:</strong> Full media controls with progress bar</li>
                <li>🎧 <strong>Listen While Analyzing:</strong> Audio playback during analysis</li>
                <li>📱 <strong>Cross-Platform:</strong> Works on desktop and mobile browsers</li>
            </ul>
            <p><strong>🚀 6-Minute Optimizations:</strong></p>
            <ul>
                <li>📈 <strong>Adaptive hop lengths:</strong> 4096 for 5+ minutes, 2048 for 2-5 minutes</li>
                <li>🎯 <strong>Smart downsampling:</strong> 12,000 points for waveforms, 3,000 for features</li>
                <li>🧠 <strong>Intelligent MFCC scaling:</strong> Reduces coefficients for longer audio</li>
                <li>🔄 <strong>Enhanced garbage collection:</strong> Optimized for longer processing times</li>
                <li>⚡ <strong>Beat marker optimization:</strong> Limits visualization points for long songs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional info cards
        col1, col2 = st.columns(2)
        with col1:
            st.info("📁 **Supported formats:** WAV, MP3, FLAC, OGG, M4A, AAC (Max: 200MB)")
        with col2:
            st.success("🎵 **New Feature:** Integrated audio playback with comprehensive 6-minute analysis")

if __name__ == "__main__":
    main()
