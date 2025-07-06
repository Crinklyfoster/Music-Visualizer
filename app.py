# 🎵 Enhanced Audio Spectrum Visualizer

# Advanced real-time audio analysis with comprehensive time and frequency domain features

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings

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
    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
    .custom-tab {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
        margin-right: 0.5rem;
    }
    .plot-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_audio(file):
    """Load audio file and return audio data and sample rate"""
    try:
        audio_data, sr = librosa.load(file, sr=None)
        return audio_data, sr
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None

def create_enhanced_waveform_plot(audio_data, sr):
    """Create an enhanced interactive waveform plot with multiple representations"""
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    # Create subplots for multiple views
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Raw Waveform', 'Envelope & RMS', 'Spectral Centroid Over Time', 'Zero Crossing Rate'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Raw waveform
    fig.add_trace(go.Scatter(
        x=time, y=audio_data,
        mode='lines', name='Waveform',
        line=dict(color='#667eea', width=1),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>'
    ), row=1, col=1)
    
    # Envelope and RMS
    hop_length = 512
    envelope = np.abs(librosa.stft(audio_data, hop_length=hop_length))
    envelope = np.mean(envelope, axis=0)
    rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
    
    time_frames = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=hop_length)
    
    fig.add_trace(go.Scatter(
        x=time_frames, y=envelope,
        mode='lines', name='Envelope',
        line=dict(color='#f093fb', width=2),
        hovertemplate='Time: %{x:.3f}s<br>Envelope: %{y:.4f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_frames, y=rms,
        mode='lines', name='RMS',
        line=dict(color='#f5576c', width=2),
        hovertemplate='Time: %{x:.3f}s<br>RMS: %{y:.4f}<extra></extra>'
    ), row=2, col=1)
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
    
    fig.add_trace(go.Scatter(
        x=time_frames, y=spectral_centroid,
        mode='lines', name='Spectral Centroid',
        line=dict(color='#4facfe', width=2),
        hovertemplate='Time: %{x:.3f}s<br>Centroid: %{y:.1f}Hz<extra></extra>'
    ), row=3, col=1)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=hop_length)[0]
    
    fig.add_trace(go.Scatter(
        x=time_frames, y=zcr,
        mode='lines', name='Zero Crossing Rate',
        line=dict(color='#ff9a9e', width=2),
        hovertemplate='Time: %{x:.3f}s<br>ZCR: %{y:.4f}<extra></extra>'
    ), row=4, col=1)
    
    fig.update_layout(
        height=1000,
        title_text="Enhanced Waveform Analysis",
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="Rate", row=4, col=1)
    
    return fig

def create_enhanced_spectrum_plot(audio_data, sr):
    """Create enhanced spectrum visualization with multiple representations"""
    # Compute various spectral representations
    stft = librosa.stft(audio_data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Chromagram
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Magnitude Spectrogram', 'Mel Spectrogram', 'Chromagram', 'Tonnetz'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Time and frequency axes
    time_frames = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    
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
    fig.add_trace(go.Heatmap(
        z=mel_spec_db,
        x=time_frames,
        y=np.arange(mel_spec_db.shape[0]),
        colorscale='Plasma',
        name='Mel',
        hovertemplate='Time: %{x:.3f}s<br>Mel Band: %{y}<br>Power: %{z:.1f}dB<extra></extra>'
    ), row=1, col=2)
    
    # Chromagram
    fig.add_trace(go.Heatmap(
        z=chroma,
        x=time_frames,
        y=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        colorscale='Blues',
        name='Chroma',
        hovertemplate='Time: %{x:.3f}s<br>Note: %{y}<br>Strength: %{z:.3f}<extra></extra>'
    ), row=2, col=1)
    
    # Tonnetz
    fig.add_trace(go.Heatmap(
        z=tonnetz,
        x=time_frames,
        y=['Tonnetz_1', 'Tonnetz_2', 'Tonnetz_3', 'Tonnetz_4', 'Tonnetz_5', 'Tonnetz_6'],
        colorscale='RdBu',
        name='Tonnetz',
        hovertemplate='Time: %{x:.3f}s<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
    ), row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text="Enhanced Spectral Analysis",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_frequency_domain_plot(audio_data, sr):
    """Create frequency domain analysis plots"""
    # Compute FFT
    fft = np.fft.fft(audio_data)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    freqs = np.fft.fftfreq(len(audio_data), 1/sr)
    
    # Only take positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    positive_phase = phase[:len(phase)//2]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('FFT Magnitude', 'FFT Phase', 'Power Spectral Density', 'Spectral Features'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # FFT Magnitude
    fig.add_trace(go.Scatter(
        x=positive_freqs, y=20*np.log10(positive_magnitude + 1e-10),
        mode='lines', name='FFT Magnitude',
        line=dict(color='#667eea', width=1),
        hovertemplate='Frequency: %{x:.1f}Hz<br>Magnitude: %{y:.1f}dB<extra></extra>'
    ), row=1, col=1)
    
    # FFT Phase
    fig.add_trace(go.Scatter(
        x=positive_freqs, y=positive_phase,
        mode='lines', name='FFT Phase',
        line=dict(color='#f093fb', width=1),
        hovertemplate='Frequency: %{x:.1f}Hz<br>Phase: %{y:.3f}rad<extra></extra>'
    ), row=1, col=2)
    
    # Power Spectral Density
    freqs_psd, psd = signal.welch(audio_data, sr, nperseg=1024)
    fig.add_trace(go.Scatter(
        x=freqs_psd, y=10*np.log10(psd),
        mode='lines', name='PSD',
        line=dict(color='#4facfe', width=2),
        hovertemplate='Frequency: %{x:.1f}Hz<br>PSD: %{y:.1f}dB/Hz<extra></extra>'
    ), row=2, col=1)
    
    # Spectral Features over frequency bands
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    
    time_frames = librosa.frames_to_time(np.arange(len(spectral_rolloff)), sr=sr)
    
    fig.add_trace(go.Scatter(
        x=time_frames, y=spectral_rolloff,
        mode='lines', name='Spectral Rolloff',
        line=dict(color='#f5576c', width=2),
        hovertemplate='Time: %{x:.3f}s<br>Rolloff: %{y:.1f}Hz<extra></extra>'
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=time_frames, y=spectral_centroid,
        mode='lines', name='Spectral Centroid',
        line=dict(color='#ff9a9e', width=2),
        hovertemplate='Time: %{x:.3f}s<br>Centroid: %{y:.1f}Hz<extra></extra>'
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=time_frames, y=spectral_bandwidth,
        mode='lines', name='Spectral Bandwidth',
        line=dict(color='#a8e6cf', width=2),
        hovertemplate='Time: %{x:.3f}s<br>Bandwidth: %{y:.1f}Hz<extra></extra>'
    ), row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text="Frequency Domain Analysis",
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=1, col=2)
    fig.update_yaxes(title_text="PSD (dB/Hz)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)
    
    return fig

def extract_comprehensive_features(audio_data, sr):
    """Extract comprehensive audio features"""
    features = {}
    
    # Basic features
    features['Duration'] = len(audio_data) / sr
    features['Sample_Rate'] = sr
    features['Channels'] = 1 if audio_data.ndim == 1 else audio_data.shape[0]
    features['Total_Samples'] = len(audio_data)
    
    # Temporal features
    features['Zero_Crossing_Rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
    features['Zero_Crossing_Rate_Std'] = np.std(librosa.feature.zero_crossing_rate(audio_data)[0])
    
    # Energy features
    features['RMS_Energy'] = np.mean(librosa.feature.rms(y=audio_data)[0])
    features['RMS_Energy_Std'] = np.std(librosa.feature.rms(y=audio_data)[0])
    features['Total_Energy'] = np.sum(audio_data**2)
    features['Average_Power'] = np.mean(audio_data**2)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    features['Spectral_Centroid_Mean'] = np.mean(spectral_centroids)
    features['Spectral_Centroid_Std'] = np.std(spectral_centroids)
    features['Spectral_Centroid_Min'] = np.min(spectral_centroids)
    features['Spectral_Centroid_Max'] = np.max(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    features['Spectral_Rolloff_Mean'] = np.mean(spectral_rolloff)
    features['Spectral_Rolloff_Std'] = np.std(spectral_rolloff)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    features['Spectral_Bandwidth_Mean'] = np.mean(spectral_bandwidth)
    features['Spectral_Bandwidth_Std'] = np.std(spectral_bandwidth)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    features['Spectral_Contrast_Mean'] = np.mean(spectral_contrast)
    features['Spectral_Contrast_Std'] = np.std(spectral_contrast)
    
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
    features['Spectral_Flatness_Mean'] = np.mean(spectral_flatness)
    features['Spectral_Flatness_Std'] = np.std(spectral_flatness)
    
    # Rhythm features
    try:
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        features['Tempo'] = float(tempo)
        features['Beat_Count'] = len(beats)
        features['Beat_Density'] = len(beats) / (len(audio_data) / sr)
    except:
        features['Tempo'] = 0.0
        features['Beat_Count'] = 0
        features['Beat_Density'] = 0.0
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'MFCC_{i+1}_Mean'] = np.mean(mfccs[i])
        features[f'MFCC_{i+1}_Std'] = np.std(mfccs[i])
        features[f'MFCC_{i+1}_Min'] = np.min(mfccs[i])
        features[f'MFCC_{i+1}_Max'] = np.max(mfccs[i])
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['Chroma_Mean'] = np.mean(chroma)
    features['Chroma_Std'] = np.std(chroma)
    features['Chroma_Min'] = np.min(chroma)
    features['Chroma_Max'] = np.max(chroma)
    
    # Individual chroma features
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, label in enumerate(chroma_labels):
        features[f'Chroma_{label}_Mean'] = np.mean(chroma[i])
        features[f'Chroma_{label}_Std'] = np.std(chroma[i])
    
    # Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
    features['Tonnetz_Mean'] = np.mean(tonnetz)
    features['Tonnetz_Std'] = np.std(tonnetz)
    
    for i in range(tonnetz.shape[0]):
        features[f'Tonnetz_{i+1}_Mean'] = np.mean(tonnetz[i])
        features[f'Tonnetz_{i+1}_Std'] = np.std(tonnetz[i])
    
    # Statistical features
    features['Audio_Mean'] = np.mean(audio_data)
    features['Audio_Std'] = np.std(audio_data)
    features['Audio_Skewness'] = skew(audio_data)
    features['Audio_Kurtosis'] = kurtosis(audio_data)
    features['Audio_Min'] = np.min(audio_data)
    features['Audio_Max'] = np.max(audio_data)
    features['Audio_Range'] = np.max(audio_data) - np.min(audio_data)
    
    # Frequency domain features
    fft = np.fft.fft(audio_data)
    magnitude = np.abs(fft)
    features['FFT_Mean'] = np.mean(magnitude)
    features['FFT_Std'] = np.std(magnitude)
    features['FFT_Max'] = np.max(magnitude)
    
    # Spectral entropy
    psd = magnitude**2
    psd_norm = psd / np.sum(psd)
    features['Spectral_Entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    # Harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(audio_data)
    features['Harmonic_Mean'] = np.mean(harmonic)
    features['Harmonic_Std'] = np.std(harmonic)
    features['Percussive_Mean'] = np.mean(percussive)
    features['Percussive_Std'] = np.std(percussive)
    features['Harmonic_Percussive_Ratio'] = np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-10)
    
    return features

def create_feature_comparison_plot(features_dict):
    """Create interactive feature comparison plots"""
    # Prepare data for different feature categories
    temporal_features = {k: v for k, v in features_dict.items() if any(x in k.lower() for x in ['zero', 'rms', 'duration', 'energy'])}
    spectral_features = {k: v for k, v in features_dict.items() if 'spectral' in k.lower()}
    rhythm_features = {k: v for k, v in features_dict.items() if any(x in k.lower() for x in ['tempo', 'beat'])}
    mfcc_features = {k: v for k, v in features_dict.items() if 'mfcc' in k.lower() and 'mean' in k.lower()}
    
    # Limit MFCC features for visibility
    mfcc_features = dict(list(mfcc_features.items())[:8])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temporal Features', 'Spectral Features', 'Rhythm Features', 'MFCC Features'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Temporal features
    if temporal_features:
        fig.add_trace(go.Bar(
            x=list(temporal_features.keys()),
            y=list(temporal_features.values()),
            name='Temporal',
            marker_color='#667eea',
            hovertemplate='Feature: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ), row=1, col=1)
    
    # Spectral features
    if spectral_features:
        fig.add_trace(go.Bar(
            x=list(spectral_features.keys()),
            y=list(spectral_features.values()),
            name='Spectral',
            marker_color='#f093fb',
            hovertemplate='Feature: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ), row=1, col=2)
    
    # Rhythm features
    if rhythm_features:
        fig.add_trace(go.Bar(
            x=list(rhythm_features.keys()),
            y=list(rhythm_features.values()),
            name='Rhythm',
            marker_color='#4facfe',
            hovertemplate='Feature: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ), row=2, col=1)
    
    # MFCC features
    if mfcc_features:
        fig.add_trace(go.Bar(
            x=list(mfcc_features.keys()),
            y=list(mfcc_features.values()),
            name='MFCC',
            marker_color='#f5576c',
            hovertemplate='Feature: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ), row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="Audio Feature Analysis",
        showlegend=False,
        template="plotly_white"
    )
    
    # Update x-axis labels to be rotated
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_3d_feature_plot(features_dict):
    """Create 3D visualization of audio features"""
    # Select key features for 3D plot
    x_feature = features_dict.get('Spectral_Centroid_Mean', 0)
    y_feature = features_dict.get('RMS_Energy', 0)
    z_feature = features_dict.get('Tempo', 0)
    
    # Create additional points for context (using feature variations)
    x_points = [x_feature, x_feature * 0.9, x_feature * 1.1]
    y_points = [y_feature, y_feature * 0.9, y_feature * 1.1]
    z_points = [z_feature, z_feature * 0.9, z_feature * 1.1]
    
    colors = ['#667eea', '#f093fb', '#4facfe']
    sizes = [20, 15, 15]
    labels = ['Main Sample', 'Variation 1', 'Variation 2']
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8
        ),
        text=labels,
        hovertemplate='Spectral Centroid: %{x:.1f}Hz<br>RMS Energy: %{y:.4f}<br>Tempo: %{z:.1f}BPM<br>%{text}<extra></extra>'
    )])
    
    fig.update_layout(
        title='3D Audio Feature Space',
        scene=dict(
            xaxis_title='Spectral Centroid (Hz)',
            yaxis_title='RMS Energy',
            zaxis_title='Tempo (BPM)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600
    )
    
    return fig

def create_chroma_analysis_plot(audio_data, sr):
    """Create detailed chroma analysis visualization"""
    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=audio_data, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=audio_data, sr=sr)
    
    time_frames = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Chroma STFT', 'Chroma CQT', 'Chroma CENS', 'Chroma Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "bar"}]]
    )
    
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Chroma STFT
    fig.add_trace(go.Heatmap(
        z=chroma,
        x=time_frames,
        y=chroma_labels,
        colorscale='Blues',
        name='Chroma STFT',
        hovertemplate='Time: %{x:.3f}s<br>Note: %{y}<br>Strength: %{z:.3f}<extra></extra>'
    ), row=1, col=1)
    
    # Chroma CQT
    fig.add_trace(go.Heatmap(
        z=chroma_cqt,
        x=time_frames,
        y=chroma_labels,
        colorscale='Reds',
        name='Chroma CQT',
        hovertemplate='Time: %{x:.3f}s<br>Note: %{y}<br>Strength: %{z:.3f}<extra></extra>'
    ), row=1, col=2)
    
    # Chroma CENS
    fig.add_trace(go.Heatmap(
        z=chroma_cens,
        x=time_frames,
        y=chroma_labels,
        colorscale='Greens',
        name='Chroma CENS',
        hovertemplate='Time: %{x:.3f}s<br>Note: %{y}<br>Strength: %{z:.3f}<extra></extra>'
    ), row=2, col=1)
    
    # Chroma summary (average strength per note)
    chroma_mean = np.mean(chroma, axis=1)
    fig.add_trace(go.Bar(
        x=chroma_labels,
        y=chroma_mean,
        name='Average Chroma',
        marker_color='#667eea',
        hovertemplate='Note: %{x}<br>Average Strength: %{y:.3f}<extra></extra>'
    ), row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text="Detailed Chroma Analysis",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_rhythm_analysis_plot(audio_data, sr):
    """Create rhythm and tempo analysis visualization"""
    try:
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Tempogram
        tempogram = librosa.feature.tempogram(y=audio_data, sr=sr)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Beats and Onsets', 'Tempogram', 'Beat Histogram', 'Rhythm Features'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Beats and onsets over waveform
        time = np.linspace(0, len(audio_data) / sr, len(audio_data))
        fig.add_trace(go.Scatter(
            x=time, y=audio_data,
            mode='lines', name='Waveform',
            line=dict(color='lightgray', width=1),
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>'
        ), row=1, col=1)
        
        # Add beat markers
        for beat_time in beat_times:
            fig.add_vline(x=beat_time, line_dash="dash", line_color="red", 
                         annotation_text="Beat", row=1, col=1)
        
        # Add onset markers
        for onset_time in onset_times:
            fig.add_vline(x=onset_time, line_dash="dot", line_color="blue", 
                         annotation_text="Onset", row=1, col=1)
        
        # Tempogram
        time_frames = librosa.frames_to_time(np.arange(tempogram.shape[1]), sr=sr)
        tempo_axis = librosa.tempo_frequencies(tempogram.shape[0])
        
        fig.add_trace(go.Heatmap(
            z=tempogram,
            x=time_frames,
            y=tempo_axis,
            colorscale='Viridis',
            name='Tempogram',
            hovertemplate='Time: %{x:.3f}s<br>Tempo: %{y:.1f}BPM<br>Strength: %{z:.3f}<extra></extra>'
        ), row=1, col=2)
        
        # Beat histogram
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            fig.add_trace(go.Histogram(
                x=beat_intervals,
                nbinsx=20,
                name='Beat Intervals',
                marker_color='#4facfe',
                hovertemplate='Interval: %{x:.3f}s<br>Count: %{y}<extra></extra>'
            ), row=2, col=1)
        
        # Rhythm features
        rhythm_features = {
            'Tempo': tempo,
            'Beat Count': len(beats),
            'Onset Count': len(onset_frames),
            'Avg Beat Interval': np.mean(np.diff(beat_times)) if len(beat_times) > 1 else 0
        }
        
        fig.add_trace(go.Bar(
            x=list(rhythm_features.keys()),
            y=list(rhythm_features.values()),
            name='Rhythm Features',
            marker_color='#f5576c',
            hovertemplate='Feature: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ), row=2, col=2)
        
    except Exception as e:
        # Fallback if rhythm analysis fails
        fig = go.Figure()
        fig.add_annotation(
            text=f"Rhythm analysis failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig.update_layout(
        height=800,
        title_text="Rhythm and Tempo Analysis",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_advanced_statistics_plot(features_dict):
    """Create advanced statistical analysis of features"""
    # Separate features by type
    feature_categories = {
        'Temporal': [k for k in features_dict.keys() if any(x in k.lower() for x in ['zero', 'rms', 'energy'])],
        'Spectral': [k for k in features_dict.keys() if 'spectral' in k.lower()],
        'MFCC': [k for k in features_dict.keys() if 'mfcc' in k.lower()],
        'Chroma': [k for k in features_dict.keys() if 'chroma' in k.lower()],
        'Rhythm': [k for k in features_dict.keys() if any(x in k.lower() for x in ['tempo', 'beat'])]
    }
    
    # Create correlation matrix for numerical features
    numerical_features = {k: v for k, v in features_dict.items() if isinstance(v, (int, float))}
    
    if len(numerical_features) > 1:
        # Create feature correlation plot
        feature_names = list(numerical_features.keys())[:20]  # Limit for visibility
        feature_values = [numerical_features[name] for name in feature_names]
        
        # Create correlation matrix
        correlation_matrix = np.corrcoef([feature_values])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distribution', 'Feature Categories', 'Top Features', 'Statistical Summary'),
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Feature distribution
        fig.add_trace(go.Histogram(
            x=feature_values,
            nbinsx=20,
            name='Feature Distribution',
            marker_color='#667eea',
            hovertemplate='Value: %{x:.4f}<br>Count: %{y}<extra></extra>'
        ), row=1, col=1)
        
        # Feature categories pie chart
        category_counts = [len(features) for features in feature_categories.values()]
        fig.add_trace(go.Pie(
            labels=list(feature_categories.keys()),
            values=category_counts,
            name='Feature Categories',
            hovertemplate='Category: %{label}<br>Count: %{value}<extra></extra>'
        ), row=1, col=2)
        
        # Top features by absolute value
        top_features = sorted(numerical_features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        fig.add_trace(go.Bar(
            x=[f[0] for f in top_features],
            y=[f[1] for f in top_features],
            name='Top Features',
            marker_color='#f093fb',
            hovertemplate='Feature: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ), row=2, col=1)
        
        # Statistical summary
        stats = {
            'Mean': np.mean(feature_values),
            'Std': np.std(feature_values),
            'Min': np.min(feature_values),
            'Max': np.max(feature_values),
            'Median': np.median(feature_values)
        }
        
        fig.add_trace(go.Bar(
            x=list(stats.keys()),
            y=list(stats.values()),
            name='Statistics',
            marker_color='#4facfe',
            hovertemplate='Statistic: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Advanced Statistical Analysis",
            showlegend=False,
            template="plotly_white"
        )
        
        fig.update_xaxes(tickangle=45, row=2, col=1)
        
        return fig
    
    else:
        # Return empty figure if not enough features
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough numerical features for statistical analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def main():
    # Main title
    st.markdown('<h1 class="main-header">🎵 Enhanced Audio Spectrum Visualizer</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Welcome to the Enhanced Audio Spectrum Visualizer</h3>
        <p>This advanced tool provides comprehensive audio analysis including waveform visualization, 
        spectral analysis, feature extraction, and rhythm detection. Upload an audio file to get started!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    st.sidebar.markdown("Upload an audio file to begin analysis")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC"
    )
    
    if uploaded_file is not None:
        # Load audio
        with st.spinner("🔄 Loading audio file..."):
            audio_data, sr = load_audio(uploaded_file)
        
        if audio_data is not None:
            # Display success message
            st.markdown("""
            <div class="success-box">
                <h3>✅ Audio loaded successfully!</h3>
                <p>Your audio file has been processed and is ready for analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Basic audio information
            st.header("📊 Audio Information")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Duration</h3>
                    <h2>{len(audio_data)/sr:.2f}s</h2>
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
                st.markdown(f'''
                <div class="metric-card">
                    <h3>Channels</h3>
                    <h2>{1 if audio_data.ndim == 1 else audio_data.shape[0]}</h2>
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
            with st.spinner("🔍 Extracting comprehensive audio features..."):
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
            show_waveform = st.sidebar.checkbox("🌊 Waveform Analysis", value=True)
            show_spectrum = st.sidebar.checkbox("🌈 Spectral Analysis", value=True)
            show_frequency = st.sidebar.checkbox("📈 Frequency Domain", value=False)
            show_chroma = st.sidebar.checkbox("🎼 Chroma Analysis", value=False)
            show_rhythm = st.sidebar.checkbox("🥁 Rhythm Analysis", value=False)
            show_features = st.sidebar.checkbox("📊 Feature Analysis", value=True)
            show_3d = st.sidebar.checkbox("🎨 3D Visualization", value=False)
            show_stats = st.sidebar.checkbox("📈 Advanced Statistics", value=False)
            
            # Waveform analysis
            if show_waveform:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🌊 Waveform Analysis")
                st.markdown("Comprehensive time-domain analysis of your audio signal")
                
                with st.spinner("Creating enhanced waveform plots..."):
                    waveform_fig = create_enhanced_waveform_plot(audio_data, sr)
                    st.plotly_chart(waveform_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Spectral analysis
            if show_spectrum:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🌈 Spectral Analysis")
                st.markdown("Multi-dimensional frequency analysis including spectrograms and chromagrams")
                
                with st.spinner("Creating spectral analysis plots..."):
                    spectrum_fig = create_enhanced_spectrum_plot(audio_data, sr)
                    st.plotly_chart(spectrum_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Frequency domain analysis
            if show_frequency:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("📈 Frequency Domain Analysis")
                st.markdown("Detailed frequency domain characteristics and spectral features")
                
                with st.spinner("Creating frequency domain plots..."):
                    frequency_fig = create_frequency_domain_plot(audio_data, sr)
                    st.plotly_chart(frequency_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Chroma analysis
            if show_chroma:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🎼 Chroma Analysis")
                st.markdown("Harmonic content analysis and pitch class profiling")
                
                with st.spinner("Creating chroma analysis plots..."):
                    chroma_fig = create_chroma_analysis_plot(audio_data, sr)
                    st.plotly_chart(chroma_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Rhythm analysis
            if show_rhythm:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🥁 Rhythm Analysis")
                st.markdown("Beat tracking, tempo estimation, and rhythmic pattern analysis")
                
                with st.spinner("Creating rhythm analysis plots..."):
                    rhythm_fig = create_rhythm_analysis_plot(audio_data, sr)
                    st.plotly_chart(rhythm_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature analysis
            if show_features:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("📊 Feature Analysis")
                st.markdown("Comprehensive feature extraction and analysis")
                
                # Feature summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Key Features")
                    key_features = {
                        'Spectral Centroid': f"{features.get('Spectral_Centroid_Mean', 0):.1f} Hz",
                        'RMS Energy': f"{features.get('RMS_Energy', 0):.4f}",
                        'Zero Crossing Rate': f"{features.get('Zero_Crossing_Rate', 0):.4f}",
                        'Tempo': f"{features.get('Tempo', 0):.1f} BPM",
                        'Spectral Rolloff': f"{features.get('Spectral_Rolloff_Mean', 0):.1f} Hz",
                        'Spectral Bandwidth': f"{features.get('Spectral_Bandwidth_Mean', 0):.1f} Hz"
                    }
                    
                    for feature, value in key_features.items():
                        st.markdown(f'<div class="feature-box"><strong>{feature}:</strong> {value}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.subheader("📊 Statistical Summary")
                    stats = {
                        'Mean Amplitude': f"{features.get('Audio_Mean', 0):.4f}",
                        'Std Deviation': f"{features.get('Audio_Std', 0):.4f}",
                        'Skewness': f"{features.get('Audio_Skewness', 0):.4f}",
                        'Kurtosis': f"{features.get('Audio_Kurtosis', 0):.4f}",
                        'Dynamic Range': f"{features.get('Audio_Range', 0):.4f}",
                        'Total Energy': f"{features.get('Total_Energy', 0):.2e}"
                    }
                    
                    for stat, value in stats.items():
                        st.markdown(f'<div class="feature-box"><strong>{stat}:</strong> {value}</div>', unsafe_allow_html=True)
                
                # Feature comparison plot
                with st.spinner("Creating feature comparison plots..."):
                    feature_fig = create_feature_comparison_plot(features)
                    st.plotly_chart(feature_fig, use_container_width=True)
                
                # Detailed feature table
                st.subheader("📋 Complete Feature Set")
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                feature_df['Value'] = feature_df['Value'].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
                st.dataframe(feature_df, use_container_width=True, height=400)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 3D visualization
            if show_3d:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("🎨 3D Feature Visualization")
                st.markdown("Interactive 3D representation of key audio features")
                
                with st.spinner("Creating 3D feature visualization..."):
                    fig_3d = create_3d_feature_plot(features)
                    st.plotly_chart(fig_3d, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced statistics
            if show_stats:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.header("📈 Advanced Statistical Analysis")
                st.markdown("Comprehensive statistical analysis of extracted features")
                
                with st.spinner("Creating advanced statistical plots..."):
                    stats_fig = create_advanced_statistics_plot(features)
                    st.plotly_chart(stats_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Export options
            st.sidebar.subheader("💾 Export Options")
            
            # Download features as CSV
            if st.sidebar.button("📊 Download Features CSV"):
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                csv = feature_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="💾 Download CSV File",
                    data=csv,
                    file_name=f"audio_features_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
            
            # Download features as JSON
            if st.sidebar.button("📋 Download Features JSON"):
                import json
                json_data = json.dumps(features, indent=2)
                st.sidebar.download_button(
                    label="💾 Download JSON File",
                    data=json_data,
                    file_name=f"audio_features_{uploaded_file.name.split('.')[0]}.json",
                    mime="application/json"
                )
            
            # Analysis summary
            st.sidebar.subheader("📈 Analysis Summary")
            st.sidebar.markdown(f"""
            <div class="sidebar-info">
                <strong>Total Features Extracted:</strong> {len(features)}<br>
                <strong>Analysis Duration:</strong> {len(audio_data)/sr:.2f}s<br>
                <strong>Sample Rate:</strong> {sr:,} Hz<br>
                <strong>File Format:</strong> {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="info-box">
            <h3>👆 Please upload an audio file to begin analysis</h3>
            <p>Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example features
        st.header("🎵 What You'll Get")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🌊 Waveform Analysis
            - Raw audio waveform visualization
            - Envelope detection and tracking
            - RMS energy analysis over time
            - Spectral centroid evolution
            - Zero crossing rate analysis
            """)
        
        with col2:
            st.markdown("""
            ### 🌈 Spectral Analysis
            - High-resolution spectrograms
            - Mel-frequency analysis
            - Chromagram visualization
            - Tonnetz harmonic analysis
            - Phase spectrum analysis
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Feature Extraction
            - 100+ comprehensive audio features
            - Temporal and spectral characteristics
            - MFCC coefficients (13 dimensions)
            - Chroma and harmonic features
            - Rhythm and tempo analysis
            """)
        
        # Additional information
        st.header("🔧 Advanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎼 Music Analysis
            - **Chroma Analysis**: Pitch class profiling
            - **Rhythm Detection**: Beat tracking and tempo
            - **Harmonic Analysis**: Tonnetz features
            - **Key Detection**: Musical key estimation
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Statistical Analysis
            - **Feature Correlation**: Inter-feature relationships
            - **Distribution Analysis**: Statistical properties
            - **3D Visualization**: Multi-dimensional feature space
            - **Export Options**: CSV and JSON formats
            """)
        
        # Technical specifications
        st.header("⚙️ Technical Specifications")
        
        st.markdown("""
        <div class="feature-box">
            <h4>🎯 Supported Analysis Types</h4>
            <ul>
                <li><strong>Time Domain:</strong> Waveform, envelope, RMS, zero-crossing rate</li>
                <li><strong>Frequency Domain:</strong> FFT, PSD, spectral features</li>
                <li><strong>Time-Frequency:</strong> STFT, mel-spectrogram, chromagram</li>
                <li><strong>Perceptual:</strong> MFCC, chroma, tonnetz features</li>
                <li><strong>Rhythmic:</strong> Tempo, beat tracking, onset detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
