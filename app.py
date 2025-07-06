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
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
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
        rows=3, cols=1,
        subplot_titles=('Raw Waveform', 'Envelope & RMS', 'Spectral Centroid Over Time'),
        vertical_spacing=0.1,
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
    
    fig.update_layout(
        height=800,
        title_text="Enhanced Waveform Analysis",
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
    
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
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Magnitude Spectrogram', 'Mel Spectrogram', 'Chromagram', 'Phase Spectrogram'),
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
    
    # Phase spectrogram
    fig.add_trace(go.Heatmap(
        z=phase,
        x=time_frames,
        y=freqs[:phase.shape[0]],
        colorscale='RdBu',
        name='Phase',
        hovertemplate='Time: %{x:.3f}s<br>Frequency: %{y:.0f}Hz<br>Phase: %{z:.3f}<extra></extra>'
    ), row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text="Enhanced Spectral Analysis",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def extract_comprehensive_features(audio_data, sr):
    """Extract comprehensive audio features"""
    features = {}
    
    # Basic features
    features['Duration'] = len(audio_data) / sr
    features['Sample_Rate'] = sr
    features['Channels'] = 1 if audio_data.ndim == 1 else audio_data.shape[0]
    
    # Temporal features
    features['Zero_Crossing_Rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
    features['RMS_Energy'] = np.mean(librosa.feature.rms(y=audio_data)[0])
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    features['Spectral_Centroid_Mean'] = np.mean(spectral_centroids)
    features['Spectral_Centroid_Std'] = np.std(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    features['Spectral_Rolloff_Mean'] = np.mean(spectral_rolloff)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    features['Spectral_Bandwidth_Mean'] = np.mean(spectral_bandwidth)
    
    # Rhythm features
    try:
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        features['Tempo'] = tempo
        features['Beat_Count'] = len(beats)
    except:
        features['Tempo'] = 0
        features['Beat_Count'] = 0
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'MFCC_{i+1}_Mean'] = np.mean(mfccs[i])
        features[f'MFCC_{i+1}_Std'] = np.std(mfccs[i])
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['Chroma_Mean'] = np.mean(chroma)
    features['Chroma_Std'] = np.std(chroma)
    
    # Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
    features['Tonnetz_Mean'] = np.mean(tonnetz)
    features['Tonnetz_Std'] = np.std(tonnetz)
    
    # Statistical features
    features['Audio_Mean'] = np.mean(audio_data)
    features['Audio_Std'] = np.std(audio_data)
    features['Audio_Skewness'] = skew(audio_data)
    features['Audio_Kurtosis'] = kurtosis(audio_data)
    
    return features

def create_feature_comparison_plot(features_dict):
    """Create interactive feature comparison plots"""
    # Prepare data for different feature categories
    temporal_features = {k: v for k, v in features_dict.items() if k in ['Zero_Crossing_Rate', 'RMS_Energy', 'Duration']}
    spectral_features = {k: v for k, v in features_dict.items() if 'Spectral' in k}
    rhythm_features = {k: v for k, v in features_dict.items() if k in ['Tempo', 'Beat_Count']}
    
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
            marker_color='#667eea'
        ), row=1, col=1)
    
    # Spectral features
    if spectral_features:
        fig.add_trace(go.Bar(
            x=list(spectral_features.keys()),
            y=list(spectral_features.values()),
            name='Spectral',
            marker_color='#f093fb'
        ), row=1, col=2)
    
    # Rhythm features
    if rhythm_features:
        fig.add_trace(go.Bar(
            x=list(rhythm_features.keys()),
            y=list(rhythm_features.values()),
            name='Rhythm',
            marker_color='#4facfe'
        ), row=2, col=1)
    
    # MFCC features (first 5 for visibility)
    mfcc_features = {k: v for k, v in features_dict.items() if 'MFCC' in k and 'Mean' in k}
    mfcc_features = dict(list(mfcc_features.items())[:5])
    
    if mfcc_features:
        fig.add_trace(go.Bar(
            x=list(mfcc_features.keys()),
            y=list(mfcc_features.values()),
            name='MFCC',
            marker_color='#f5576c'
        ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        title_text="Audio Feature Analysis",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_3d_feature_plot(features_dict):
    """Create 3D visualization of audio features"""
    # Select key features for 3D plot
    x_feature = features_dict.get('Spectral_Centroid_Mean', 0)
    y_feature = features_dict.get('RMS_Energy', 0)
    z_feature = features_dict.get('Tempo', 0)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=[x_feature],
        y=[y_feature],
        z=[z_feature],
        mode='markers',
        marker=dict(
            size=20,
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

def main():
    # Main title
    st.markdown('<h1 class="main-header">🎵 Enhanced Audio Spectrum Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    st.sidebar.markdown("Upload an audio file to begin analysis")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
    )
    
    if uploaded_file is not None:
        # Load audio
        with st.spinner("Loading audio file..."):
            audio_data, sr = load_audio(uploaded_file)
        
        if audio_data is not None:
            # Display basic info
            st.success(f"✅ Audio loaded successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="metric-card"><h3>Duration</h3><h2>{len(audio_data)/sr:.2f}s</h2></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="metric-card"><h3>Sample Rate</h3><h2>{sr:,} Hz</h2></div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="metric-card"><h3>Samples</h3><h2>{len(audio_data):,}</h2></div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'<div class="metric-card"><h3>File Size</h3><h2>{len(audio_data)*4/1024/1024:.1f} MB</h2></div>', unsafe_allow_html=True)
            
            # Audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Extract features
            with st.spinner("Extracting audio features..."):
                features = extract_comprehensive_features(audio_data, sr)
            
            # Display tempo with error handling
            tempo_value = features.get('Tempo', 0)
            st.markdown(f'<div class="metric-card"><h3>Tempo</h3><h2>{tempo_value:.1f} BPM</h2></div>', unsafe_allow_html=True)
            
            # Analysis options
            st.sidebar.subheader("📊 Analysis Options")
            show_waveform = st.sidebar.checkbox("Waveform Analysis", value=True)
            show_spectrum = st.sidebar.checkbox("Spectral Analysis", value=True)
            show_features = st.sidebar.checkbox("Feature Analysis", value=True)
            show_3d = st.sidebar.checkbox("3D Visualization", value=False)
            
            # Waveform analysis
            if show_waveform:
                st.header("🌊 Waveform Analysis")
                with st.spinner("Creating waveform plots..."):
                    waveform_fig = create_enhanced_waveform_plot(audio_data, sr)
                    st.plotly_chart(waveform_fig, use_container_width=True)
            
            # Spectral analysis
            if show_spectrum:
                st.header("🌈 Spectral Analysis")
                with st.spinner("Creating spectral plots..."):
                    spectrum_fig = create_enhanced_spectrum_plot(audio_data, sr)
                    st.plotly_chart(spectrum_fig, use_container_width=True)
            
            # Feature analysis
            if show_features:
                st.header("📈 Feature Analysis")
                
                # Feature summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Key Features")
                    key_features = {
                        'Spectral Centroid': f"{features['Spectral_Centroid_Mean']:.1f} Hz",
                        'RMS Energy': f"{features['RMS_Energy']:.4f}",
                        'Zero Crossing Rate': f"{features['Zero_Crossing_Rate']:.4f}",
                        'Tempo': f"{features['Tempo']:.1f} BPM"
                    }
                    
                    for feature, value in key_features.items():
                        st.markdown(f'<div class="feature-box"><strong>{feature}:</strong> {value}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.subheader("📊 Statistical Summary")
                    stats = {
                        'Mean Amplitude': f"{features['Audio_Mean']:.4f}",
                        'Std Deviation': f"{features['Audio_Std']:.4f}",
                        'Skewness': f"{features['Audio_Skewness']:.4f}",
                        'Kurtosis': f"{features['Audio_Kurtosis']:.4f}"
                    }
                    
                    for stat, value in stats.items():
                        st.markdown(f'<div class="feature-box"><strong>{stat}:</strong> {value}</div>', unsafe_allow_html=True)
                
                # Feature comparison plot
                with st.spinner("Creating feature plots..."):
                    feature_fig = create_feature_comparison_plot(features)
                    st.plotly_chart(feature_fig, use_container_width=True)
                
                # Detailed feature table
                st.subheader("📋 Complete Feature Set")
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                st.dataframe(feature_df, use_container_width=True)
            
            # 3D visualization
            if show_3d:
                st.header("🎨 3D Feature Visualization")
                with st.spinner("Creating 3D plot..."):
                    fig_3d = create_3d_feature_plot(features)
                    st.plotly_chart(fig_3d, use_container_width=True)
            
            # Download features
            st.sidebar.subheader("💾 Export")
            if st.sidebar.button("Download Features as CSV"):
                feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
                csv = feature_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"audio_features_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Please upload an audio file to begin analysis")
        
        # Show example features
        st.header("🎵 What You'll Get")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🌊 Waveform Analysis
            - Raw audio waveform
            - Envelope detection
            - RMS energy tracking
            - Spectral centroid over time
            """)
        
        with col2:
            st.markdown("""
            ### 🌈 Spectral Analysis
            - Magnitude spectrogram
            - Mel-frequency spectrogram
            - Chromagram
            - Phase analysis
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Feature Extraction
            - 40+ audio features
            - Temporal characteristics
            - Spectral properties
            - Rhythm analysis
            """)

if __name__ == "__main__":
    main()
