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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #667eea 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .analysis-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #333;
        font-weight: bold;
    }
    
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .real-time-indicator {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        padding: 10px 20px;
        border-radius: 20px;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
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
    envelope = np.abs(signal.hilbert(audio_data))
    rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    
    fig.add_trace(go.Scatter(
        x=time, y=envelope,
        mode='lines', name='Envelope',
        line=dict(color='#f093fb', width=2),
        fill='tonexty'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=rms_times, y=rms,
        mode='lines', name='RMS Energy',
        line=dict(color='#f5576c', width=2)
    ), row=2, col=1)
    
    # Spectral centroid over time
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    cent_times = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
    
    fig.add_trace(go.Scatter(
        x=cent_times, y=spectral_centroids,
        mode='lines', name='Spectral Centroid',
        line=dict(color='#764ba2', width=2),
        fill='tozeroy'
    ), row=3, col=1)
    
    fig.update_layout(
        title='🎵 Enhanced Waveform Analysis',
        template='plotly_dark',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Energy", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
    
    return fig

def create_3d_spectrogram(audio_data, sr):
    """Create a 3D spectrogram visualization"""
    # Compute spectrogram
    D = librosa.stft(audio_data, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        z=S_db,
        x=times,
        y=freqs[:len(S_db)],
        colorscale='Viridis',
        showscale=True,
        hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Magnitude: %{z:.1f}dB<extra></extra>'
    )])
    
    fig.update_layout(
        title='🌟 3D Spectrogram Surface',
        scene=dict(
            xaxis_title='Time (seconds)',
            yaxis_title='Frequency (Hz)',
            zaxis_title='Magnitude (dB)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        template='plotly_dark',
        height=600
    )
    
    return fig

def create_mel_spectrogram(audio_data, sr):
    """Create mel-scale spectrogram"""
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    times = librosa.frames_to_time(np.arange(mel_spec_db.shape[1]), sr=sr)
    
    fig = go.Figure(data=go.Heatmap(
        z=mel_spec_db,
        x=times,
        y=np.arange(128),
        colorscale='Plasma',
        hovertemplate='Time: %{x:.2f}s<br>Mel Band: %{y}<br>Power: %{z:.1f}dB<extra></extra>'
    ))
    
    fig.update_layout(
        title='🎶 Mel-Scale Spectrogram',
        xaxis_title='Time (seconds)',
        yaxis_title='Mel Bands',
        template='plotly_dark',
        height=500
    )
    
    return fig

def create_harmonic_percussive_analysis(audio_data, sr):
    """Separate and analyze harmonic and percussive components"""
    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Original Audio', 'Harmonic Component', 'Percussive Component'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Original
    fig.add_trace(go.Scatter(
        x=time, y=audio_data,
        mode='lines', name='Original',
        line=dict(color='#667eea', width=1)
    ), row=1, col=1)
    
    # Harmonic
    fig.add_trace(go.Scatter(
        x=time, y=y_harmonic,
        mode='lines', name='Harmonic',
        line=dict(color='#f093fb', width=1)
    ), row=2, col=1)
    
    # Percussive
    fig.add_trace(go.Scatter(
        x=time, y=y_percussive,
        mode='lines', name='Percussive',
        line=dict(color='#f5576c', width=1)
    ), row=3, col=1)
    
    fig.update_layout(
        title='🎭 Harmonic-Percussive Source Separation',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    
    return fig, y_harmonic, y_percussive

def create_onset_detection(audio_data, sr):
    """Create onset detection visualization"""
    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr, units='time')
    onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
    times = librosa.frames_to_time(np.arange(len(onset_strength)), sr=sr)
    
    fig = go.Figure()
    
    # Onset strength
    fig.add_trace(go.Scatter(
        x=times, y=onset_strength,
        mode='lines', name='Onset Strength',
        line=dict(color='#667eea', width=2),
        fill='tozeroy'
    ))
    
    # Onset markers
    for onset_time in onset_frames:
        fig.add_vline(
            x=onset_time,
            line=dict(color='#f5576c', width=2, dash='dash'),
            opacity=0.8
        )
    
    fig.update_layout(
        title=f'🎯 Onset Detection ({len(onset_frames)} onsets detected)',
        xaxis_title='Time (seconds)',
        yaxis_title='Onset Strength',
        template='plotly_dark',
        height=400
    )
    
    return fig, onset_frames

def create_tonnetz_visualization(audio_data, sr):
    """Create Tonnetz (tonal centroid) visualization"""
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
    times = librosa.frames_to_time(np.arange(tonnetz.shape[1]), sr=sr)
    
    fig = go.Figure(data=go.Heatmap(
        z=tonnetz,
        x=times,
        y=['5th (x)', '5th (y)', 'min3rd (x)', 'min3rd (y)', 'maj3rd (x)', 'maj3rd (y)'],
        colorscale='RdBu',
        hovertemplate='Time: %{x:.2f}s<br>Dimension: %{y}<br>Value: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🎼 Tonnetz (Tonal Centroid Features)',
        xaxis_title='Time (seconds)',
        yaxis_title='Tonal Dimensions',
        template='plotly_dark',
        height=400
    )
    
    return fig

def extract_comprehensive_features(audio_data, sr):
    """Extract comprehensive audio features"""
    features = {}
    
    # Basic features
    features['Duration'] = len(audio_data) / sr
    features['Sample Rate'] = sr
    features['RMS Energy'] = np.sqrt(np.mean(audio_data**2))
    features['Max Amplitude'] = np.max(np.abs(audio_data))
    features['Dynamic Range'] = np.max(audio_data) - np.min(audio_data)
    
    # Statistical features
    features['Mean'] = np.mean(audio_data)
    features['Std Dev'] = np.std(audio_data)
    features['Skewness'] = skew(audio_data)
    features['Kurtosis'] = kurtosis(audio_data)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    features['Spectral Centroid Mean'] = np.mean(spectral_centroids)
    features['Spectral Centroid Std'] = np.std(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    features['Spectral Rolloff Mean'] = np.mean(spectral_rolloff)
    features['Spectral Rolloff Std'] = np.std(spectral_rolloff)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    features['Spectral Bandwidth Mean'] = np.mean(spectral_bandwidth)
    features['Spectral Bandwidth Std'] = np.std(spectral_bandwidth)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    features['Spectral Contrast Mean'] = np.mean(spectral_contrast)
    features['Spectral Contrast Std'] = np.std(spectral_contrast)
    
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
    features['Spectral Flatness Mean'] = np.mean(spectral_flatness)
    features['Spectral Flatness Std'] = np.std(spectral_flatness)
    
    # Rhythm features
    features['Zero Crossing Rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
    
    # Tempo and beat features
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr)
    features['Tempo'] = tempo
    features['Beat Count'] = len(beat_frames)
    features['Beat Density'] = len(beat_frames) / features['Duration']
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'MFCC_{i+1}_Mean'] = np.mean(mfccs[i])
        features[f'MFCC_{i+1}_Std'] = np.std(mfccs[i])
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['Chroma Mean'] = np.mean(chroma)
    features['Chroma Std'] = np.std(chroma)
    
    # Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sr)
    features['Tonnetz Mean'] = np.mean(tonnetz)
    features['Tonnetz Std'] = np.std(tonnetz)
    
    # Harmonic-percussive features
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    features['Harmonic RMS'] = np.sqrt(np.mean(y_harmonic**2))
    features['Percussive RMS'] = np.sqrt(np.mean(y_percussive**2))
    features['Harmonic-Percussive Ratio'] = features['Harmonic RMS'] / (features['Percussive RMS'] + 1e-10)
    
    return features

def create_advanced_feature_analysis(features):
    """Create advanced feature analysis visualizations"""
    # Feature importance based on variance
    feature_names = list(features.keys())
    feature_values = list(features.values())
    
    # Create multiple subplots for different feature categories
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Statistical Features', 'Spectral Features', 'Rhythm Features', 'MFCC Features'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Statistical features
    stat_features = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'RMS Energy', 'Max Amplitude']
    stat_values = [features.get(f, 0) for f in stat_features]
    
    fig.add_trace(go.Bar(
        x=stat_features, y=stat_values,
        name='Statistical', marker_color='#667eea'
    ), row=1, col=1)
    
    # Spectral features
    spec_features = ['Spectral Centroid Mean', 'Spectral Rolloff Mean', 'Spectral Bandwidth Mean', 'Spectral Contrast Mean']
    spec_values = [features.get(f, 0) for f in spec_features]
    
    fig.add_trace(go.Bar(
        x=spec_features, y=spec_values,
        name='Spectral', marker_color='#f093fb'
    ), row=1, col=2)
    
    # Rhythm features
    rhythm_features = ['Tempo', 'Beat Count', 'Beat Density', 'Zero Crossing Rate']
    rhythm_values = [features.get(f, 0) for f in rhythm_features]
    
    fig.add_trace(go.Bar(
        x=rhythm_features, y=rhythm_values,
        name='Rhythm', marker_color='#f5576c'
    ), row=2, col=1)
    
    # MFCC features (first 5)
    mfcc_features = [f'MFCC_{i+1}_Mean' for i in range(5)]
    mfcc_values = [features.get(f, 0) for f in mfcc_features]
    
    fig.add_trace(go.Bar(
        x=mfcc_features, y=mfcc_values,
        name='MFCC', marker_color='#764ba2'
    ), row=2, col=2)
    
    fig.update_layout(
        title='📊 Advanced Feature Analysis',
        template='plotly_dark',
        height=600,
        showlegend=False
    )
    
    return fig

def create_real_time_simulation(audio_data, sr, segment_duration=0.1):
    """Create a real-time visualization simulation"""
    segment_length = int(segment_duration * sr)
    n_segments = len(audio_data) // segment_length
    
    time_segments = []
    rms_segments = []
    centroid_segments = []
    
    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = min((i + 1) * segment_length, len(audio_data))
        segment = audio_data[start_idx:end_idx]
        
        time_segments.append(i * segment_duration)
        rms_segments.append(np.sqrt(np.mean(segment**2)))
        
        if len(segment) > 0:
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            centroid_segments.append(np.mean(centroid))
        else:
            centroid_segments.append(0)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Real-Time RMS Energy', 'Real-Time Spectral Centroid'),
        shared_xaxes=True
    )
    
    fig.add_trace(go.Scatter(
        x=time_segments, y=rms_segments,
        mode='lines+markers', name='RMS Energy',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=time_segments, y=centroid_segments,
        mode='lines+markers', name='Spectral Centroid',
        line=dict(color='#f093fb', width=3),
        marker=dict(size=8)
    ), row=2, col=1)
    
    fig.update_layout(
        title='⚡ Real-Time Audio Analysis Simulation',
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="RMS Energy", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
    
    return fig

# Main app
def main():
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Enhanced Audio Spectrum Visualizer</h1>
        <p>Advanced real-time audio analysis with comprehensive time and frequency domain features</p>
        <div class="real-time-indicator">
            🔴 LIVE ANALYSIS MODE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced controls
    st.sidebar.title("🎛️ Advanced Controls")
    st.sidebar.markdown("### Upload Audio File")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
        help="Upload an audio file in various formats"
    )
    
    if uploaded_file is not None:
        # Load audio with progress
        with st.spinner("🎵 Loading and analyzing audio file..."):
            audio_data, sr = load_audio(uploaded_file)
        
        if audio_data is not None:
            # Audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Extract comprehensive features
            with st.spinner("🔍 Extracting comprehensive features..."):
                features = extract_comprehensive_features(audio_data, sr)
            
            # Enhanced metrics display
            st.markdown("### 📊 Audio Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⏱️ Duration</h3>
                    <h2>{features['Duration']:.2f}s</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Sample Rate</h3>
                    <h2>{features['Sample Rate']:,}Hz</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎵 Tempo</h3>
                    <h2>{features['Tempo']:.1f} BPM</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ RMS Energy</h3>
                    <h2>{features['RMS Energy']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎼 Spectral Centroid</h3>
                    <h2>{features['Spectral Centroid Mean']:.0f}Hz</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced visualization selection
            st.sidebar.markdown("### 🎨 Visualization Options")
            viz_categories = {
                "Basic Analysis": ["Enhanced Waveform", "3D Spectrogram", "Mel Spectrogram"],
                "Advanced Analysis": ["Harmonic-Percussive", "Onset Detection", "Tonnetz"],
                "Real-Time Features": ["Real-Time Simulation", "Feature Analysis", "Comprehensive Profile"]
            }
            
            selected_viz = []
            for category, options in viz_categories.items():
                st.sidebar.markdown(f"**{category}**")
                for option in options:
                    if st.sidebar.checkbox(option, key=option):
                        selected_viz.append(option)
            
            # Create visualizations based on selection
            if "Enhanced Waveform" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                with st.spinner("🌊 Creating enhanced waveform..."):
                    fig_waveform = create_enhanced_waveform_plot(audio_data, sr)
                    st.plotly_chart(fig_waveform, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "3D Spectrogram" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                with st.spinner("🌟 Creating 3D spectrogram..."):
                    fig_3d = create_3d_spectrogram(audio_data, sr)
                    st.plotly_chart(fig_3d, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "Mel Spectrogram" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                with st.spinner("🎶 Creating mel spectrogram..."):
                    fig_mel = create_mel_spectrogram(audio_data, sr)
                    st.plotly_chart(fig_mel, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "Harmonic-Percussive" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                with st.spinner("🎭 Analyzing harmonic-percussive components..."):
                    fig_hp, y_harm, y_perc = create_harmonic_percussive_analysis(audio_data, sr)
                    st.plotly_chart(fig_hp, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f'<div class="feature-highlight">Harmonic RMS: {features["Harmonic RMS"]:.4f}</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="feature-highlight">Percussive RMS: {features["Percussive RMS"]:.4f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "Onset Detection" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                with st.spinner("🎯 Detecting onsets..."):
                    fig_onset, onset_times = create_onset_detection(audio_data, sr)
                    st.plotly_chart(fig_onset, use_container_width=True)
                    st.markdown(f'<div class="feature-highlight">Detected {len(onset_times)} onsets with average spacing of {np.mean(np.diff(onset_times)):.3f} seconds</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "Tonnetz" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                with st.spinner("🎼 Creating tonnetz visualization..."):
                    fig_tonnetz = create_tonnetz_visualization(audio_data, sr)
                    st.plotly_chart(fig_tonnetz, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "Real-Time Simulation" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### ⚡ Real-Time Analysis Simulation")
                
                # Real-time simulation controls
                col1, col2 = st.columns(2)
                with col1:
                    segment_duration = st.slider("Segment Duration (seconds)", 0.05, 1.0, 0.1, 0.05)
                with col2:
                    update_rate = st.slider("Update Rate (Hz)", 10, 100, 50, 10)
                
                with st.spinner("⚡ Creating real-time simulation..."):
                    fig_realtime = create_real_time_simulation(audio_data, sr, segment_duration)
                    st.plotly_chart(fig_realtime, use_container_width=True)
                
                st.markdown(f'<div class="feature-highlight">Simulating real-time analysis with {1/segment_duration:.0f} updates per second</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "Feature Analysis" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                with st.spinner("📊 Creating advanced feature analysis..."):
                    fig_features = create_advanced_feature_analysis(features)
                    st.plotly_chart(fig_features, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if "Comprehensive Profile" in selected_viz:
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 🎯 Comprehensive Audio Profile")
                
                # Create comprehensive radar chart
                radar_features = {
                    'RMS Energy': min(features['RMS Energy'] * 10, 1.0),
                    'Spectral Centroid': features['Spectral Centroid Mean'] / 5000,
                    'Spectral Rolloff': features['Spectral Rolloff Mean'] / 10000,
                    'Spectral Bandwidth': features['Spectral Bandwidth Mean'] / 3000,
                    'Zero Crossing Rate': features['Zero Crossing Rate'] * 100,
                    'Tempo': features['Tempo'] / 200,
                    'Beat Density': min(features['Beat Density'] / 2, 1.0),
                    'Harmonic-Percussive Ratio': min(features['Harmonic-Percussive Ratio'] / 5, 1.0),
                    'Spectral Contrast': min(features['Spectral Contrast Mean'] / 30, 1.0),
                    'Spectral Flatness': features['Spectral Flatness Mean'] * 100
                }
                
                categories = list(radar_features.keys())
                values = list(radar_features.values())
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Audio Profile',
                    line=dict(color='#667eea', width=3),
                    fillcolor='rgba(102, 126, 234, 0.3)',
                    marker=dict(size=8)
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            gridcolor='rgba(255, 255, 255, 0.3)',
                            tickfont=dict(color='white')
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255, 255, 255, 0.3)',
                            tickfont=dict(color='white')
                        ),
                        bgcolor='rgba(0, 0, 0, 0.1)'
                    ),
                    showlegend=True,
                    title='🎯 Comprehensive Audio Feature Profile',
                    template='plotly_dark',
                    height=600,
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced analysis section
            st.markdown("### 🔬 Advanced Analysis Tools")
            
            analysis_tabs = st.tabs(["Statistical Analysis", "Spectral Analysis", "Temporal Analysis", "Export Data"])
            
            with analysis_tabs[0]:
                st.markdown("#### Statistical Properties")
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.metric("Mean Amplitude", f"{features['Mean']:.6f}")
                    st.metric("Standard Deviation", f"{features['Std Dev']:.6f}")
                
                with stat_cols[1]:
                    st.metric("Skewness", f"{features['Skewness']:.4f}")
                    st.metric("Kurtosis", f"{features['Kurtosis']:.4f}")
                
                with stat_cols[2]:
                    st.metric("Dynamic Range", f"{features['Dynamic Range']:.4f}")
                    st.metric("Max Amplitude", f"{features['Max Amplitude']:.4f}")
                
                with stat_cols[3]:
                    st.metric("RMS Energy", f"{features['RMS Energy']:.6f}")
                    st.metric("Zero Crossing Rate", f"{features['Zero Crossing Rate']:.6f}")
                
                # Distribution analysis
                st.markdown("#### Amplitude Distribution")
                
                fig_dist = go.Figure()
                
                # Create histogram
                hist_data = np.histogram(audio_data, bins=50)
                fig_dist.add_trace(go.Bar(
                    x=hist_data[1][:-1],
                    y=hist_data[0],
                    name='Amplitude Distribution',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                fig_dist.update_layout(
                    title='📊 Amplitude Distribution',
                    xaxis_title='Amplitude',
                    yaxis_title='Frequency',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with analysis_tabs[1]:
                st.markdown("#### Spectral Properties")
                
                spec_cols = st.columns(3)
                
                with spec_cols[0]:
                    st.metric("Spectral Centroid (Mean)", f"{features['Spectral Centroid Mean']:.2f} Hz")
                    st.metric("Spectral Centroid (Std)", f"{features['Spectral Centroid Std']:.2f} Hz")
                    st.metric("Spectral Rolloff (Mean)", f"{features['Spectral Rolloff Mean']:.2f} Hz")
                
                with spec_cols[1]:
                    st.metric("Spectral Bandwidth (Mean)", f"{features['Spectral Bandwidth Mean']:.2f} Hz")
                    st.metric("Spectral Contrast (Mean)", f"{features['Spectral Contrast Mean']:.4f}")
                    st.metric("Spectral Flatness (Mean)", f"{features['Spectral Flatness Mean']:.6f}")
                
                with spec_cols[2]:
                    st.metric("Chroma (Mean)", f"{features['Chroma Mean']:.4f}")
                    st.metric("Tonnetz (Mean)", f"{features['Tonnetz Mean']:.4f}")
                    st.metric("Harmonic-Percussive Ratio", f"{features['Harmonic-Percussive Ratio']:.4f}")
                
                # MFCC Analysis
                st.markdown("#### MFCC Analysis")
                mfcc_means = [features[f'MFCC_{i+1}_Mean'] for i in range(13)]
                mfcc_stds = [features[f'MFCC_{i+1}_Std'] for i in range(13)]
                
                fig_mfcc = go.Figure()
                
                fig_mfcc.add_trace(go.Bar(
                    x=list(range(1, 14)),
                    y=mfcc_means,
                    name='MFCC Mean',
                    marker_color='#f093fb',
                    error_y=dict(type='data', array=mfcc_stds, visible=True)
                ))
                
                fig_mfcc.update_layout(
                    title='🎵 MFCC Coefficients (Mean ± Std)',
                    xaxis_title='MFCC Coefficient',
                    yaxis_title='Value',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig_mfcc, use_container_width=True)
            
            with analysis_tabs[2]:
                st.markdown("#### Temporal Properties")
                
                temp_cols = st.columns(3)
                
                with temp_cols[0]:
                    st.metric("Tempo", f"{features['Tempo']:.2f} BPM")
                    st.metric("Beat Count", f"{features['Beat Count']}")
                    st.metric("Beat Density", f"{features['Beat Density']:.4f} beats/sec")
                
                with temp_cols[1]:
                    st.metric("Duration", f"{features['Duration']:.2f} seconds")
                    st.metric("Harmonic RMS", f"{features['Harmonic RMS']:.6f}")
                    st.metric("Percussive RMS", f"{features['Percussive RMS']:.6f}")
                
                with temp_cols[2]:
                    if 'onset_times' in locals():
                        st.metric("Onset Count", f"{len(onset_times)}")
                        st.metric("Onset Density", f"{len(onset_times)/features['Duration']:.4f} onsets/sec")
                    st.metric("Sample Rate", f"{features['Sample Rate']:,} Hz")
                
                # Tempo stability analysis
                st.markdown("#### Tempo Stability")
                
                # Create tempo over time analysis
                tempo_segments = []
                segment_times = []
                segment_length = sr * 5  # 5-second segments
                
                for i in range(0, len(audio_data), segment_length):
                    segment = audio_data[i:i+segment_length]
                    if len(segment) > sr:  # At least 1 second
                        try:
                            tempo_seg, _ = librosa.beat.beat_track(y=segment, sr=sr)
                            tempo_segments.append(tempo_seg)
                            segment_times.append(i / sr)
                        except:
                            pass
                
                if tempo_segments:
                    fig_tempo_stability = go.Figure()
                    
                    fig_tempo_stability.add_trace(go.Scatter(
                        x=segment_times,
                        y=tempo_segments,
                        mode='lines+markers',
                        name='Tempo over Time',
                        line=dict(color='#f5576c', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig_tempo_stability.update_layout(
                        title='🥁 Tempo Stability Over Time',
                        xaxis_title='Time (seconds)',
                        yaxis_title='Tempo (BPM)',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig_tempo_stability, use_container_width=True)
                    
                    tempo_stability = np.std(tempo_segments)
                    st.metric("Tempo Stability (Lower = More Stable)", f"{tempo_stability:.2f}")
            
            with analysis_tabs[3]:
                st.markdown("#### Export Analysis Data")
                
                # Create comprehensive feature DataFrame
                feature_df = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])
                feature_df.index.name = 'Feature'
                
                # Display summary statistics
                st.markdown("##### Feature Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Features", len(features))
                    st.metric("Audio Duration", f"{features['Duration']:.2f} seconds")
                
                with col2:
                    st.metric("Sample Rate", f"{features['Sample Rate']:,} Hz")
                    st.metric("File Size", f"{len(audio_data)} samples")
                
                # Feature data table
                st.markdown("##### All Features")
                st.dataframe(
                    feature_df.style.format({'Value': '{:.8f}'}),
                    use_container_width=True,
                    height=400
                )
                
                # Export options
                st.markdown("##### Export Options")
                export_cols = st.columns(3)
                
                with export_cols[0]:
                    csv_data = feature_df.to_csv()
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"audio_features_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                with export_cols[1]:
                    json_data = feature_df.to_json(orient='index', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"audio_features_{uploaded_file.name}.json",
                        mime="application/json"
                    )
                
                with export_cols[2]:
                    # Create summary report
                    report = f"""
# Audio Analysis Report

## File Information
- **Filename**: {uploaded_file.name}
- **Duration**: {features['Duration']:.2f} seconds
- **Sample Rate**: {features['Sample Rate']:,} Hz
- **Channels**: Mono

## Key Features
- **Tempo**: {features['Tempo']:.2f} BPM
- **RMS Energy**: {features['RMS Energy']:.6f}
- **Spectral Centroid**: {features['Spectral Centroid Mean']:.2f} Hz
- **Zero Crossing Rate**: {features['Zero Crossing Rate']:.6f}

## Statistical Properties
- **Mean**: {features['Mean']:.6f}
- **Standard Deviation**: {features['Std Dev']:.6f}
- **Skewness**: {features['Skewness']:.4f}
- **Kurtosis**: {features['Kurtosis']:.4f}

## Spectral Analysis
- **Spectral Rolloff**: {features['Spectral Rolloff Mean']:.2f} Hz
- **Spectral Bandwidth**: {features['Spectral Bandwidth Mean']:.2f} Hz
- **Spectral Contrast**: {features['Spectral Contrast Mean']:.4f}
- **Spectral Flatness**: {features['Spectral Flatness Mean']:.6f}

## Harmonic-Percussive Analysis
- **Harmonic RMS**: {features['Harmonic RMS']:.6f}
- **Percussive RMS**: {features['Percussive RMS']:.6f}
- **H-P Ratio**: {features['Harmonic-Percussive Ratio']:.4f}

---
*Generated by Enhanced Audio Spectrum Visualizer*
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"audio_report_{uploaded_file.name}.md",
                        mime="text/markdown"
                    )
    
    else:
        st.info("👆 Please upload an audio file to start the enhanced analysis!")
        
        # Enhanced demo section
        st.markdown("### 🎵 Enhanced Features Available:")
        
        feature_cols = st.columns(3)
        
        with feature_cols[0]:
            st.markdown("""
            #### 🌊 Time Domain Analysis
            - **Enhanced Waveform** with envelope and RMS
            - **Harmonic-Percussive Separation**
            - **Onset Detection** with precise timing
            - **Real-Time Simulation** of live analysis
            """)
        
        with feature_cols[1]:
            st.markdown("""
            #### 🌈 Frequency Domain Analysis
            - **3D Spectrogram** visualization
            - **Mel-Scale Spectrogram**
            - **Spectral Features** (centroid, rolloff, bandwidth)
            - **MFCC Analysis** (20 coefficients)
            """)
        
        with feature_cols[2]:
            st.markdown("""
            #### 🎯 Advanced Features
            - **Tonnetz** (tonal centroid features)
            - **Comprehensive Feature Profile**
            - **Statistical Analysis**
            - **Export Capabilities** (CSV, JSON, Report)
            """)
        
        st.markdown("### 🔬 What Makes This Enhanced?")
        st.markdown("""
        - **Real-time simulation** of live audio analysis
        - **Comprehensive feature extraction** (100+ features)
        - **Advanced visualizations** including 3D plots
        - **Statistical analysis** with distribution plots
        - **Export capabilities** for further analysis
        - **Interactive controls** for customization
        - **Professional-grade** audio processing
        """)

if __name__ == "__main__":
    main()
