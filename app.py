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
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎵 Audio Spectrum Visualizer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #a8edea 0%, #fed6e3 100%);
    }
    
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
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

def create_waveform_plot(audio_data, sr):
    """Create an interactive waveform plot"""
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    fig = go.Figure()
    
    # Add waveform
    fig.add_trace(go.Scatter(
        x=time,
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#667eea', width=1),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.4f}<extra></extra>'
    ))
    
    # Add envelope
    envelope = np.abs(signal.hilbert(audio_data))
    fig.add_trace(go.Scatter(
        x=time,
        y=envelope,
        mode='lines',
        name='Envelope',
        line=dict(color='#764ba2', width=2),
        hovertemplate='Time: %{x:.2f}s<br>Envelope: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time,
        y=-envelope,
        mode='lines',
        name='Envelope (Negative)',
        line=dict(color='#764ba2', width=2),
        showlegend=False,
        hovertemplate='Time: %{x:.2f}s<br>Envelope: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🎵 Audio Waveform with Envelope',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        template='plotly_dark',
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_spectrogram_plot(audio_data, sr):
    """Create a beautiful spectrogram plot"""
    # Compute spectrogram
    D = librosa.stft(audio_data)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Create time and frequency axes
    times = librosa.times_like(S_db, sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        x=times,
        y=freqs[:len(S_db)],
        colorscale='Viridis',
        hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Magnitude: %{z:.1f}dB<extra></extra>'
    ))
    
    fig.update_layout(
        title='🌈 Audio Spectrogram',
        xaxis_title='Time (seconds)',
        yaxis_title='Frequency (Hz)',
        template='plotly_dark',
        height=500
    )
    
    return fig

def create_frequency_analysis(audio_data, sr):
    """Create frequency domain analysis"""
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
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frequency,
        y=magnitude_db,
        mode='lines',
        name='Frequency Response',
        line=dict(color='#f093fb', width=2),
        fill='tonexty',
        hovertemplate='Frequency: %{x:.0f}Hz<br>Magnitude: %{y:.1f}dB<extra></extra>'
    ))
    
    fig.update_layout(
        title='📊 Frequency Domain Analysis',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude (dB)',
        template='plotly_dark',
        height=400,
        xaxis_type='log'
    )
    
    return fig

def create_chromagram(audio_data, sr):
    """Create a chromagram visualization"""
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    times = librosa.times_like(chroma, sr=sr)
    
    fig = go.Figure(data=go.Heatmap(
        z=chroma,
        x=times,
        y=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        colorscale='Plasma',
        hovertemplate='Time: %{x:.2f}s<br>Note: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🎼 Chromagram (Pitch Class Distribution)',
        xaxis_title='Time (seconds)',
        yaxis_title='Pitch Class',
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_tempo_analysis(audio_data, sr):
    """Create tempo and beat analysis"""
    # Compute tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Create onset strength
    onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
    times = librosa.times_like(onset_strength, sr=sr)
    
    fig = go.Figure()
    
    # Add onset strength
    fig.add_trace(go.Scatter(
        x=times,
        y=onset_strength,
        mode='lines',
        name='Onset Strength',
        line=dict(color='#667eea', width=2),
        fill='tozeroy'
    ))
    
    # Add beat markers
    for beat_time in beat_times:
        fig.add_vline(
            x=beat_time,
            line=dict(color='#f5576c', width=1, dash='dash'),
            opacity=0.7
        )
    
    fig.update_layout(
        title=f'🥁 Tempo Analysis (BPM: {tempo:.1f})',
        xaxis_title='Time (seconds)',
        yaxis_title='Onset Strength',
        template='plotly_dark',
        height=400
    )
    
    return fig, tempo

def extract_audio_features(audio_data, sr):
    """Extract various audio features"""
    features = {}
    
    # Basic features
    features['Duration'] = len(audio_data) / sr
    features['Sample Rate'] = sr
    features['RMS Energy'] = np.sqrt(np.mean(audio_data**2))
    
    # Spectral features
    features['Spectral Centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
    features['Spectral Rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
    features['Zero Crossing Rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'MFCC_{i+1}'] = np.mean(mfccs[i])
    
    return features

def create_feature_radar_chart(features):
    """Create a radar chart for audio features"""
    # Select key features for radar chart
    radar_features = {
        'RMS Energy': features['RMS Energy'],
        'Spectral Centroid': features['Spectral Centroid'] / 5000,  # Normalize
        'Spectral Rolloff': features['Spectral Rolloff'] / 10000,   # Normalize
        'Zero Crossing Rate': features['Zero Crossing Rate'] * 100,  # Scale up
        'MFCC_1': abs(features['MFCC_1']) / 50,  # Normalize
        'MFCC_2': abs(features['MFCC_2']) / 20,  # Normalize
    }
    
    categories = list(radar_features.keys())
    values = list(radar_features.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Audio Features',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
        showlegend=True,
        title='🎯 Audio Feature Profile',
        template='plotly_dark',
        height=500
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Audio Spectrum Visualizer</h1>
        <p>Upload your audio files and explore beautiful visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    st.sidebar.markdown("Upload an audio file to get started!")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Upload an audio file in WAV, MP3, FLAC, OGG, or M4A format"
    )
    
    if uploaded_file is not None:
        # Load audio
        with st.spinner("🎵 Loading audio file..."):
            audio_data, sr = load_audio(uploaded_file)
        
        if audio_data is not None:
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Extract features
            with st.spinner("🔍 Analyzing audio features..."):
                features = extract_audio_features(audio_data, sr)
            
            # Display basic metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⏱️ Duration</h3>
                    <h2>{features['Duration']:.1f}s</h2>
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
                    <h3>⚡ RMS Energy</h3>
                    <h2>{features['RMS Energy']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎼 Spectral Centroid</h3>
                    <h2>{features['Spectral Centroid']:.0f}Hz</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization selection
            st.sidebar.subheader("📈 Visualizations")
            viz_options = st.sidebar.multiselect(
                "Select visualizations to display:",
                ["Waveform", "Spectrogram", "Frequency Analysis", "Chromagram", "Tempo Analysis", "Feature Profile"],
                default=["Waveform", "Spectrogram", "Frequency Analysis"]
            )
            
            # Create visualizations
            if "Waveform" in viz_options:
                with st.spinner("🌊 Creating waveform..."):
                    fig_waveform = create_waveform_plot(audio_data, sr)
                    st.plotly_chart(fig_waveform, use_container_width=True)
            
            if "Spectrogram" in viz_options:
                with st.spinner("🌈 Creating spectrogram..."):
                    fig_spectrogram = create_spectrogram_plot(audio_data, sr)
                    st.plotly_chart(fig_spectrogram, use_container_width=True)
            
            if "Frequency Analysis" in viz_options:
                with st.spinner("📊 Analyzing frequency domain..."):
                    fig_freq = create_frequency_analysis(audio_data, sr)
                    st.plotly_chart(fig_freq, use_container_width=True)
            
            if "Chromagram" in viz_options:
                with st.spinner("🎼 Creating chromagram..."):
                    fig_chroma = create_chromagram(audio_data, sr)
                    st.plotly_chart(fig_chroma, use_container_width=True)
            
            if "Tempo Analysis" in viz_options:
                with st.spinner("🥁 Analyzing tempo..."):
                    fig_tempo, tempo = create_tempo_analysis(audio_data, sr)
                    st.plotly_chart(fig_tempo, use_container_width=True)
                    st.info(f"🎵 Detected tempo: {tempo:.1f} BPM")
            
            if "Feature Profile" in viz_options:
                with st.spinner("🎯 Creating feature profile..."):
                    fig_radar = create_feature_radar_chart(features)
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            # Feature details
            if st.sidebar.checkbox("Show Detailed Features"):
                st.subheader("🔍 Detailed Audio Features")
                
                # Create a DataFrame for better display
                feature_df = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])
                feature_df.index.name = 'Feature'
                
                # Display as a styled table
                st.dataframe(feature_df.style.format({'Value': '{:.6f}'}), use_container_width=True)
                
                # Download feature data
                csv = feature_df.to_csv()
                st.download_button(
                    label="📥 Download Feature Data",
                    data=csv,
                    file_name=f"audio_features_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Please upload an audio file to start visualizing!")
        
        # Show demo section
        st.markdown("### 🎵 What you'll see:")
        st.markdown("""
        - **🌊 Waveform**: Visual representation of audio amplitude over time
        - **🌈 Spectrogram**: Time-frequency representation showing how frequency content changes
        - **📊 Frequency Analysis**: Frequency domain analysis showing the spectral content
        - **🎼 Chromagram**: Pitch class distribution over time
        - **🥁 Tempo Analysis**: Beat tracking and tempo detection
        - **🎯 Feature Profile**: Radar chart of key audio characteristics
        """)

if __name__ == "__main__":
    main()
