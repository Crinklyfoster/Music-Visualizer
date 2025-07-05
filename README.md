# 🎵 Audio Spectrum Visualizer

A beautiful and interactive Streamlit web application for visualizing audio files with advanced signal processing and stunning visualizations.

![Audio Visualizer](https://img.shields.io/badge/Audio-Visualizer-blue?style=for-the-badge&logo=music)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Formats](#-supported-formats)
- [Visualizations](#-visualizations)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🎨 **Beautiful UI Design**
- Modern gradient backgrounds and custom CSS styling
- Dark theme with vibrant, professional colors
- Interactive metric cards and responsive layout
- Smooth animations and hover effects

### 📊 **Advanced Visualizations**
- **Interactive Waveform** with envelope detection
- **Spectrogram** showing time-frequency representation
- **Frequency Domain Analysis** with logarithmic scaling
- **Chromagram** displaying pitch class distribution
- **Tempo Analysis** with beat detection
- **Feature Radar Chart** for multi-dimensional audio characteristics

### 🔧 **Audio Processing Features**
- Real-time audio feature extraction
- MFCC (Mel-Frequency Cepstral Coefficients) analysis
- Spectral feature computation
- Beat and tempo detection
- Onset strength analysis

### 💾 **Data Export**
- Download extracted features as CSV
- Export analysis results for further processing

## 🎬 Demo

The application provides:
1. **File Upload Interface** - Drag and drop audio files
2. **Audio Player** - Built-in playback controls
3. **Real-time Metrics** - Duration, sample rate, energy, etc.
4. **Interactive Visualizations** - Zoom, pan, and hover for details
5. **Feature Analysis** - Detailed audio characteristics

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/audio-spectrum-visualizer.git
cd audio-spectrum-visualizer
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run audio_visualizer.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 🎯 Usage

1. **Launch the App**: Run `streamlit run audio_visualizer.py`
2. **Upload Audio**: Click "Choose an audio file" in the sidebar
3. **Select Visualizations**: Choose which charts to display
4. **Explore**: Interact with the visualizations using zoom, pan, and hover
5. **Analyze**: View detailed audio features and metrics
6. **Export**: Download feature data as CSV for further analysis

### Quick Start Example
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run audio_visualizer.py

# Open browser to http://localhost:8501
# Upload an audio file and explore!
```

## 🎼 Supported Formats

The application supports the following audio formats:
- **WAV** (.wav) - Uncompressed audio
- **MP3** (.mp3) - Compressed audio
- **FLAC** (.flac) - Lossless compression
- **OGG** (.ogg) - Open-source format
- **M4A** (.m4a) - Apple audio format

## 📈 Visualizations

### 1. 🌊 Interactive Waveform
- Shows audio amplitude over time
- Includes envelope detection
- Hover for precise time and amplitude values

### 2. 🌈 Spectrogram
- Time-frequency representation
- Color-coded magnitude in dB
- Interactive heatmap with zoom capabilities

### 3. 📊 Frequency Analysis
- Frequency domain representation
- Logarithmic frequency scale
- Magnitude response in decibels

### 4. 🎼 Chromagram
- Pitch class distribution over time
- Shows musical note content
- Useful for music analysis

### 5. 🥁 Tempo Analysis
- Beat detection and tracking
- Onset strength visualization
- Automatic BPM calculation

### 6. 🎯 Feature Radar Chart
- Multi-dimensional audio characteristics
- Normalized feature representation
- Visual feature profile comparison

## 🔬 Technical Details

### Audio Processing Pipeline
1. **Audio Loading**: Uses librosa for robust audio file handling
2. **Feature Extraction**: Computes spectral and temporal features
3. **Signal Processing**: FFT, STFT, and other transforms
4. **Visualization**: Interactive plots with Plotly

### Key Libraries Used
- **Streamlit**: Web app framework
- **Librosa**: Audio analysis library
- **Plotly**: Interactive visualizations
- **NumPy/SciPy**: Numerical computing
- **Pandas**: Data manipulation

### Extracted Features
- **Basic**: Duration, sample rate, RMS energy
- **Spectral**: Centroid, rolloff, bandwidth
- **Temporal**: Zero crossing rate, onset strength
- **Cepstral**: 13 MFCC coefficients
- **Rhythmic**: Tempo, beat locations

## 🎨 Customization

### Styling
The app uses custom CSS for beautiful styling. You can modify the styles in the `st.markdown()` sections:

```python
# Custom CSS example
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        /* Add your custom styles here */
    }
</style>
""", unsafe_allow_html=True)
```

### Adding New Visualizations
To add new visualizations:

1. Create a new function following the pattern:
```python
def create_new_visualization(audio_data, sr):
    # Your visualization code here
    return fig
```

2. Add it to the visualization options in the sidebar
3. Include it in the main visualization loop

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/audio-spectrum-visualizer.git

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # Optional: for code formatting and testing
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Librosa** team for the excellent audio analysis library
- **Streamlit** team for the amazing web app framework
- **Plotly** team for interactive visualization tools
- The open-source community for continuous inspiration

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/audio-spectrum-visualizer/issues) page
2. Create a new issue with detailed description
3. Include your Python version and error messages

## 🔄 Updates

### Version 1.0.0
- Initial release with core visualization features
- Support for major audio formats
- Interactive UI with modern design
- Feature extraction and export capabilities

---

**Made with ❤️ and 🎵 by [Your Name]**

*Star ⭐ this repository if you find it useful!*
