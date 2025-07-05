# 🎵 Enhanced Audio Spectrum Visualizer v2.0

A comprehensive, real-time audio analysis application built with Streamlit that provides advanced time and frequency domain analysis of audio files with professional-grade visualizations.

## 🆕 What's New in v2.0

### 🚀 Major Enhancements
- **Real-Time Analysis Simulation** with customizable parameters
- **3D Spectrogram Visualization** with interactive surface plots
- **Advanced Feature Extraction** (100+ audio features)
- **Comprehensive Export System** (CSV, JSON, Markdown reports)
- **Enhanced UI/UX** with modern glassmorphism design
- **Harmonic-Percussive Separation** for detailed component analysis
- **Tonnetz Visualization** for tonal analysis
- **Statistical Analysis Tools** with distribution plots
- **Tempo Stability Analysis** for rhythm assessment
- **Professional Radar Charts** for feature profiling

### 🎯 Performance Improvements
- **Optimized Processing Pipeline** for faster analysis
- **Memory Efficient** handling of large audio files
- **Streamlined Interface** with organized analysis tabs
- **Enhanced Error Handling** and user feedback
- **Improved Visualization Performance** with better rendering

## ✨ Features

### 🌊 Time Domain Analysis
- **Enhanced Waveform** visualization with envelope and RMS energy
- **Harmonic-Percussive Source Separation** for detailed component analysis
- **Onset Detection** with precise timing and visual markers
- **Real-Time Simulation** of live audio analysis with customizable parameters

### 🌈 Frequency Domain Analysis
- **3D Spectrogram** with interactive surface plots
- **Mel-Scale Spectrogram** for perceptual frequency analysis
- **Comprehensive Spectral Features** (centroid, rolloff, bandwidth, contrast, flatness)
- **MFCC Analysis** with 20 coefficients for audio fingerprinting

### 🎯 Advanced Features
- **Tonnetz Visualization** for tonal centroid features
- **Comprehensive Feature Profile** with radar charts
- **Statistical Analysis** with distribution plots and stability metrics
- **Export Capabilities** (CSV, JSON, Markdown reports)
- **Interactive Controls** for customization and real-time parameter adjustment

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### System Dependencies
For audio processing, you'll need FFmpeg installed on your system:

#### Windows
```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

#### macOS
```bash
# Using Homebrew
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### Python Dependencies
1. Clone or download the repository
2. Navigate to the project directory
3. Install required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload an audio file** using the sidebar file uploader
   - Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC

4. **Select visualizations** from the sidebar checkboxes:
   - Choose from Basic Analysis, Advanced Analysis, or Real-Time Features
   - Multiple visualizations can be selected simultaneously

5. **Explore the results:**
   - View interactive plots and metrics
   - Use the analysis tabs for detailed insights
   - Export data and reports for further analysis

## 📊 Analysis Capabilities

### Extracted Features (100+)
- **Basic Properties**: Duration, sample rate, RMS energy, dynamic range
- **Statistical Features**: Mean, std deviation, skewness, kurtosis
- **Spectral Features**: Centroid, rolloff, bandwidth, contrast, flatness
- **Rhythm Features**: Tempo, beat tracking, onset detection
- **Perceptual Features**: MFCC coefficients, chroma, tonnetz
- **Advanced Features**: Harmonic-percussive separation, zero crossing rate

### Visualizations
- **Enhanced Waveform**: Multi-layer waveform with envelope and spectral centroid
- **3D Spectrogram**: Interactive 3D surface plot of frequency content over time
- **Mel Spectrogram**: Perceptually-weighted frequency analysis
- **Harmonic-Percussive**: Separated harmonic and percussive components
- **Onset Detection**: Precise onset timing with strength visualization
- **Tonnetz**: Tonal centroid features for harmonic analysis
- **Real-Time Simulation**: Live analysis simulation with customizable parameters
- **Feature Profile**: Comprehensive radar chart of all extracted features

## 🎛️ Interface Guide

### Main Dashboard
- **Audio Player**: Play uploaded audio files directly in the browser
- **Metrics Cards**: Key audio properties displayed prominently
- **Visualization Selection**: Choose specific analysis types from the sidebar

### Analysis Tabs
- **Statistical Analysis**: Amplitude distribution and statistical properties
- **Spectral Analysis**: Frequency domain features and MFCC coefficients
- **Temporal Analysis**: Time-based features and tempo stability
- **Export Data**: Download analysis results in multiple formats

## 📁 File Structure

```
audio-visualizer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── sample_audio/      # (Optional) Sample audio files for testing
```

## 🔧 Configuration

### Real-Time Simulation Settings
- **Segment Duration**: 0.05-1.0 seconds (adjustable)
- **Update Rate**: 10-100 Hz (adjustable)
- **Analysis Window**: Customizable for different time resolutions

### Visualization Options
- **Color Schemes**: Multiple professional color palettes
- **Interactive Controls**: Zoom, pan, hover tooltips
- **Export Formats**: PNG, HTML, SVG for visualizations

## 🎯 Use Cases

- **Music Analysis**: Tempo detection, harmonic analysis, onset timing
- **Audio Research**: Feature extraction, spectral analysis, statistical modeling
- **Education**: Learning audio signal processing concepts
- **Quality Control**: Audio file validation and analysis
- **Real-Time Processing**: Simulation of live audio analysis systems

## 🔍 Technical Details

### Audio Processing Pipeline
1. **Loading**: Multi-format audio file support with librosa
2. **Preprocessing**: Normalization and format conversion
3. **Feature Extraction**: Comprehensive time and frequency domain analysis
4. **Visualization**: Interactive plots with Plotly
5. **Export**: Multiple format support for data and reports

### Performance Optimization
- **Efficient Processing**: Optimized librosa operations
- **Memory Management**: Streaming processing for large files
- **Caching**: Streamlit caching for improved performance
- **Parallel Processing**: Multi-threaded feature extraction

## 📝 Export Formats

### Data Export
- **CSV**: Tabular data with all extracted features
- **JSON**: Structured data for programmatic access
- **Markdown**: Comprehensive analysis reports

### Visualization Export
- **PNG**: High-resolution images
- **HTML**: Interactive plots for web sharing
- **SVG**: Vector graphics for publications

## 🛠️ Troubleshooting

### Common Issues

1. **Audio file not loading:**
   - Ensure FFmpeg is properly installed
   - Check file format compatibility
   - Verify file is not corrupted

2. **Slow processing:**
   - Reduce file size or duration
   - Close other applications
   - Check available system memory

3. **Visualization not appearing:**
   - Refresh the browser page
   - Check browser JavaScript is enabled
   - Try a different browser

### Performance Tips
- Use WAV files for fastest processing
- Limit analysis to essential visualizations
- Process shorter audio clips for real-time analysis
- Close unused browser tabs to free memory

## 🔄 Version History

### v2.0 (Current)
- **Complete UI Overhaul**: Modern glassmorphism design with enhanced visual appeal
- **Advanced Feature Extraction**: 100+ comprehensive audio features
- **3D Visualizations**: Interactive 3D spectrogram and surface plots
- **Real-Time Simulation**: Live analysis simulation with customizable parameters
- **Export System**: Multiple format support (CSV, JSON, Markdown)
- **Harmonic-Percussive Analysis**: Advanced source separation techniques
- **Statistical Tools**: Distribution analysis and stability metrics
- **Performance Optimization**: Faster processing and better memory management

### v1.0 (Legacy)
- Basic waveform visualization
- Standard spectrogram analysis
- Simple feature extraction
- Basic export capabilities

## 🔄 Updates and Contributions

This application is actively maintained and updated. For feature requests, bug reports, or contributions:

1. Check the issue tracker for existing reports
2. Create detailed bug reports with audio file examples
3. Suggest new features with use case descriptions
4. Submit pull requests with comprehensive testing

## 📚 Dependencies

- **Streamlit**: Web application framework
- **Librosa**: Audio analysis library
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **SciPy**: Scientific computing
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualization

## 🎵 Sample Audio Files

For testing purposes, consider using:
- Short music clips (30-60 seconds)
- Different genres (electronic, classical, rock)
- Various audio qualities (different sample rates)
- Both mono and stereo files

## 🔒 Privacy and Security

- All audio processing is performed locally
- No audio data is transmitted to external servers
- Files are processed in memory and not stored permanently
- User data privacy is maintained throughout the analysis

## 📞 Support

For technical support or questions:
- Check the troubleshooting section above
- Review the technical documentation
- Test with sample audio files
- Verify system requirements are met

---

**Enhanced Audio Spectrum Visualizer v2.0 - Built with ❤️ using Streamlit and Librosa**

*Professional audio analysis made accessible - Now with advanced real-time capabilities*

## 📋 Changelog v2.0

### 🎨 Interface Improvements
- Modern glassmorphism design with gradient backgrounds
- Enhanced metric cards with hover effects
- Organized analysis tabs for better navigation
- Real-time indicator animations
- Improved responsive design

### 🔧 Technical Enhancements
- Advanced feature extraction pipeline
- Optimized librosa operations
- Memory-efficient processing
- Enhanced error handling
- Performance monitoring

### 📊 New Analysis Features
- Tempo stability analysis
- Harmonic-percussive ratio calculation
- Advanced onset detection
- Comprehensive statistical analysis
- Tonal centroid (Tonnetz) features
- MFCC coefficient analysis (20 coefficients)

### 🎯 Export & Reporting
- Professional markdown reports
- Comprehensive CSV data export
- Structured JSON format
- Feature summary statistics
- Analysis metadata inclusion
