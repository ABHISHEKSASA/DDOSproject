import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import time
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards

# Page configuration
st.set_page_config(
    page_title="DDOS AI",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blockchain theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6e48aa;
        --secondary: #9d50bb;
        --accent: #4776E6;
        --dark: #1a1a2e;
        --light: #f8f9fa;
    }
    
    /styling */
    .block-card {
        border-radius: 12px;
        border-left: 5px solid var(--primary);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .block-header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    .block-button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .feature-card {
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: var(--light);
        border-left: 3px solid var(--accent);
    }

</style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("cnn_feature_extractor (1).h5")

@st.cache_resource
def load_rf_model():
    return joblib.load("rf_model (1).pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

cnn_model = load_cnn_model()
rf_model = load_rf_model()
scaler = load_scaler()

# Generate traffic function
def generate_traffic():
    return np.array([[np.random.randint(0, 65535),
                     np.random.randint(100, 10000),
                     np.random.randint(50, 1500),
                     np.random.randint(50, 1500),
                     np.random.uniform(1000, 100000),
                     np.random.uniform(10, 1000),
                     np.random.uniform(1, 500)]],
                     dtype=np.float32)

# ========== MAIN APP ========== #

# Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/blockchain-new-logo.png", width=80)
st.sidebar.title("DDOS AI")

menu = st.sidebar.radio("", ["Home", "Why ML and DDOS", "How It Works"])

# Home Page
if menu == "Home":
    # Header section
    st.markdown("""
    <div class="block-header">
        <h1 style="color:white; margin-bottom:0.5rem;">Unleashing the Power of ML Security</h1>
        <p style="color:white; font-size:1.1rem;">Protecting networks with ML detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.button("Get Started with DDOS", type="primary")
    with col2:
        st.button("Discover How It Works")
    
    st.markdown("---")
    
    # Why Blockchain section
    st.markdown("""
    <div class="block-card">
        <h2>Why ML for Security?</h2>
        <p>MACHINE LEARNING is revolutionizing cybersecurity with Continuous monitoring protection against DDoS attacks.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Continuous Monitoring</h4>
            <p>No single point of failure in our detection network</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Security</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>Transparency</h4>
            <p>Publicly verifiable detection results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>Efficiency</h4>
            <p>Cost-effective distributed protection</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detection system
    st.markdown("---")
    st.markdown("""
    <div class="block-card">
        <h2>Real-Time DDoS Detection</h2>
        <p>Our Hybrid AI-blockchain system monitors your network 24/7</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state['running'] = False
    if 'normal_count' not in st.session_state:
        st.session_state['normal_count'] = 0
    if 'ddos_count' not in st.session_state:
        st.session_state['ddos_count'] = 0
    
    # Control buttons
    col1, col2 = st.columns([1,5])
    with col1:
        if st.button("▶ Start Monitoring", type="primary"):
            st.session_state['running'] = True
    with col2:
        if st.button("⏹ Stop Monitoring"):
            st.session_state['running'] = False
    
    if st.session_state['running']:
        # Stats cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Traffic", value=f"{st.session_state['normal_count'] + st.session_state['ddos_count']}", delta="live")
        with col2:
            st.metric(label="Normal Traffic", value=st.session_state['normal_count'])
        with col3:
            st.metric(label="DDoS Alerts", value=st.session_state['ddos_count'])
        
        style_metric_cards(border_left_color="#6e48aa")
        
        # Detection display
        try:
            while st.session_state['running']:
                input_data = generate_traffic()
                scaled_input = scaler.transform(input_data)
                
                if len(cnn_model.input_shape) == 3:
                    scaled_input = scaled_input.reshape(1, 7, 1)
                elif len(cnn_model.input_shape) == 2:
                    scaled_input = scaled_input.reshape(1, 7)
                
                cnn_features = cnn_model.predict(scaled_input, verbose=0)
                prediction = rf_model.predict(cnn_features)
                is_ddos = prediction[0] == 1
                
                if is_ddos:
                    st.session_state['ddos_count'] += 1
                    st.error("""
                    ## 🚨 DDoS ATTACK DETECTED!
                    *Threat detected in network traffic*  
                    Immediate action recommended to mitigate this attack.
                    """)
                else:
                    st.session_state['normal_count'] += 1
                    st.success("""
                    ## ✅ NORMAL TRAFFIC
                    Network activity appears within expected parameters.
                    """)
                
                time.sleep(1)
        except Exception as e:
            st.error(f"System error: {str(e)}")
            st.session_state['running'] = False

# Why Blockchain Page
elif menu == "Why Blockchain":
    st.markdown("""
    <div class="block-header">
        <h1 style="color:white;">Why ML Matters for Security</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="block-card">
        <p>Machine Learning is transforming the way we approach cybersecurity, threat detection, and network protection.
By leveraging intelligent algorithms and pattern recognition, our AI-based detection system offers real-time protection that's adaptive and highly accurate against evolving threats.</p>
        
        <h3>Key Benefits:</h3>
        <ul><ul>
    <li>Adaptive threat detection that evolves with emerging attack patterns</li>
    <li>Real-time analysis for immediate threat identification and response</li>
    <li>Accurate classification of network traffic for reliable alerts</li>
    <li>Automated decision-making to trigger rapid responses based on model predictions</li>
</ul>

        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://via.placeholder.com/1200x500?text=Blockchain+Security+Infographic", use_column_width=True)

# How It Works Page
elif menu == "How It Works":
    st.markdown("""
    <div class="block-header">
        <h1 style="color:white;">How DDOS AI Works</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="block-card">
        <h3>Our Hybrid Protection System</h3>
        <p>Artificial intelligence technology for unbeatable security:</p>
        
        <div style="display: flex; justify-content: space-between; margin: 2rem 0;">
            <div style="text-align: center; width: 30%;">
                <h4>1. Traffic Analysis</h4>
                <p>AI models monitor network patterns in real-time</p>
            </div>
            <div style="text-align: center; width: 30%;">
                <h4>2. Threat Detection</h4>
                <p>CNN-RF hybrid identifies attack signatures</p>
            </div>
            <div style="text-align: center; width: 30%;">
                <h4>3. Verification</h4>
                <p>Alerts are recorded on immutable ledger</p>
            </div>
        </div>
        
        <h3>Technical Architecture</h3>
        <p>Our system combines multiple layers of protection:</p>
        <ol>
            <li><strong>Data Layer:</strong> Collects network traffic metrics</li>
            <li><strong>AI Layer:</strong> Analyzes patterns using deep learning</li>
            <li><strong>Blockchain Layer:</strong> Secures detection results</li>
            <li><strong>Response Layer:</strong> Automates mitigation</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://via.placeholder.com/1200x600?text=System+Architecture+Diagram", use_column_width=True)
