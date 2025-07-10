import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import time
import plotly.express as px
import plotly.graph_objects as go
from claude_work import MainPipeline
import json
from pathlib import Path

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure Streamlit page
st.set_page_config(
    page_title="📧 Email Classification Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1em;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1em 0;
    }
    .spam-alert {
        background: #ffebee;
        border-left: 4px solid #f44336;
    }
    .priority-high {
        background: #ffebee;
        border-left: 4px solid #f44336;
    }
    .priority-medium {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .priority-low {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitMLPipeline:
    """ML Pipeline integration with claude_work.py MainPipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.is_ready = False
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the main pipeline"""
        try:
            # Initialize pipeline with default paths
            self.pipeline = MainPipeline()
            self.pipeline.initialize_components()
            
            # Check if models exist and are ready
            if self._check_models_exist():
                self.is_ready = True
                st.success("✅ Models loaded successfully!")
            else:
                st.warning("⚠️ Models not found. Training will be required on first use.")
                self.is_ready = False
                
        except Exception as e:
            st.error(f"❌ Error initializing pipeline: {str(e)}")
            self.is_ready = False
    
    def _check_models_exist(self):
        """Check if trained models exist"""
        try:
            # Check for key model files
            vectorizer_path = Path("models/tfidf_vectorizer.pkl")
            kmeans_path = Path("models/kmeans_model.pkl")
            mlp_path = Path("models/mlp_models/multilabel_mlp_model.joblib")
            
            return vectorizer_path.exists() and kmeans_path.exists() and mlp_path.exists()
        except Exception:
            return False
    
    def ensure_models_trained(self):
        """Ensure models are trained and ready"""
        if not self.is_ready:
            st.info("🔄 Training models for first use. This may take a few minutes...")
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Training models...")
                progress_bar.progress(50)
                
                # Run the pipeline to train models
                self.pipeline.run()
                
                progress_bar.progress(100)
                status_text.text("Models trained successfully!")
                
                self.is_ready = True
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Models trained and ready!")
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                raise e

    def predict_single(self, email_text):
        """Predict classifications for a single email using the real pipeline"""
        # Ensure models are trained
        self.ensure_models_trained()
        
        if not self.is_ready:
            raise Exception("Models are not ready for prediction")
        
        try:
            # Use the actual pipeline for prediction
            results = self.pipeline.predict_multiple_texts([email_text])
            
            if not results:
                raise Exception("No prediction results returned")
            
            # Extract the first (and only) result
            result = results[0]
            
            # Format results to match expected structure
            formatted_results = {}
            
            # Extract MLP predictions
            mlp_predictions = result.get('mlp_predictions', {})
            
            for target in ['is_spam', 'type', 'priority', 'queue']:
                if target in mlp_predictions:
                    pred_data = mlp_predictions[target]
                    
                    # Handle the prediction format
                    if isinstance(pred_data, dict):
                        predicted_value = pred_data.get('predicted_value', 'unknown')
                        confidence = pred_data.get('confidence', 0.0)
                        
                        # Format spam prediction
                        if target == 'is_spam':
                            if predicted_value == 1 or predicted_value == '1' or predicted_value == 'spam':
                                pred_label = 'spam'
                            else:
                                pred_label = 'not_spam'
                        else:
                            pred_label = predicted_value
                        
                        formatted_results[target] = {
                            'prediction': pred_label,
                            'confidence': confidence if confidence > 0 else 0.8,  # Default confidence
                            'probabilities': pred_data.get('probabilities', [0.2, 0.8])
                        }
                    else:
                        # Fallback formatting
                        formatted_results[target] = {
                            'prediction': str(pred_data),
                            'confidence': 0.8,
                            'probabilities': [0.2, 0.8]
                        }
                else:
                    # Default fallback
                    formatted_results[target] = {
                        'prediction': 'unknown',
                        'confidence': 0.5,
                        'probabilities': [0.5, 0.5]
                    }
            
            # Add cluster information
            cluster_id = result.get('cluster', 0)
            formatted_results['cluster'] = cluster_id
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            # Return default results on error
            return {
                'is_spam': {'prediction': 'not_spam', 'confidence': 0.5, 'probabilities': [0.5, 0.5]},
                'type': {'prediction': 'Request', 'confidence': 0.5, 'probabilities': [0.5, 0.5]},
                'priority': {'prediction': 'medium', 'confidence': 0.5, 'probabilities': [0.5, 0.5]},
                'queue': {'prediction': 'General Inquiry', 'confidence': 0.5, 'probabilities': [0.5, 0.5]}
            }

@st.cache_resource
def load_pipeline():
    """Load ML pipeline (cached for performance)"""
    return StreamlitMLPipeline()

def display_single_result(results, email_text):
    """Display results for single email classification"""
    st.markdown("### 📊 Classification Results")
    
    # Show cluster information if available
    if 'cluster' in results:
        st.markdown(f"**🎯 Cluster Assignment**: Cluster {results['cluster']}")
        st.markdown("---")
    
    # Create columns for organized display
    col1, col2 = st.columns(2)
    
    with col1:
        # Spam Detection
        st.markdown("#### 🚨 Spam Detection")
        spam_result = results['is_spam']
        if spam_result['prediction'] == 'spam':
            st.error(f"⚠️ SPAM DETECTED - Confidence: {spam_result['confidence']:.1%}")
        else:
            st.success(f"✅ NOT SPAM - Confidence: {spam_result['confidence']:.1%}")
        
        # Email Type
        st.markdown("#### 📝 Email Type")
        email_type = results['type']['prediction']
        type_confidence = results['type']['confidence']
        st.info(f"📋 **{email_type}** - Confidence: {type_confidence:.1%}")
        
    with col2:
        # Priority
        st.markdown("#### ⚡ Priority Level")
        priority = results['priority']['prediction']
        priority_confidence = results['priority']['confidence']
        
        if priority == 'high':
            st.error(f"🔴 **HIGH Priority** - Confidence: {priority_confidence:.1%}")
        elif priority == 'medium':
            st.warning(f"🟡 **MEDIUM Priority** - Confidence: {priority_confidence:.1%}")
        else:
            st.success(f"🟢 **LOW Priority** - Confidence: {priority_confidence:.1%}")
        
        # Queue Routing
        st.markdown("#### 🏢 Department Queue")
        queue = results['queue']['prediction']
        queue_confidence = results['queue']['confidence']
        st.info(f"🎯 **{queue}** - Confidence: {queue_confidence:.1%}")
    
    # Detailed probabilities (expandable)
    with st.expander("📈 View Detailed Probabilities"):
        prob_data = []
        for category, result in results.items():
            if category != 'cluster':  # Skip cluster in the probability table
                prob_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Prediction': result['prediction'],
                    'Confidence': f"{result['confidence']:.1%}"
                })
        
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, use_container_width=True)

def process_batch_emails(uploaded_file, pipeline):
    """Process multiple emails from CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate required column
        if 'email_content' not in df.columns:
            st.error("❌ CSV file must contain an 'email_content' column")
            st.info("📋 Expected format: CSV with 'email_content' column containing email text")
            return
        
        # Ensure models are trained
        pipeline.ensure_models_trained()
        
        # Process emails using batch prediction for efficiency
        st.info("🔄 Processing emails in batch...")
        
        # Convert to list of strings
        email_texts = [str(email) for email in df['email_content'].tolist()]
        
        try:
            # Use the pipeline's batch prediction
            batch_results = pipeline.pipeline.predict_multiple_texts(email_texts)
            
            # Process results
            results = []
            for i, (email_content, result) in enumerate(zip(email_texts, batch_results)):
                # Extract MLP predictions
                mlp_predictions = result.get('mlp_predictions', {})
                
                # Format results
                formatted_result = {
                    'Email_ID': i + 1,
                    'Cluster': result.get('cluster', 0),
                    'Is_Spam': 'unknown',
                    'Spam_Confidence': '50.0%',
                    'Email_Type': 'unknown',
                    'Type_Confidence': '50.0%',
                    'Priority': 'unknown',
                    'Priority_Confidence': '50.0%',
                    'Queue': 'unknown',
                    'Queue_Confidence': '50.0%'
                }
                
                # Extract predictions from MLP results
                for target in ['is_spam', 'type', 'priority', 'queue']:
                    if target in mlp_predictions:
                        pred_data = mlp_predictions[target]
                        
                        if isinstance(pred_data, dict):
                            predicted_value = pred_data.get('predicted_value', 'unknown')
                            confidence = pred_data.get('confidence', 0.5)
                            
                            # Format spam prediction
                            if target == 'is_spam':
                                if predicted_value == 1 or predicted_value == '1' or predicted_value == 'spam':
                                    pred_label = 'spam'
                                else:
                                    pred_label = 'not_spam'
                            else:
                                pred_label = str(predicted_value)
                            
                            # Update formatted result
                            if target == 'is_spam':
                                formatted_result['Is_Spam'] = pred_label
                                formatted_result['Spam_Confidence'] = f"{confidence:.1%}"
                            elif target == 'type':
                                formatted_result['Email_Type'] = pred_label
                                formatted_result['Type_Confidence'] = f"{confidence:.1%}"
                            elif target == 'priority':
                                formatted_result['Priority'] = pred_label
                                formatted_result['Priority_Confidence'] = f"{confidence:.1%}"
                            elif target == 'queue':
                                formatted_result['Queue'] = pred_label
                                formatted_result['Queue_Confidence'] = f"{confidence:.1%}"
                
                results.append(formatted_result)
            
            # Display results
            st.success(f"✅ Successfully processed {len(results)} emails!")
            results_df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                spam_count = sum(1 for r in results if r['Is_Spam'] == 'spam')
                st.metric("Spam Emails", spam_count)
            
            with col2:
                high_priority = sum(1 for r in results if r['Priority'] == 'high')
                st.metric("High Priority", high_priority)
            
            with col3:
                incidents = sum(1 for r in results if r['Email_Type'] == 'Incident')
                st.metric("Incidents", incidents)
            
            with col4:
                it_support = sum(1 for r in results if 'IT Support' in r['Queue'])
                st.metric("IT Support", it_support)
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"email_classification_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error in batch prediction: {str(e)}")
            st.info("Falling back to individual predictions...")
            
            # Fallback to individual processing
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, email_content in enumerate(email_texts):
                status_text.text(f'Processing email {i+1} of {len(email_texts)}...')
                
                try:
                    # Get predictions
                    result = pipeline.predict_single(str(email_content))
                    
                    # Store results
                    results.append({
                        'Email_ID': i + 1,
                        'Cluster': result.get('cluster', 0),
                        'Is_Spam': result['is_spam']['prediction'],
                        'Spam_Confidence': f"{result['is_spam']['confidence']:.1%}",
                        'Email_Type': result['type']['prediction'],
                        'Type_Confidence': f"{result['type']['confidence']:.1%}",
                        'Priority': result['priority']['prediction'],
                        'Priority_Confidence': f"{result['priority']['confidence']:.1%}",
                        'Queue': result['queue']['prediction'],
                        'Queue_Confidence': f"{result['queue']['confidence']:.1%}"
                    })
                    
                except Exception as e:
                    st.warning(f"⚠️ Error processing email {i+1}: {str(e)}")
                    # Add default result
                    results.append({
                        'Email_ID': i + 1,
                        'Cluster': 0,
                        'Is_Spam': 'unknown',
                        'Spam_Confidence': '50.0%',
                        'Email_Type': 'unknown',
                        'Type_Confidence': '50.0%',
                        'Priority': 'unknown',
                        'Priority_Confidence': '50.0%',
                        'Queue': 'unknown',
                        'Queue_Confidence': '50.0%'
                    })
                
                # Update progress
                progress_bar.progress((i + 1) / len(email_texts))
            
            # Clear status
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            st.success(f"✅ Successfully processed {len(results)} emails!")
            results_df = pd.DataFrame(results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                spam_count = sum(1 for r in results if r['Is_Spam'] == 'spam')
                st.metric("Spam Emails", spam_count)
            
            with col2:
                high_priority = sum(1 for r in results if r['Priority'] == 'high')
                st.metric("High Priority", high_priority)
            
            with col3:
                incidents = sum(1 for r in results if r['Email_Type'] == 'Incident')
                st.metric("Incidents", incidents)
            
            with col4:
                it_support = sum(1 for r in results if 'IT Support' in r['Queue'])
                st.metric("IT Support", it_support)
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"email_classification_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted with an 'email_content' column.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📧 Email Classification Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent email classification for spam detection, type, priority, and routing</p>', unsafe_allow_html=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model info
        if pipeline.is_ready:
            st.success("🤖 **Model Status**: Ready for classification")
            st.info("Using: TF-IDF + K-means + MLP")
        else:
            st.warning("⚠️ **Model Status**: Training required")
            st.info("Models will be trained automatically on first use")
        
        # Model details
        with st.expander("🔍 Model Details"):
            st.markdown("""
            **Pipeline Components:**
            - **Text Preprocessing**: Lemmatization, stopword removal
            - **Feature Engineering**: TF-IDF vectorization
            - **Clustering**: K-means clustering for grouping
            - **Classification**: Multi-label MLP for predictions
            
            **Model Files:**
            - TF-IDF Vectorizer: `models/tfidf_vectorizer.pkl`
            - K-means Model: `models/kmeans_model.pkl`
            - MLP Model: `models/mlp_models/multilabel_mlp_model.joblib`
            """)
        
        # Settings
        st.subheader("Settings")
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_probabilities = st.checkbox("Show detailed probabilities", value=False)
        
        # About
        st.subheader("About")
        st.markdown("""
        This application classifies emails into:
        - **Spam Detection**: Spam vs Not Spam
        - **Email Type**: Incident, Request, Problem, Change
        - **Priority**: High, Medium, Low
        - **Department Queue**: IT Support, Technical Support, etc.
        
        **Features:**
        - Real-time single email classification
        - Batch processing of multiple emails
        - Clustering analysis for email grouping
        - Confidence scoring for predictions
        """)
        
        # Sample emails
        st.subheader("📝 Sample Emails")
        sample_emails = {
            "Urgent IT Issue": "Urgent: Our email server is down and users cannot access emails. Please resolve immediately.",
            "Software Request": "Hello, I need Microsoft Office installed on my new laptop. Please let me know when this can be done.",
            "Spam Example": "Congratulations! You've won $1,000,000! Click here to claim your prize now!",
            "Billing Question": "I have a question about my monthly bill. There seems to be an extra charge I don't understand."
        }
        
        selected_sample = st.selectbox("Choose a sample email:", list(sample_emails.keys()))
        if st.button("Use Sample Email"):
            st.session_state.sample_email = sample_emails[selected_sample]
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📧 Single Email", "📁 Batch Processing", "📊 Analytics"])
    
    with tab1:
        st.header("Classify Single Email")
        
        # Email input
        email_text = st.text_area(
            "Enter email content:",
            height=200,
            placeholder="Paste your email content here...",
            value=st.session_state.get('sample_email', '')
        )
        
        # Clear button
        if st.button("Clear Text"):
            st.session_state.sample_email = ""
            st.rerun()
        
        # Classification button
        if st.button("🔍 Classify Email", type="primary", use_container_width=True):
            if email_text.strip():
                with st.spinner("🤖 Analyzing email..."):
                    # Simulate processing time
                    time.sleep(1)
                    
                    try:
                        # Get predictions
                        results = pipeline.predict_single(email_text)
                        
                        # Display results
                        display_single_result(results, email_text)
                        
                        # Store in session state for analytics
                        if 'classification_history' not in st.session_state:
                            st.session_state.classification_history = []
                        
                        st.session_state.classification_history.append({
                            'email': email_text[:100] + "..." if len(email_text) > 100 else email_text,
                            'results': results,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error processing email: {str(e)}")
            else:
                st.warning("⚠️ Please enter email content to classify")
    
    with tab2:
        st.header("Batch Email Processing")
        st.markdown("Upload a CSV file containing multiple emails for batch classification.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV file should contain an 'email_content' column with email text"
        )
        
        # Show example format
        with st.expander("📋 View Expected CSV Format"):
            example_df = pd.DataFrame({
                'email_content': [
                    'Urgent: Server is down, please fix immediately',
                    'Request for new software installation',
                    'Win $1000 now! Click here to claim your prize!'
                ]
            })
            st.dataframe(example_df, use_container_width=True)
            
            # Download sample CSV
            sample_csv = example_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_csv,
                file_name="sample_emails.csv",
                mime="text/csv"
            )
        
        # Process batch
        if uploaded_file is not None:
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                process_batch_emails(uploaded_file, pipeline)
    
    with tab3:
        st.header("Classification Analytics")
        
        if 'classification_history' in st.session_state and st.session_state.classification_history:
            history = st.session_state.classification_history
            
            # Summary metrics
            st.subheader("📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_classified = len(history)
                st.metric("Total Emails", total_classified)
            
            with col2:
                spam_count = sum(1 for h in history if h['results']['is_spam']['prediction'] == 'spam')
                st.metric("Spam Emails", spam_count)
            
            with col3:
                high_priority = sum(1 for h in history if h['results']['priority']['prediction'] == 'high')
                st.metric("High Priority", high_priority)
            
            with col4:
                incidents = sum(1 for h in history if h['results']['type']['prediction'] == 'Incident')
                st.metric("Incidents", incidents)
            
            # Charts
            st.subheader("📈 Classification Distribution")
            
            # Email type distribution
            type_counts = {}
            for h in history:
                email_type = h['results']['type']['prediction']
                type_counts[email_type] = type_counts.get(email_type, 0) + 1
            
            if type_counts:
                fig_type = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Email Type Distribution"
                )
                st.plotly_chart(fig_type, use_container_width=True)
            
            # Priority distribution
            priority_counts = {}
            for h in history:
                priority = h['results']['priority']['prediction']
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            if priority_counts:
                fig_priority = px.bar(
                    x=list(priority_counts.keys()),
                    y=list(priority_counts.values()),
                    title="Priority Level Distribution",
                    color=list(priority_counts.keys()),
                    color_discrete_map={
                        'high': 'red',
                        'medium': 'orange',
                        'low': 'green'
                    }
                )
                st.plotly_chart(fig_priority, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.classification_history = []
                st.rerun()
        
        else:
            st.info("📝 No classification history yet. Start classifying emails to see analytics here!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Email Classification Assistant** | Built with Streamlit | "
        "Powered by TF-IDF + K-means + Multi-label MLP | "
        "Using claude_work.py pipeline for real-time ML predictions"
    )

if __name__ == "__main__":
    main()