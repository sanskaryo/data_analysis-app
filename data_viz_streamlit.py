import streamlit as st
import pandas as pd
from groq import Groq
import os
import numpy as np
from datetime import datetime
import io
import openpyxl
import plotly.express as px
import plotly.graph_objects as go
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Data Chat Assistant",
    page_icon="📊",
    layout="wide"
)

# Load environment variables
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    model_option = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
else:
    st.error("Please create a .env file with your GROQ_API_KEY and GROQ_MODEL")
    st.stop()

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

def load_data(file):
    """Load data from uploaded file."""
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        
        return data, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def convert_df_to_csv(df):
    """Convert DataFrame to CSV string."""
    return df.to_csv(index=False).encode('utf-8')

def display_and_download_plotly(fig, title):
    """Display Plotly figure and add download button."""
    try:
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Convert figure to bytes for download
        try:
            fig_bytes = fig.to_image(format="png")
            st.download_button(
                label="Download this visualization",
                data=fig_bytes,
                file_name=f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        except Exception as e:
            st.warning("Could not generate downloadable image. You can still use the interactive plot.")
    except Exception as e:
        st.error(f"Error displaying visualization: {str(e)}")

def analyze_visualization(df, viz_config):
    """Generate actionable insights from the visualization."""
    try:
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        # Prepare data context
        data_info = {
            "visualization_type": viz_config["visualization_type"],
            "columns": viz_config["columns"],
            "data_summary": {
                col: {
                    "mean": float(df[col].mean()) if df[col].dtype in ['int64', 'float64'] else None,
                    "median": float(df[col].median()) if df[col].dtype in ['int64', 'float64'] else None,
                    "min": float(df[col].min()) if df[col].dtype in ['int64', 'float64'] else None,
                    "max": float(df[col].max()) if df[col].dtype in ['int64', 'float64'] else None,
                    "unique_values": int(df[col].nunique()) if df[col].dtype in ['object', 'category'] else None,
                    "most_common": df[col].value_counts().head(3).to_dict() if df[col].dtype in ['object', 'category'] else None
                } for col in viz_config["columns"]
            }
        }
        
        # Create prompt for analysis
        analysis_prompt = f"""You are a data visualization expert who excels at presenting clear, impactful insights. Analyze this visualization and provide a concise, focused interpretation.

Visualization Information:
{data_info}

Please provide your analysis in this format:

Key Insight: [One clear, impactful sentence that captures the most important finding]

[One brief explanation of why this matters, if needed]



Keep your analysis:
1. Focused on the single most important insight
2. Concise and to the point
3. Free of technical jargon
4. Action-oriented"""

        # Get analysis from LLM
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": analysis_prompt}],
            model=model_option,
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def create_visualization(df, prompt):
    """Create visualization based on prompt using LLM for understanding."""
    try:
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        # Prepare data context
        data_info = {
            "columns": df.columns.tolist(),
            "dtypes": dict(zip(df.columns, df.dtypes.astype(str).tolist())),
            "sample": df.head(3).to_dict('records'),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        # Create prompt for LLM to understand visualization request
        viz_prompt = f"""You are a data visualization expert. Given the following data and user request, determine the appropriate visualization type and columns to use.

Data Information:
{data_info}

User Request: {prompt}

IMPORTANT: You must respond with ONLY a valid JSON object in the following format:
{{
    "visualization_type": "histogram|bar|scatter|box|pie|line|area",
    "columns": ["column1", "column2"],
    "title": "Appropriate title for the visualization",
    "x_label": "Label for x-axis",
    "y_label": "Label for y-axis"
}}

Rules:
1. The response must be a single valid JSON object
2. Do not include any text before or after the JSON
3. Do not include any explanations or comments
4. Make sure all strings are properly quoted
5. Choose the most appropriate visualization type based on the data and request
6. Select columns that make sense for the requested visualization
7. Create clear, descriptive titles and labels

Example response:
{{"visualization_type": "bar", "columns": ["department"], "title": "Employee Count by Department", "x_label": "Department", "y_label": "Count"}}"""

        # Get response from LLM
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": viz_prompt}],
            model=model_option,
            temperature=0.1,
            max_tokens=500
        )
        
        # Get the response content
        response_content = response.choices[0].message.content.strip()
        
        # Try to clean the response if it's not valid JSON
        try:
            # First attempt: direct JSON parsing
            viz_config = json.loads(response_content)
        except json.JSONDecodeError:
            try:
                # Second attempt: try to extract JSON from the response
                # Look for content between curly braces
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    viz_config = json.loads(json_match.group())
                else:
                    raise json.JSONDecodeError("No JSON object found in response", response_content, 0)
            except Exception as e:
                st.error(f"Error parsing LLM response: {str(e)}")
                st.error("Raw response: " + response_content)
                return None, None, None
        
        # Validate required fields
        required_fields = ["visualization_type", "columns", "title"]
        if not all(field in viz_config for field in required_fields):
            st.error("LLM response missing required fields")
            st.error("Response received: " + str(viz_config))
            return None, None, None
        
        # Validate column names
        for col in viz_config["columns"]:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in dataset")
                st.error("Available columns: " + ", ".join(df.columns))
                return None, None, None
        
        # Create visualization based on LLM's understanding
        if viz_config["visualization_type"] == "histogram":
            fig = px.histogram(df, x=viz_config["columns"][0], 
                             title=viz_config["title"])
            
        elif viz_config["visualization_type"] == "bar":
            value_counts = df[viz_config["columns"][0]].value_counts().reset_index()
            value_counts.columns = [viz_config["columns"][0], 'count']
            fig = px.bar(value_counts, 
                        x=viz_config["columns"][0], 
                        y='count',
                        title=viz_config["title"])
            
        elif viz_config["visualization_type"] == "scatter":
            fig = px.scatter(df, 
                           x=viz_config["columns"][0], 
                           y=viz_config["columns"][1],
                           title=viz_config["title"])
            
        elif viz_config["visualization_type"] == "box":
            fig = px.box(df, 
                        y=viz_config["columns"][0],
                        title=viz_config["title"])
            
        elif viz_config["visualization_type"] == "pie":
            value_counts = df[viz_config["columns"][0]].value_counts().reset_index()
            value_counts.columns = [viz_config["columns"][0], 'count']
            fig = px.pie(value_counts, 
                        names=viz_config["columns"][0], 
                        values='count',
                        title=viz_config["title"])
            
        elif viz_config["visualization_type"] == "line":
            fig = px.line(df, 
                         y=viz_config["columns"][0],
                         title=viz_config["title"])
            
        elif viz_config["visualization_type"] == "area":
            fig = px.area(df, 
                         y=viz_config["columns"][0],
                         title=viz_config["title"])
        
        else:
            st.error(f"Unsupported visualization type: {viz_config['visualization_type']}")
            return None, None, None
        
        # Update layout with LLM's suggestions
        fig.update_layout(
            xaxis_title=viz_config.get("x_label", ""),
            yaxis_title=viz_config.get("y_label", ""),
            showlegend=True
        )
        
        # Generate analysis
        analysis = analyze_visualization(df, viz_config)
        
        return fig, viz_config["visualization_type"], analysis
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None, None, None

def generate_dataset_summary(df):
    """Generate a comprehensive summary of the dataset."""
    summary = []
    
    # Basic info
    summary.append(f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Column types
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    
    if len(numeric_cols) > 0:
        summary.append(f"\nNumeric columns: {', '.join(numeric_cols)}")
    if len(categorical_cols) > 0:
        summary.append(f"\nCategorical/text columns: {', '.join(categorical_cols)}")
    
    # Missing values
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        summary.append(f"\nMissing values are present in {len(cols_with_missing)} columns:")
        for col, count in cols_with_missing.items():
            summary.append(f"- {col}: {count} missing values ({count/len(df)*100:.1f}%)")
    else:
        summary.append("\nNo missing values detected in the dataset.")
        
    # Basic statistics for numeric columns
    if len(numeric_cols) > 0:
        summary.append("\nKey statistics for numeric columns:")
        for col in numeric_cols:
            stats = df[col].describe()
            summary.append(f"\n{col}:")
            summary.append(f"- Mean: {stats['mean']:.2f}")
            summary.append(f"- Median: {stats['50%']:.2f}")
            summary.append(f"- Min: {stats['min']:.2f}")
            summary.append(f"- Max: {stats['max']:.2f}")
    
    # Unique values for categorical columns
    if len(categorical_cols) > 0:
        summary.append("\nUnique values in categorical columns:")
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 10:  # Only show if not too many unique values
                values = df[col].value_counts().head(5)
                summary.append(f"\n{col} ({unique_vals} unique values):")
                for val, count in values.items():
                    summary.append(f"- {val}: {count} occurrences")
            else:
                summary.append(f"\n{col}: {unique_vals} unique values")
    
    return "\n".join(summary)

def generate_example_prompts(df):
    """Generate example prompts based on the dataset."""
    prompts = []
    
    # Get column information
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Visualization prompts for numeric data
    if len(numeric_cols) > 0:
        prompts.extend([
            f"Show me the distribution of {numeric_cols[0]}",
            f"Visualize the range of {numeric_cols[0]}",
            f"Create a chart showing {numeric_cols[0]} trends",
            f"Plot the top 5 values of {numeric_cols[0]}",
            f"Show me a box plot of {numeric_cols[0]}"
        ])
    
    # Visualization prompts for categorical data
    if len(cat_cols) > 0:
        prompts.extend([
            f"Show me a bar chart of {cat_cols[0]}",
            f"Create a pie chart showing {cat_cols[0]} distribution",
            f"Visualize the most common {cat_cols[0]}",
            f"Show me the count of each {cat_cols[0]}",
            f"Plot the top categories in {cat_cols[0]}"
        ])
    
    # Cross-analysis visualization prompts
    if len(numeric_cols) > 0 and len(cat_cols) > 0:
        prompts.extend([
            f"Show me how {numeric_cols[0]} varies by {cat_cols[0]}",
            f"Create a chart comparing {numeric_cols[0]} across {cat_cols[0]}",
            f"Visualize the relationship between {cat_cols[0]} and {numeric_cols[0]}",
            f"Show me a grouped bar chart of {numeric_cols[0]} by {cat_cols[0]}"
        ])
    
    # General visualization prompts
    prompts.extend([
        "Show me the key trends in this data",
        "Create a summary visualization of the main patterns",
        "Visualize the most important insights",
        "Show me the top 5 findings in a chart",
        "Create a dashboard of key metrics"
    ])
    
    return prompts

# Main UI
st.title("Data Chat Assistant")

# File uploader in main area
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")
    
    # Load data
    data, error = load_data(uploaded_file)
    if error:
        st.error(error)
    else:
        # Store in session state
        st.session_state.data = data
        st.session_state.file_name = uploaded_file.name
        
        # Show data info
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        
        # Download options
        st.download_button(
            label="Download as CSV",
            data=convert_df_to_csv(data),
            file_name=f"processed_{uploaded_file.name.split('.')[0]}.csv",
            mime="text/csv"
        )

# Main area
if st.session_state.data is not None:
    # Data preview in expander
    with st.expander("Data Preview", expanded=False):
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
    
    # Chat interface
    st.subheader("Chat with Your Data")
    st.write("Ask questions about your data. The assistant will provide clear, easy-to-understand answers and visualizations when requested.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display visualization if present
            if "visualization" in message:
                viz_info = message["visualization"]
                display_and_download_plotly(viz_info["figure"], viz_info["title"])
            
            # Display dataframe if present
            if "dataframe" in message:
                st.dataframe(message["dataframe"], use_container_width=True)
                
                # Add download button for the dataframe
                st.download_button(
                    label="Download this table",
                    data=convert_df_to_csv(message["dataframe"]),
                    file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Chat input
    if prompt := st.chat_input("Ask about your data"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Check for visualization requests
                if any(keyword in prompt.lower() for keyword in 
                       ['show', 'plot', 'visualize', 'graph', 'chart', 'histogram', 
                        'bar chart', 'scatter', 'correlation', 'distribution', 'pie']):
                    # Visualization request: do NOT use spinner
                    # Create a container for the response
                    response_container = st.container()
                    # Try to create visualization directly
                    fig, chart_type, analysis = create_visualization(st.session_state.data, prompt)
                    if fig is not None:
                        with response_container:
                            # Success message
                            st.write(f"Here's the {chart_type} chart you requested:")
                            # Display visualization
                            st.plotly_chart(fig, use_container_width=True)
                            # Display analysis
                            if analysis:
                                st.write("**Key Insights:**")
                                st.write(analysis)
                            # Add download button for the plot
                            try:
                                fig_bytes = fig.to_image(format="png")
                                st.download_button(
                                    label="Download this visualization",
                                    data=fig_bytes,
                                    file_name=f"{chart_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png"
                                )
                            except Exception as e:
                                st.warning("Could not generate downloadable image. You can still use the interactive plot.")
                        # Add to message history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Here's the {chart_type} chart you requested:\n\n{analysis}",
                            "visualization": {
                                "type": "plotly",
                                "figure": fig,
                                "title": f"{chart_type.title()} Chart"
                            }
                        })
                    else:
                        # If visualization fails, use Groq
                        use_groq = True
                else:
                    # Not a visualization request, use spinner for LLM text/table answers
                    use_groq = True
                # Use Groq for non-visualization requests or failed visualizations
                if 'use_groq' in locals() and use_groq:
                    with st.spinner("Analyzing your data..."):
                        if not groq_api_key:
                            st.error("Please set up your Groq API key in the .env file")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "I need a Groq API key to answer that question. Please set up your API key in the .env file."
                            })
                        else:
                            # Initialize Groq client
                            client = Groq(api_key=groq_api_key)
                            # Prepare context about the data
                            data_info = f"Data columns: {', '.join(st.session_state.data.columns.tolist())}\n"
                            data_info += f"Data types: {dict(zip(st.session_state.data.columns, st.session_state.data.dtypes.astype(str).tolist()))}\n"
                            data_info += f"Sample data:\n{st.session_state.data.head(5).to_string()}\n"
                            data_info += f"Summary statistics:\n{st.session_state.data.describe().to_string()}"
                            # Get previous context from conversation history
                            previous_context = ""
                            if len(st.session_state.messages) > 1:
                                context_messages = st.session_state.messages[-min(4, len(st.session_state.messages)):]
                                for msg in context_messages:
                                    previous_context += f"{msg['role']}: {msg['content']}\n"
                            # Create the full prompt with context
                            full_prompt = f"""You are a data visualization expert who excels at presenting insights through clear visuals and concise explanations. Your goal is to help users understand their data primarily through charts and tables, with minimal but impactful text.

                            Data Information:
                            {data_info}
                            
                            Previous conversation:
                            {previous_context}
                            
                            User Question: {prompt}
                            
                            Important Instructions:
                            1. ALWAYS create a visualization or table if it can help explain the data
                            2. Keep text explanations to 2-3 sentences maximum
                            3. Focus on the most important insight only
                            4. Use tables for comparing values or showing rankings
                            5. Use charts for showing trends, distributions, or relationships
                            6. Format numbers to be readable (e.g., "1.2M" instead of "1,200,000")
                            7. Include just one follow-up question that could lead to deeper insights
                            8. If multiple visualizations would help, choose the most impactful one
                            
                            Example response format:
                            [One clear, concise insight about the data]
                            
                            [Visualization or table showing the key finding]
                            
                            Would you like to explore [one follow-up question]?
                            """
                            # Get response from Groq
                            response = client.chat.completions.create(
                                messages=[{"role": "user", "content": full_prompt}],
                                model=model_option,
                                temperature=0.2,
                                max_tokens=2048
                            )
                            answer = response.choices[0].message.content
                            # Display the response
                            st.markdown(answer)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer
                            })
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I'm sorry, I encountered an error while processing your request: {str(e)}"
                })
    
    # Example prompts based on the dataset
    with st.expander("Example Prompts"):
        example_prompts = generate_example_prompts(st.session_state.data)
        st.markdown("\n".join([f"- {prompt}" for prompt in example_prompts]))
else:
    # No file uploaded yet
    st.info("Please upload a CSV or Excel file to begin analysis.")
    
    st.markdown("""
    ## Welcome to Data Chat Assistant!
    
    This application helps you analyze your data through natural language conversations.
    
    **Features:**
    - 💬 **Chat with your data**: Ask questions and get instant insights
    - 📊 **Clear explanations**: Get easy-to-understand answers
    - 📈 **Visualizations**: Create charts and graphs when needed
    - 📂 **Support for CSV and Excel**: Analyze data from various sources
    - 💾 **Downloadable results**: Save your analyses for sharing
    
    **To get started:**
    1. Upload a CSV or Excel file
    2. Ask questions about your data in the chat
    3. Get clear, simple explanations
    4. Request visualizations when needed
    5. Download results for sharing
    """)
    
    # Add example datasets
    with st.expander("Try Example Datasets"):
        st.markdown("""
        Don't have a dataset handy? Try one of these:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Titanic Dataset")
            st.write("Famous dataset with passenger information from the Titanic")
            if st.button("Load Titanic Dataset"):
                with st.spinner("Loading example dataset..."):
                    # Load Titanic dataset
                    import requests
                    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                    s = requests.get(url).content
                    titanic_df = pd.read_csv(io.StringIO(s.decode('utf-8')))
                    
                    # Save to session state
                    st.session_state.data = titanic_df
                    st.session_state.file_name = "titanic.csv"
                    st.rerun()
        
        with col2:
            st.markdown("### Iris Dataset")
            st.write("Classic dataset with flower measurements for three iris species")
            if st.button("Load Iris Dataset"):
                with st.spinner("Loading example dataset..."):
                    # Load Iris dataset
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
                    iris_df['species'] = pd.Series(iris.target).map({
                        0: 'setosa', 
                        1: 'versicolor', 
                        2: 'virginica'
                    })
                    
                    # Save to session state
                    st.session_state.data = iris_df
                    st.session_state.file_name = "iris.csv"
                    st.rerun()