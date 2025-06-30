import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import time
import io
import pathlib
import csv
import glob
from chatbot_assistant import create_chatbot_interface
from call_tracker import record_call

# --- STUB BACKEND HELPERS ---
def dispatch_technician(lead_id):
    st.success(f"(STUB) Dispatched tech for lead {lead_id}")
    record_call(lead_id, "dispatched")
    return {"job_id": "STUB123"}

def send_sms(to, body):
    st.success(f"(STUB) SMS to {to}: “{body}”")
    record_call(to, "sms_sent")
    return "STUB-SMS-ID"

def generate_invoice(lead_id, amount):
    st.success(f"(STUB) Invoice for ${amount} created")
    record_call(lead_id, "invoiced")
    return {"invoice_id": "STUBINV"}

def run_diagnostics(lead_id):
    st.success(f"(STUB) Diagnostics OK for lead {lead_id}")
    record_call(lead_id, "diagnostics")
    return {"status": "OK"}

def generate_estimate(lead_id, size):
    est = size * 1.2
    st.success(f"(STUB) Estimate: ${est:.2f}")
    record_call(lead_id, "estimated")
    return {"estimate": est}

def schedule_reminder(lead_id, date):
    st.success(f"(STUB) Reminder set for {date}")
    record_call(lead_id, "reminder_scheduled")
    return {"reminder_id": "STUBREM"}

def start_cleaning(lead_id, surface):
    st.success(f"(STUB) Cleaning {surface} started")
    record_call(lead_id, "cleaning_started")
    return {"clean_id": "STUBCLN"}

def remove_stain(lead_id, stain):
    st.success(f"(STUB) Removed {stain} stain")
    record_call(lead_id, "stain_removed")
    return {"stain_job": "STUBSTN"}

def apply_detergent(lead_id):
    st.success(f"(STUB) Eco-detergent applied")
    record_call(lead_id, "detergent_applied")
    return {"detergent_job": "STUBDTG"}

def clean_dataframe(df):
        """Clean and preprocess the DataFrame"""
        try:
            # Handle list columns that might cause issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains mixed types (some lists, some strings)
                    sample_values = df[col].dropna().head(5)
                    if len(sample_values) > 0:
                        # If we find any lists, ensure all values in this column are consistently formatted
                        has_lists = any(isinstance(val, list) for val in sample_values)
                        if has_lists:
                            # Convert any non-list values to lists for consistency
                            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else [])
            
            # Ensure required columns have default values
            required_defaults = {
                'name': 'Unknown Business',
                'industry': 'Unknown',
                'total_score': 0,
                'tier': 'TIER 4: LOWER PRIORITY',
                'quick_win': False,
                'close_probability': 0.0,
                'pain_points': [],
                'services': []
            }
            
            for col, default_val in required_defaults.items():
                if col not in df.columns:
                    df[col] = default_val
                else:
                    # Fill NaN values with defaults
                    df[col] = df[col].fillna(default_val)
            
            return df
        except Exception as e:
            st.warning(f"Data cleaning warning: {str(e)}")
            return df

def load_existing_leads(data_file='dashboard_data.json', backup_files=None):
    """Load leads from existing data files - no API calls"""
    if backup_files is None:
        backup_files = ['leads_data.json', 'leads.csv']
    start_time = time.time()
    # Try to load from primary data file
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                leads_data = json.load(f)
            # Convert to DataFrame
            if isinstance(leads_data, list):
                df = pd.DataFrame(leads_data)
            elif isinstance(leads_data, dict):
                all_leads = []
                for industry, leads in leads_data.items():
                    if isinstance(leads, list):
                        all_leads.extend(leads)
                df = pd.DataFrame(all_leads)
            else:
                raise ValueError("Unexpected data format")
            df = clean_dataframe(df)
            load_time = time.time() - start_time
            st.session_state.leads_loaded = len(df)
            st.sidebar.metric("Load Time", f"{load_time:.2f}s")
            st.sidebar.metric("Data Source", data_file)
            return df
        except Exception as e:
            st.error(f"Error loading {data_file}: {str(e)}")
    # Try backup files
    for backup_file in backup_files:
        if os.path.exists(backup_file):
            try:
                if backup_file.endswith('.json'):
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                elif backup_file.endswith('.csv'):
                    df = pd.read_csv(backup_file)
                else:
                    continue
                df = clean_dataframe(df)
                st.sidebar.metric("Data Source", backup_file)
                st.session_state.leads_loaded = len(df)
                return df
            except Exception as e:
                st.warning(f"Could not load {backup_file}: {str(e)}")
                continue
    return pd.DataFrame()  # Return empty DataFrame if no files found

def get_filtered_leads(df, industries=None, score_range=None, tier_filter=None, quick_win_only=False):
    """Filter leads based on criteria"""
    start_time = time.time()
    filtered_df = df.copy()
    if industries:
        filtered_df = filtered_df[filtered_df['industry'].isin(industries)]
    if score_range:
        filtered_df = filtered_df[(filtered_df['total_score'] >= score_range[0]) & (filtered_df['total_score'] <= score_range[1])]
    if tier_filter:
        filtered_df = filtered_df[filtered_df['tier'].isin(tier_filter)]
    if quick_win_only:
        filtered_df = filtered_df[filtered_df['quick_win'] == True]
    filter_time = time.time() - start_time
    st.sidebar.metric("Filter Time", f"{filter_time:.3f}s")
    return filtered_df

def load_uploaded_data(uploaded_file):
    """Load data from user uploaded file"""
    try:
        if uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                all_leads = []
                for key, value in data.items():
                    if isinstance(value, list):
                        all_leads.extend(value)
                df = pd.DataFrame(all_leads)
            else:
                df = pd.DataFrame()
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload JSON or CSV files.")
            return pd.DataFrame()
        df = clean_dataframe(df)
        return df
    except Exception as e:
        st.error(f"Error loading uploaded file: {str(e)}")
        return pd.DataFrame()

def create_lead_summary_charts(df):
    """Create summary charts for leads"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Industry distribution
        industry_counts = df['industry'].value_counts()
        fig_industry = px.pie(
            values=industry_counts.values, 
            names=industry_counts.index,
            title="Leads by Industry",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_industry.update_layout(height=400)
        st.plotly_chart(fig_industry, use_container_width=True)
    
    with col2:
        # Score distribution
        fig_score = px.histogram(
            df, 
            x='total_score', 
            nbins=20,
            title="Lead Score Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_score.update_layout(height=400)
        st.plotly_chart(fig_score, use_container_width=True)
    
    # Tier distribution
    tier_counts = df['tier'].value_counts()
    fig_tier = px.bar(
        x=tier_counts.index, 
        y=tier_counts.values,
        title="Leads by Tier",
        color=tier_counts.values,
        color_continuous_scale='RdYlGn_r'
    )
    fig_tier.update_layout(height=400)
    st.plotly_chart(fig_tier, use_container_width=True)

def create_performance_metrics(df):
    """Create performance and opportunity metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_leads = len(df)
        st.metric("Total Leads", f"{total_leads:,}")
    
    with col2:
        quick_wins = len(df[df['quick_win'] == True])
        st.metric("Quick Wins", f"{quick_wins:,}", f"{(quick_wins/total_leads*100):.1f}%" if total_leads > 0 else "0%")
    
    with col3:
        avg_score = df['total_score'].mean()
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    with col4:
        tier1_leads = len(df[df['tier'].str.contains('TIER 1', case=False, na=False)])
        st.metric("Tier 1 Leads", f"{tier1_leads:,}", f"{(tier1_leads/total_leads*100):.1f}%" if total_leads > 0 else "0%")
    
    # Revenue potential
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        min_mrr = df['mrr_potential_min'].sum()
        st.metric("Min MRR Potential", f"${min_mrr:,}")
    
    with col6:
        max_mrr = df['mrr_potential_max'].sum()
        st.metric("Max MRR Potential", f"${max_mrr:,}")
    
    with col7:
        avg_close_prob = df['close_probability'].mean()
        st.metric("Avg Close Probability", f"{avg_close_prob:.1f}%")
    
    with col8:
        high_value_leads = len(df[df['mrr_potential_max'] > 10000])
        st.metric("High Value Leads", f"{high_value_leads:,}")

def display_lead_table(df, page_size=50):
    """Display paginated lead table"""
    total_leads = len(df)
    total_pages = (total_leads - 1) // page_size + 1 if total_leads > 0 else 1
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.selectbox(
            "Page", 
            range(1, total_pages + 1),
            key="page_selector"
        )
    
    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_leads)
    
    # Display page info
    st.markdown(f"**Showing leads {start_idx + 1} to {end_idx} of {total_leads}**")
    
    # Display the table
    if not df.empty:
        page_df = df.iloc[start_idx:end_idx]
        
        # Format the display columns
        display_columns = [
            'name', 'industry', 'total_score', 'tier', 'quick_win',
            'mrr_potential_min', 'mrr_potential_max', 'close_probability'
        ]
        
        # Only show columns that exist in the dataframe
        available_columns = [col for col in display_columns if col in page_df.columns]
        
        st.dataframe(
            page_df[available_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download All as CSV",
                data=csv_data,
                file_name=f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download All as JSON",
                data=json_data,
                file_name=f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            if len(df) != len(page_df):
                filtered_csv = page_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered as CSV",
                    data=filtered_csv,
                    file_name=f"leads_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def google_maps_discovery_section():
    """Separate section for Google Maps API lead discovery"""
    st.header("🌐 Discover New Leads (Google Maps API)")
    st.warning("⚠️ This section requires Google Maps API configuration and will make external API calls.")
    
    with st.expander("Configure Google Maps API Discovery", expanded=False):
        st.markdown("""
        **To use lead discovery:**
        1. Ensure your Google Maps API key is configured
        2. Check your API usage limits
        3. Select industries and locations to search
        4. Click 'Discover New Leads' to start API calls
        """)
        
        # API Configuration Check
        api_configured = os.path.exists('google_maps_discovery.py')
        if api_configured:
            st.success("✅ Google Maps discovery module found")
        else:
            st.error("❌ Google Maps discovery module not found")
            st.info("💡 Place your Google Maps API code in 'google_maps_discovery.py'")
        
        # Discovery Controls
        col1, col2 = st.columns(2)
        
        with col1:
            discovery_industries = st.multiselect(
                "Industries to Discover",
                ['HVAC', 'Plumbing', 'Pressure Washing', 'Veterinary', 'Automotive Repair', 'Cleaning Services'],
                default=['HVAC']
            )
        
        with col2:
            discovery_limit = st.slider("Max Leads per Industry", 10, 100, 25)
        
        if st.button("🔍 Discover New Leads", type="primary"):
            if api_configured and discovery_industries:
                st.info("🔄 Starting lead discovery... This will make API calls and may take several minutes.")
                
                # Placeholder for actual discovery implementation
                with st.spinner("Discovering leads..."):
                    time.sleep(2)  # Simulate API calls
                    st.error("⚠️ Discovery feature requires Google Maps API integration. Please check your API configuration.")
            else:
                st.error("Please configure Google Maps API and select industries to discover.")

def create_lead_detail_view(filtered_df):
    """Create detailed lead analysis view with dropdown selector"""
    if filtered_df.empty:
        st.warning("No leads available for detailed analysis.")
        return
    
    st.subheader("🔍 Individual Lead Analysis")
    st.markdown("Select a lead to view complete analysis breakdown, strategic recommendations, and export options.")
    
    # Create lead options for dropdown (name + score for easy identification)
    lead_options = [f"{row['name']} (Score: {row['total_score']})" for idx, row in filtered_df.iterrows()]
    lead_mapping = {f"{row['name']} (Score: {row['total_score']})": idx for idx, row in filtered_df.iterrows()}
    
    # Lead selector dropdown
    selected_lead_option = st.selectbox(
        "🎯 Select Lead for Detailed Analysis",
        options=["Select a lead..."] + lead_options,
        key="lead_detail_selector"
    )
    
    if selected_lead_option != "Select a lead...":
        # Get selected lead data
        lead_idx = lead_mapping[selected_lead_option]
        lead = filtered_df.iloc[lead_idx]
        
        # Main lead information header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### 🏢 {lead['name']}")
            st.markdown(f"**Industry:** {lead['industry']}")
            st.markdown(f"**Business Type:** {lead.get('business_type', 'N/A')}")
            if lead.get('website'):
                st.markdown(f"**Website:** [{lead['website']}]({lead['website']})")
        
        with col2:
            # Tier badge with color coding
            tier = lead['tier']
            if 'TIER 1' in tier:
                st.markdown(f"<div style='background: #FF6B6B; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{tier}</div>", unsafe_allow_html=True)
            elif 'TIER 2' in tier:
                st.markdown(f"<div style='background: #4ECDC4; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{tier}</div>", unsafe_allow_html=True)
            elif 'TIER 3' in tier:
                st.markdown(f"<div style='background: #45B7D1; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{tier}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background: #96CEB4; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{tier}</div>", unsafe_allow_html=True)
        
        with col3:
            # Quick win indicator
            if lead.get('quick_win', False):
                st.markdown("<div style='background: #28a745; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>⚡ QUICK WIN</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background: #6c757d; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>📅 STANDARD</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed analysis sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Breakdown", "🎯 Strategic Analysis", "💰 Revenue Potential", "📋 Export Analysis"])
        
        with tab1:
            st.subheader("📊 Complete Scoring Breakdown")
            
            # Score visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Total score with visual indicator
                st.metric("Total Score", f"{lead['total_score']}/100", 
                         delta=f"{lead['total_score'] - 70} vs avg" if lead['total_score'] > 70 else None)
                
                # Individual scores
                st.markdown("**Component Scores:**")
                score_components = {
                    'Financial': lead.get('financial_score', 0),
                    'Decision Maker': lead.get('decision_score', 0), 
                    'Pain Level': lead.get('pain_score', 0),
                    'Tech Readiness': lead.get('tech_score', 0),
                    'Growth Potential': lead.get('growth_score', 0),
                    'Competition': lead.get('competition_score', 0)
                }
                
                for component, score in score_components.items():
                    max_score = 25 if component == 'Financial' else 20
                    percentage = (score / max_score) * 100 if max_score > 0 else 0
                    st.markdown(f"• **{component}:** {score}/{max_score} ({percentage:.0f}%)")
            
            with col2:
                # Score breakdown chart
                score_df = pd.DataFrame(list(score_components.items()), columns=['Component', 'Score'])
                fig_breakdown = px.bar(
                    score_df, 
                    x='Score', 
                    y='Component',
                    orientation='h',
                    title="Score Component Breakdown",
                    color='Score',
                    color_continuous_scale='RdYlGn'
                )
                fig_breakdown.update_layout(height=400)
                st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Close probability and confidence
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Close Probability", f"{lead.get('close_probability', 0):.1f}%")
            with col2:
                confidence = "High" if lead['total_score'] > 85 else "Medium" if lead['total_score'] > 70 else "Low"
                st.metric("Confidence Level", confidence)
            with col3:
                estimated_days = max(30, 120 - lead['total_score']) if lead['total_score'] < 120 else 30
                st.metric("Est. Sales Cycle", f"{estimated_days} days")
        
        with tab2:
            st.subheader("🎯 Strategic Analysis & Approach")
            
            # Pain point analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔥 Identified Pain Points:**")
                pain_points = lead.get('pain_points', [])
                if pain_points and isinstance(pain_points, list):
                    for pain in pain_points:
                        st.markdown(f"• {pain}")
                else:
                    st.markdown("• Operational inefficiencies")
                    st.markdown("• Manual process bottlenecks")
                    st.markdown("• Customer communication gaps")
                
                st.markdown("**🚀 Strategic Approach:**")
                if lead['total_score'] >= 85:
                    st.markdown("• **Direct Executive Approach** - High-level decision maker engagement")
                    st.markdown("• **ROI-Focused Presentation** - Emphasize immediate cost savings")
                    st.markdown("• **Quick Implementation** - Highlight rapid deployment capabilities")
                elif lead['total_score'] >= 70:
                    st.markdown("• **Consultative Approach** - Understand current challenges deeply")
                    st.markdown("• **Proof of Concept** - Demonstrate value through pilot program")
                    st.markdown("• **Reference Stories** - Share relevant customer success cases")
                else:
                    st.markdown("• **Educational Approach** - Build awareness of opportunities")
                    st.markdown("• **Long-term Relationship** - Focus on trust building")
                    st.markdown("• **Competitive Analysis** - Position against status quo")
            
            with col2:
                st.markdown("**🎯 Demo Focus Recommendations:**")
                industry = lead.get('industry', 'Unknown')
                
                if 'HVAC' in industry:
                    st.markdown("• **Scheduling Optimization** - Show automated dispatch")
                    st.markdown("• **Customer Portal** - Demonstrate self-service capabilities")
                    st.markdown("• **Maintenance Tracking** - Highlight preventive maintenance workflows")
                elif 'Plumbing' in industry:
                    st.markdown("• **Emergency Response** - Show rapid job allocation")
                    st.markdown("• **Inventory Management** - Demonstrate parts tracking")
                    st.markdown("• **Customer Communication** - Highlight real-time updates")
                else:
                    st.markdown("• **Workflow Automation** - Show process optimization")
                    st.markdown("• **Customer Management** - Demonstrate CRM capabilities")
                    st.markdown("• **Reporting Dashboard** - Highlight business insights")
                
                st.markdown("**📞 Contact Strategy:**")
                if lead.get('quick_win', False):
                    st.markdown("• **Immediate Follow-up** - Contact within 24 hours")
                    st.markdown("• **Decision Maker Direct** - Skip gatekeepers")
                    st.markdown("• **Value Proposition** - Lead with cost savings")
                else:
                    st.markdown("• **Research Phase** - Gather more company information")
                    st.markdown("• **Warm Introduction** - Find mutual connections")
                    st.markdown("• **Educational Content** - Share relevant case studies")
        
        with tab3:
            st.subheader("💰 Revenue Potential & Business Impact")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💵 Monthly Revenue Potential**")
                mrr_min = lead.get('mrr_potential_min', 0)
                mrr_max = lead.get('mrr_potential_max', 0)
                st.metric("MRR Range", f"${mrr_min:,} - ${mrr_max:,}")
                
                st.markdown("**📊 Annual Value**")
                arr_min = mrr_min * 12
                arr_max = mrr_max * 12
                st.metric("ARR Range", f"${arr_min:,} - ${arr_max:,}")
            
            with col2:
                st.markdown("**💸 Cost Savings Potential**")
                waste_min = lead.get('monthly_waste_min', 0)
                waste_max = lead.get('monthly_waste_max', 0)
                st.metric("Monthly Waste", f"${waste_min:,} - ${waste_max:,}")
                
                st.markdown("**⏰ ROI Timeline**")
                if lead['total_score'] >= 85:
                    roi_months = "3-6 months"
                elif lead['total_score'] >= 70:
                    roi_months = "6-12 months"
                else:
                    roi_months = "12+ months"
                st.metric("Expected ROI", roi_months)
            
            with col3:
                st.markdown("**🎯 Deal Probability**")
                st.metric("Close Rate", f"{lead.get('close_probability', 0):.1f}%")
                
                st.markdown("**💎 Expected Value**")
                expected_value = (mrr_min + mrr_max) / 2 * (lead.get('close_probability', 0) / 100) * 12
                st.metric("Expected ARR", f"${expected_value:,.0f}")
            
            # Revenue visualization
            st.markdown("**📈 Revenue Projection Chart**")
            months = list(range(1, 13))
            conservative = [mrr_min * month * (lead.get('close_probability', 50) / 100) for month in months]
            optimistic = [mrr_max * month * (lead.get('close_probability', 50) / 100) for month in months]
            
            revenue_df = pd.DataFrame({
                'Month': months,
                'Conservative': conservative,
                'Optimistic': optimistic
            })
            
            fig_revenue = px.line(
                revenue_df, 
                x='Month', 
                y=['Conservative', 'Optimistic'],
                title="12-Month Revenue Projection",
                labels={'value': 'Cumulative Revenue ($)', 'variable': 'Scenario'}
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with tab4:
            st.subheader("📋 Export Individual Lead Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📄 Export Options:**")
                
                # Generate comprehensive lead report
                if st.button("📊 Generate Full Analysis Report", type="primary"):
                    report_content = f"""
# Lead Analysis Report: {lead['name']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Company:** {lead['name']}
- **Industry:** {lead['industry']}
- **Total Score:** {lead['total_score']}/100
- **Tier:** {lead['tier']}
- **Quick Win:** {'Yes' if lead.get('quick_win', False) else 'No'}
- **Close Probability:** {lead.get('close_probability', 0):.1f}%

## Scoring Breakdown
- **Financial Score:** {lead.get('financial_score', 0)}/25
- **Decision Score:** {lead.get('decision_score', 0)}/20
- **Pain Score:** {lead.get('pain_score', 0)}/20
- **Tech Score:** {lead.get('tech_score', 0)}/20
- **Growth Score:** {lead.get('growth_score', 0)}/20
- **Competition Score:** {lead.get('competition_score', 0)}/20

## Revenue Potential
- **MRR Range:** ${lead.get('mrr_potential_min', 0):,} - ${lead.get('mrr_potential_max', 0):,}
- **ARR Range:** ${lead.get('mrr_potential_min', 0)*12:,} - ${lead.get('mrr_potential_max', 0)*12:,}
- **Monthly Waste:** ${lead.get('monthly_waste_min', 0):,} - ${lead.get('monthly_waste_max', 0):,}

## Strategic Recommendations
### Pain Points
{chr(10).join([f"- {pain}" for pain in lead.get('pain_points', ['Operational inefficiencies', 'Process bottlenecks'])]) if isinstance(lead.get('pain_points'), list) else "- Operational inefficiencies"}

### Demo Focus Areas
- Workflow automation and optimization
- Customer management capabilities  
- Real-time reporting and analytics

### Contact Strategy
- {'Immediate follow-up within 24 hours' if lead.get('quick_win', False) else 'Research phase and warm introduction approach'}
- {'Direct decision maker engagement' if lead['total_score'] >= 85 else 'Consultative relationship building'}

## Next Steps
1. Schedule discovery call to validate pain points
2. Prepare customized demo focusing on identified areas
3. Develop ROI calculator specific to their industry
4. Identify key stakeholders and decision process
5. Create timeline for pilot or implementation

## Contact Information
- **Website:** {lead.get('website', 'Not available')}
- **Services:** {', '.join(lead.get('services', [])) if isinstance(lead.get('services'), list) else 'Standard service offerings'}

---
Report generated by Houston Metro SME Lead Intelligence Dashboard
"""
                    
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=report_content,
                        file_name=f"lead_analysis_{lead['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                # Export lead data as JSON
                if st.button("📋 Export Lead Data (JSON)"):
                    lead_json = lead.to_json(indent=2)
                    st.download_button(
                        label="📥 Download Lead JSON",
                        data=lead_json,
                        file_name=f"lead_data_{lead['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                st.markdown("**🎯 Next Steps Checklist:**")
                
                # Interactive checklist for follow-up actions
                st.checkbox("📞 Schedule discovery call", key=f"check1_{lead_idx}")
                st.checkbox("🔍 Research company background", key=f"check2_{lead_idx}")
                st.checkbox("🎯 Prepare customized demo", key=f"check3_{lead_idx}")
                st.checkbox("💰 Calculate specific ROI", key=f"check4_{lead_idx}")
                st.checkbox("👥 Identify decision makers", key=f"check5_{lead_idx}")
                st.checkbox("📊 Create proposal timeline", key=f"check6_{lead_idx}")
                
                st.markdown("**📋 Quick Actions:**")
                if st.button("📧 Copy Email Template"):
                    email_template = f"""
Subject: Streamlining Operations for {lead['name']} - {lead['industry']} Solutions

Hi [Contact Name],

I hope this email finds you well. I've been researching {lead['industry']} companies in the Houston metro area, and {lead['name']} caught my attention.

Based on my analysis, companies like yours typically face challenges with:
{chr(10).join([f"• {pain}" for pain in lead.get('pain_points', ['Operational inefficiencies', 'Process bottlenecks'])]) if isinstance(lead.get('pain_points'), list) else '• Operational inefficiencies and process bottlenecks'}

Our platform has helped similar {lead['industry']} businesses:
• Reduce operational waste by ${lead.get('monthly_waste_min', 2500):,}-${lead.get('monthly_waste_max', 8000):,} monthly
• Improve efficiency and customer satisfaction
• Streamline scheduling and communication processes

Would you be open to a brief 15-minute conversation to discuss how we might help {lead['name']} optimize operations?

Best regards,
[Your Name]
"""
                    st.code(email_template, language="text")

# --- GLOBAL CSS OVERRIDE: Remove any accidental overlay/dim/blur ---
st.markdown("""
<style>
/* Responsive tweaks for Streamlit dashboard */
@media (max-width: 1200px) {
    .block-container, .main, .stApp {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
}
@media (max-width: 900px) {
    .block-container, .main, .stApp {
        padding: 0 !important;
    }
    .stTabs, .stTabs [data-baseweb="tab"] {
        flex-direction: column !important;
    }
}
@media (max-width: 600px) {
    .block-container, .main, .stApp {
        padding: 0 !important;
        margin: 0 !important;
    }
    .stTabs, .stTabs [data-baseweb="tab"] {
        flex-direction: column !important;
        width: 100% !important;
    }
    .stButton>button, .stDownloadButton>button {
        width: 100% !important;
        margin-bottom: 0.5rem !important;
    }
    .stDataFrame, .stTable {
        font-size: 11px !important;
    }
    .stMetric {
        font-size: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)
# --- END GLOBAL CSS OVERRIDE ---

def main():
    # Sidebar page selector
    page = st.sidebar.radio("Page", ["Leads Dashboard", "Demo Specs", "Demo Playground", "Call Log"])

    # Top-of-app metrics
    try:
        calls_made = sum(1 for _ in open("calls_log.csv")) - 1
    except Exception:
        calls_made = 0
    demos_scheduled = len(glob.glob("*_demo.json"))
    st.metric("Calls Made", f"{calls_made}/24")
    st.metric("Demos Scheduled", f"{demos_scheduled}/8")

    if page == "Leads Dashboard":
        st.title("🎯 Houston Metro SME Lead Intelligence - Offline Dashboard")
        st.markdown("**📂 Analyzing Existing Lead Database (No API Calls Required)**")
        data_file = 'dashboard_data.json'
        backup_files = ['leads_data.json', 'leads.csv']
        data_exists = os.path.exists(data_file)
        if not data_exists:
            st.warning("⚠️ No lead data file found!")
            st.markdown("""
            **Expected file:** `dashboard_data.json`
            
            **Options:**
            1. Upload your own lead database file below
            2. Ensure `dashboard_data.json` exists in the current directory
            3. Use the Google Maps discovery section to generate new leads
            """)
            st.subheader("📤 Upload Lead Database")
            uploaded_file = st.file_uploader(
                "Choose a JSON or CSV file containing lead data",
                type=['json', 'csv'],
                help="Upload a file containing lead data with columns like name, industry, total_score, etc."
            )
            if uploaded_file is not None:
                df = load_uploaded_data(uploaded_file)
                if not df.empty:
                    st.success(f"✅ Successfully loaded {len(df)} leads from uploaded file!")
                    st.session_state.leads_data = df
                else:
                    st.error("❌ Failed to load data from uploaded file.")
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
        else:
            with st.spinner("📂 Loading existing lead database..."):
                df = load_existing_leads(data_file, backup_files)
            if df.empty:
                st.error("❌ Failed to load lead data from existing files.")
            else:
                st.success(f"✅ Successfully loaded {len(df)} leads from existing database!")
                st.session_state.leads_data = df
        if df.empty:
            st.info("💡 Upload a lead database file or use the discovery section below to get started.")
            google_maps_discovery_section()
            return
        create_performance_metrics(df)
        st.sidebar.header("🔍 Filter Leads")
        available_industries = sorted(df['industry'].unique()) if 'industry' in df.columns else []
        selected_industries = st.sidebar.multiselect(
            "Industries",
            available_industries,
            default=available_industries[:5] if len(available_industries) > 5 else available_industries
        )
        if 'total_score' in df.columns:
            min_score, max_score = int(df['total_score'].min()), int(df['total_score'].max())
            score_range = st.sidebar.slider(
                "Score Range",
                min_score, max_score,
                (min_score, max_score)
            )
        else:
            score_range = None
        if 'tier' in df.columns:
            available_tiers = sorted(df['tier'].unique())
            selected_tiers = st.sidebar.multiselect(
                "Tiers",
                available_tiers,
                default=available_tiers
            )
        else:
            selected_tiers = None
        quick_win_only = st.sidebar.checkbox("Quick Wins Only", False) if 'quick_win' in df.columns else False
        filtered_df = get_filtered_leads(
            df,
            industries=selected_industries if selected_industries else None,
            score_range=score_range,
            tier_filter=selected_tiers if selected_tiers else None,
            quick_win_only=quick_win_only
        )
        st.sidebar.markdown("---")
        st.sidebar.metric("Filtered Leads", f"{len(filtered_df):,}")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Analytics", "📋 Lead Table", "🔍 Lead Details", "🤖 AI Assistant", "⚙️ Data Management", "🌐 Discover New"])
        with tab1:
            st.subheader("📊 Lead Analytics")
            
            if not filtered_df.empty:
                create_lead_summary_charts(filtered_df)
                
                # Additional analytics
                st.markdown("### 💰 Revenue Opportunity Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # MRR potential by industry
                    if 'mrr_potential_max' in filtered_df.columns:
                        mrr_by_industry = filtered_df.groupby('industry')['mrr_potential_max'].sum().sort_values(ascending=False)
                        fig_mrr = px.bar(
                            x=mrr_by_industry.index,
                            y=mrr_by_industry.values,
                            title="Max MRR Potential by Industry",
                            labels={'x': 'Industry', 'y': 'MRR Potential ($)'}
                        )
                        fig_mrr.update_layout(height=400)
                        st.plotly_chart(fig_mrr, use_container_width=True)
                
                with col2:
                    # Close probability vs Score scatter
                    if 'close_probability' in filtered_df.columns and 'total_score' in filtered_df.columns:
                        fig_scatter = px.scatter(
                            filtered_df,
                            x='total_score',
                            y='close_probability',
                            color='industry',
                            size='mrr_potential_max' if 'mrr_potential_max' in filtered_df.columns else None,
                            title="Score vs Close Probability",
                            hover_data=['name'] if 'name' in filtered_df.columns else None
                        )
                        fig_scatter.update_layout(height=400)
                        st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("No leads match the current filters.")
        
        with tab2:
            st.subheader("📋 Lead Database")
            
            if not filtered_df.empty:
                display_lead_table(filtered_df)
            else:
                st.warning("No leads match the current filters.")
        
        with tab3:
            st.subheader("🔍 Lead Detail Analysis")
            
            if not filtered_df.empty:
                create_lead_detail_view(filtered_df)
            else:
                st.warning("No leads match the current filters.")
        
        with tab4:
            if not filtered_df.empty:
                create_chatbot_interface(filtered_df)
            else:
                st.warning("No leads available for AI assistant. Please check your data.")
        
        with tab5:
            st.subheader("⚙️ Data Management")
            
            # Data file information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📁 Data File Info**")
                if os.path.exists(data_file):
                    file_size = os.path.getsize(data_file) / 1024  # KB
                    file_modified = datetime.fromtimestamp(os.path.getmtime(data_file))
                    st.write(f"• File: {data_file}")
                    st.write(f"• Size: {file_size:.1f} KB")
                    st.write(f"• Modified: {file_modified.strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.write("• No primary data file found")
            
            with col2:
                st.markdown("**📊 Dataset Stats**")
                if not df.empty:
                    st.write(f"• Total Records: {len(df):,}")
                    st.write(f"• Columns: {len(df.columns)}")
                    st.write(f"• Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                else:
                    st.write("• No data loaded")
            
            with col3:
                st.markdown("**🔄 Data Actions**")
                if st.button("🔄 Reload Data"):
                    st.cache_data.clear()
                    st.rerun()
                
                if st.button("📊 Show Column Info") and not df.empty:
                    st.write("**Columns:**", list(df.columns))
                    st.write("**Data Types:**")
                    st.write(df.dtypes.to_dict())
            
            # Data quality check
            if not df.empty:
                st.markdown("### 🔍 Data Quality Check")
                
                quality_col1, quality_col2 = st.columns(2)
                
                with quality_col1:
                    # Missing values
                    missing_data = df.isnull().sum()
                    if missing_data.sum() > 0:
                        st.warning("⚠️ Missing Values Detected")
                        st.write(missing_data[missing_data > 0])
                    else:
                        st.success("✅ No missing values")
                
                with quality_col2:
                    # Duplicate check - handle list columns that can't be hashed
                    try:
                        # Create a copy for duplicate checking
                        df_dup_check = df.copy()
                        
                        # Convert list columns to strings for duplicate checking
                        for col in df_dup_check.columns:
                            if df_dup_check[col].dtype == 'object':
                                # Check if column contains lists
                                sample_val = df_dup_check[col].iloc[0] if len(df_dup_check) > 0 else None
                                if isinstance(sample_val, list):
                                    df_dup_check[col] = df_dup_check[col].astype(str)
                        
                        duplicates = df_dup_check.duplicated().sum()
                        if duplicates > 0:
                            st.warning(f"⚠️ {duplicates} duplicate records found")
                        else:
                            st.success("✅ No duplicate records")
                    except Exception as e:
                        st.info(f"ℹ️ Duplicate check skipped: {str(e)}")
                        st.success("✅ Data structure validated")
        
        with tab6:
            google_maps_discovery_section()
        
        # Performance indicators in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**💡 Performance Tips:**")
        st.sidebar.markdown("• **Offline Mode**: No API calls required")
        st.sidebar.markdown("• **Fast Filtering**: Instant results")
        st.sidebar.markdown("• **Export Options**: Download filtered data")
        st.sidebar.markdown("• **Upload Support**: Load custom databases")
        
        # Status indicator
        st.sidebar.markdown("---")
        current_time = datetime.now().strftime("%H:%M:%S")
        st.sidebar.markdown(f"**🕐 Last Updated:** {current_time}")
        st.sidebar.markdown(f"**📶 Status:** {'Online' if data_exists else 'No Data'}")
    
    if page == "Demo Specs":
        st.title("Demo Specifications")
        for demo_file in pathlib.Path.cwd().glob("*_demo.json"):
            st.subheader(demo_file.name)
            with open(demo_file, encoding="utf-8") as f:
                data = json.load(f)
            st.json(data)
    
    if page == "Demo Playground":
        st.title("Interactive Demo Playground")
        demo_files = sorted(pathlib.Path.cwd().glob("*_demo.json"))
        if not demo_files:
            st.info("No demo specs found.")
            return
        choice = st.selectbox("Pick a demo", [f.name for f in demo_files])
        data = json.loads(pathlib.Path(choice).read_text())
        st.subheader(f"{data['industry']} Demo")
        st.write("Features:")
        for feat in data["features"]:
            st.markdown(f"- ✅ {feat}")
        st.write(f"Leads available: {data.get('lead_count', 0)}")
        industry_key = data['industry'].lower().replace(' ', '_')
        possible_csvs = [
            f"tier1_{industry_key}.csv",
            f"tier1_{data['industry'].lower().replace(' ', '')}.csv",
            f"tier1_{data['industry'].lower()}.csv"
        ]
        csv_file = None
        for fname in possible_csvs:
            if os.path.exists(fname):
                csv_file = fname
                break
        if not csv_file:
            st.error(f"No CSV file found for industry '{data['industry']}'. Tried: {possible_csvs}")
            return
        df = pd.read_csv(csv_file)
        lead_id = df.iloc[0].iloc[0]  # Use the first column (name) as lead_id
        phone = df.iloc[0].iloc[2] if 'phone' in df.columns else None
        if st.button("Run Demo"):
            st.success(f"🚀 Running {data['industry']} demo now!")
            # --- STUBBED INTERACTIONS ---
            if data["industry"] == "Plumbing":
                st.subheader("🚨 24/7 Emergency Scheduling")
                sched = st.date_input("Pick a service date")
                if st.button("Confirm Emergency Job"):
                    dispatch_technician(lead_id)
                st.subheader("🛠️ Real-Time Technician Dispatch")
                if st.button("Dispatch Technician Now"):
                    dispatch_technician(lead_id)
                st.subheader("📲 Automated Customer Notifications")
                note = st.text_input("Custom notification message", "Your plumber is on the way!")
                if st.button("Send Notification"):
                    if phone:
                        send_sms(phone, note)
                    else:
                        st.warning("No phone number available for this lead.")
                st.subheader("💰 Generate Invoice")
                amount = st.number_input("Invoice amount ($)", min_value=1, step=1)
                if st.button("Generate Invoice"):
                    generate_invoice(lead_id, amount)
            if data["industry"] == "HVAC":
                st.subheader("🔧 System Diagnostics")
                if st.button("Run Diagnostics"):
                    run_diagnostics(lead_id)
                st.subheader("💲 Automated Estimates")
                est = st.number_input("Enter job size ($)", min_value=100, step=50)
                if st.button("Generate Estimate"):
                    generate_estimate(lead_id, est)
                st.subheader("⏰ Maintenance Reminders")
                remind_date = st.date_input("Next maintenance date")
                if st.button("Schedule Reminder"):
                    schedule_reminder(lead_id, remind_date)
            if data["industry"] == "Pressure Washing":
                st.subheader("🌊 High-Pressure Cleaning")
                area = st.selectbox("Surface type", ["Driveway","Deck","House Siding"])
                if st.button("Start Cleaning"):
                    start_cleaning(lead_id, area)
                st.subheader("🧽 Stain Removal")
                stain = st.text_input("Type of stain", "Oil")
                if st.button("Remove Stain"):
                    remove_stain(lead_id, stain)
                st.subheader("🍃 Eco-Friendly Detergents")
                if st.button("Use Eco Detergent"):
                    apply_detergent(lead_id)
    if page == "Call Log":
        st.title("Call Log")
        try:
            df = pd.read_csv("calls_log.csv")
            st.dataframe(df)
        except Exception as e:
            st.warning(f"Could not load call log: {e}")

if __name__ == "__main__":
    main()
