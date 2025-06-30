import openai
import streamlit as st
import pandas as pd
import json
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

class LeadAnalysisChatbot:
    def __init__(self, api_key: str, leads_data: pd.DataFrame = None):
        """Initialize the AI chatbot assistant for lead analysis"""
        self.client = openai.OpenAI(api_key=api_key)
        self.leads_data = leads_data
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())
        
    def get_system_prompt(self) -> str:
        """Get the system prompt that defines the chatbot's personality and capabilities"""
        return """You are a friendly AI assistant for a lead generation company that helps small and medium businesses (SMEs) find customers. You specialize in explaining lead analysis and sales strategy in simple, non-technical terms.

Your personality:
- Speak like a helpful sales coach who explains things clearly
- Use simple language that anyone can understand
- Be encouraging and positive
- Give specific, actionable advice
- Translate technical terms into plain English
- Keep responses concise but helpful

Your expertise:
- Lead scoring (how we rate potential customers)
- Sales strategy (what to say and when to call)
- Business analysis (understanding what companies need)
- Action planning (daily/weekly tasks)

Key explanations to remember:
- "Lead score" = How likely someone is to buy (higher = better)
- "Pain points" = Problems the business is having
- "Quick win" = Easy sale that can happen fast
- "Decision maker" = Person who can say yes to buying
- "Pipeline" = List of potential customers in different stages

When someone asks about leads:
1. Explain scores in simple terms ("85 means they really need help")
2. Give specific talking points ("Say this when you call...")
3. Prioritize actions ("Call these 3 companies first")
4. Explain the "why" behind recommendations

Always be helpful, encouraging, and focus on practical next steps. Keep responses under 300 words unless asked for detailed analysis."""

    def get_lead_context(self) -> str:
        """Generate context about the current lead data for the AI"""
        if self.leads_data is None or self.leads_data.empty:
            return "No lead data is currently available."
        
        # Generate summary statistics
        total_leads = len(self.leads_data)
        avg_score = self.leads_data['total_score'].mean() if 'total_score' in self.leads_data.columns else 0
        high_score_leads = len(self.leads_data[self.leads_data['total_score'] >= 80]) if 'total_score' in self.leads_data.columns else 0
        quick_wins = len(self.leads_data[self.leads_data.get('quick_win', False) == True]) if 'quick_win' in self.leads_data.columns else 0
        
        # Get top industries
        top_industries = []
        if 'industry' in self.leads_data.columns:
            top_industries = self.leads_data['industry'].value_counts().head(3).to_dict()
        
        # Get top pain points
        top_pain_points = []
        if 'pain_points' in self.leads_data.columns:
            pain_points_flat = []
            for pp in self.leads_data['pain_points'].dropna():
                if isinstance(pp, list):
                    pain_points_flat.extend(pp)
                elif isinstance(pp, str):
                    pain_points_flat.append(pp)
            if pain_points_flat:
                from collections import Counter
                top_pain_points = dict(Counter(pain_points_flat).most_common(5))
        
        context = f"""
Current Lead Database Summary:
- Total leads: {total_leads}
- Average score: {avg_score:.1f}/100
- High-priority leads (80+ score): {high_score_leads}
- Quick win opportunities: {quick_wins}

Top Industries: {top_industries}
Top Pain Points: {top_pain_points}

The scoring system rates leads from 0-100 based on:
- Business size and revenue potential
- Pain points that match our services
- Decision maker accessibility
- Likelihood to buy soon
"""
        return context

    def get_conversation_starter_suggestions(self) -> List[str]:
        """Get suggested questions users can ask"""
        return [
            "Which leads should I call first today?",
            "What does a score of 85 mean?",
            "How do I start a conversation with a plumbing company?",
            "Show me the best quick win opportunities",
            "What should I say to get past the receptionist?",
            "Which businesses need our help the most?",
            "Give me a daily action plan",
            "What are the most common pain points?",
            "How do I explain our services simply?",
            "Which leads are most likely to buy soon?"
        ]

    def generate_sales_script(self, lead_data: Dict) -> str:
        """Generate a personalized sales script for a specific lead"""
        company_name = lead_data.get('name', 'the company')
        industry = lead_data.get('industry', 'business')
        pain_points = lead_data.get('pain_points', [])
        score = lead_data.get('total_score', 0)
        
        pain_points_text = ""
        if pain_points and isinstance(pain_points, list) and len(pain_points) > 0:
            pain_points_text = f"I noticed {industry} companies often struggle with {pain_points[0].lower()}"
        
        script = f"""
📞 CALL SCRIPT FOR {company_name.upper()}

Opening:
"Hi, this is [Your Name] from [Company]. I help {industry} businesses like {company_name} grow their customer base. Do you have 2 minutes?"

{pain_points_text and f"Problem Hook:\\n{pain_points_text}. Is this something you're dealing with?"} 

Value Proposition:
"We've helped similar {industry} companies increase their leads by 40-60% without expensive advertising. Would you be interested in hearing how?"

Next Steps:
"I'd love to show you exactly how this works. Are you free for a quick 15-minute call this week?"

💡 Why this lead scored {score}/100:
{score >= 80 and "HIGH PRIORITY - Strong indicators they need help and can afford it" or 
 score >= 60 and "GOOD OPPORTUNITY - Solid potential, worth pursuing" or
 "LOWER PRIORITY - May need more nurturing before they're ready"}
"""
        return script

    def analyze_lead_question(self, question: str, lead_data: Dict) -> str:
        """Analyze a specific lead and provide detailed explanation"""
        name = lead_data.get('name', 'Unknown')
        score = lead_data.get('total_score', 0)
        industry = lead_data.get('industry', 'Unknown')
        pain_points = lead_data.get('pain_points', [])
        quick_win = lead_data.get('quick_win', False)
        
        analysis = f"""
🎯 ANALYSIS FOR {name.upper()}

Score: {score}/100 
{"🔥 HIGH PRIORITY" if score >= 80 else "✅ GOOD PROSPECT" if score >= 60 else "⏳ NEEDS NURTURING"}

What this means in simple terms:
{score >= 80 and "This business is practically raising their hand saying 'I need help!' They have clear problems, money to spend, and are likely to buy soon." or
 score >= 60 and "This is a solid opportunity. They have some pain points and the ability to purchase, but may need a bit more convincing." or
 "This lead needs more work. They might not be ready to buy right now, or we need to understand their needs better."}

Industry: {industry}
{"Quick Win Opportunity! ⚡" if quick_win else "Standard Sales Process 📈"}

Pain Points (their problems we can solve):
{chr(10).join([f"• {pp}" for pp in pain_points if isinstance(pain_points, list)]) if pain_points else "• No specific pain points identified yet"}

WHAT TO DO NEXT:
{quick_win and "1. Call them TODAY - they're ready to buy\\n2. Lead with cost savings and quick results\\n3. Ask for decision maker directly" or
"1. Research them a bit more online\\n2. Find warm introduction if possible\\n3. Lead with education, not sales pitch"}
"""
        return analysis

    def get_daily_action_plan(self) -> str:
        """Generate a daily action plan based on current leads"""
        if self.leads_data is None or self.leads_data.empty:
            return "No leads available for action planning."
        
        # Get top leads
        top_leads = self.leads_data.nlargest(5, 'total_score') if 'total_score' in self.leads_data.columns else self.leads_data.head(5)
        quick_wins = self.leads_data[self.leads_data.get('quick_win', False) == True] if 'quick_win' in self.leads_data.columns else pd.DataFrame()
        
        plan = f"""
📋 YOUR DAILY ACTION PLAN

🔥 TOP PRIORITY CALLS (Do these first):
"""
        
        for idx, (_, lead) in enumerate(top_leads.head(3).iterrows(), 1):
            name = lead.get('name', 'Unknown')
            score = lead.get('total_score', 0)
            industry = lead.get('industry', 'Unknown')
            plan += f"{idx}. {name} ({industry}) - Score: {score}/100\n"
        
        if not quick_wins.empty:
            plan += f"\n⚡ QUICK WIN OPPORTUNITIES ({len(quick_wins)} available):\n"
            for _, lead in quick_wins.head(2).iterrows():
                name = lead.get('name', 'Unknown')
                plan += f"• {name} - Call today, they're ready to buy!\n"
        
        plan += """
📞 DAILY TARGETS:
• Make 10 calls minimum
• Book 2 discovery meetings
• Send 5 follow-up emails
• Update CRM after each call

🎯 CONVERSATION GOALS:
• Identify their biggest business challenge
• Understand their budget and timeline
• Find out who makes buying decisions
• Schedule next step (demo/meeting)

💡 REMEMBER:
• Focus on helping, not selling
• Ask questions, don't just pitch
• Listen for pain points
• Be genuinely curious about their business
"""
        return plan

    def chat_response(self, user_message: str, message_id: str = None) -> str:
        """Generate AI response to user message with proper session management"""
        try:
            # Validate input
            if not user_message or user_message.strip() == "":
                return "Please ask me a question about your leads or sales strategy!"
            
            # Prevent duplicate processing
            if message_id and any(msg.get('id') == message_id for msg in self.conversation_history):
                return "I've already responded to that message."
            
            # Add user message to conversation history with ID
            user_msg = {
                "role": "user", 
                "content": user_message.strip(),
                "id": message_id or str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_history.append(user_msg)
            
            # Prepare context
            lead_context = self.get_lead_context()
            system_prompt = self.get_system_prompt()
            
            # Handle simple greetings without API call
            greeting_words = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']
            if any(word in user_message.lower() for word in greeting_words) and len(user_message.split()) <= 3:
                response = "Hello! I'm your AI sales assistant. I can help you understand your leads, create sales scripts, plan your daily calls, and much more. Try asking me 'Which leads should I call first?' or click one of the suggestion buttons below!"
            
            # Check for specific requests
            elif "daily action plan" in user_message.lower() or "what should i do today" in user_message.lower():
                response = self.get_daily_action_plan()
            elif "script" in user_message.lower() and "call" in user_message.lower() and self.leads_data is not None:
                # For script requests, use the top lead as example
                top_lead = self.leads_data.iloc[0].to_dict() if not self.leads_data.empty else {}
                response = self.generate_sales_script(top_lead)
            else:
                # Use OpenAI for general questions
                messages = [
                    {"role": "system", "content": f"{system_prompt}\n\nCurrent Lead Data:\n{lead_context}"},
                    *[{"role": msg["role"], "content": msg["content"]} for msg in self.conversation_history[-8:]]  # Keep last 8 messages for context
                ]
                
                completion = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7,
                    timeout=10
                )
                
                response = completion.choices[0].message.content
            
            # Add AI response to conversation history with ID
            ai_msg = {
                "role": "assistant", 
                "content": response,
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }
            self.conversation_history.append(ai_msg)
            
            return response
            
        except Exception as e:
            error_msg = f"I'm having trouble right now. Please try again in a moment. (Error: {str(e)[:50]}...)"
            return error_msg

    def update_leads_data(self, new_leads_data: pd.DataFrame):
        """Update the leads data for the chatbot"""
        self.leads_data = new_leads_data
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())

def create_chatbot_interface(leads_data: pd.DataFrame = None):
    """Create the Streamlit interface for the chatbot with improved UX"""
    
    # Add custom CSS for better chat styling and mobile optimization
    st.markdown("""
    <style>
    /* Enhanced mobile-responsive chat styling */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        -webkit-overflow-scrolling: touch;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #0066cc;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        color: #262730 !important;
        word-wrap: break-word;
    }
    
    .bot-message {
        background-color: #ffffff;
        border-left: 4px solid #28a745;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #e6e6e6;
        color: #262730 !important;
        word-wrap: break-word;
    }
    
    .chat-input {
        border: 2px solid #0066cc;
        border-radius: 8px;
        padding: 8px;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #262730;
        font-size: 1rem !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
    }
    
    .quick-action-btn {
        margin: 2px;
        padding: 12px 16px;
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.9rem;
        width: 100%;
        touch-action: manipulation;
        min-height: 48px;
    }
    
    .quick-action-btn:hover {
        background-color: #0056b3;
        transform: scale(1.02);
        transition: all 0.2s ease;
    }
    
    /* Mobile-specific optimizations */
    @media screen and (max-width: 768px) {
        .chat-container {
            max-height: 250px;
            padding: 8px;
            font-size: 0.9rem;
        }
        
        .user-message, .bot-message {
            padding: 10px;
            font-size: 0.9rem;
            margin: 6px 0;
        }
        
        .quick-action-btn {
            padding: 14px 12px;
            font-size: 0.85rem;
            margin: 4px 0;
            min-height: 52px;
        }
        
        .stButton > button {
            height: 3rem !important;
            font-size: 0.9rem !important;
            padding: 0.75rem !important;
            margin: 0.25rem 0 !important;
            width: 100% !important;
            touch-action: manipulation;
        }
        
        .stForm {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .stTextInput > div > div > input {
            font-size: 1rem !important;
            padding: 0.75rem !important;
            min-height: 48px !important;
        }
        
        .stColumns {
            gap: 0.5rem !important;
        }
        
        .element-container {
            margin-bottom: 0.5rem !important;
        }
    }
    
    /* Tablet optimizations */
    @media screen and (min-width: 769px) and (max-width: 1024px) {
        .chat-container {
            max-height: 350px;
            font-size: 0.95rem;
        }
        
        .quick-action-btn {
            padding: 12px 14px;
            font-size: 0.9rem;
            min-height: 48px;
        }
    }
    
    /* Enhanced touch feedback */
    .stButton > button:active {
        transform: scale(0.98) !important;
        background-color: #004494 !important;
    }
    
    /* Improved form styling for mobile */
    .stForm {
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Better visual hierarchy on mobile */
    h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
        color: #262730 !important;
    }
    
    @media screen and (max-width: 768px) {
        h3 {
            font-size: 1rem !important;
            margin-bottom: 0.4rem !important;
        }
    }
    
    /* Expandable sections mobile optimization */
    .streamlit-expanderHeader {
        padding: 1rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        background-color: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    @media screen and (max-width: 768px) {
        .streamlit-expanderHeader {
            padding: 0.75rem !important;
            font-size: 0.9rem !important;
        }
    }
    
    /* Loading states mobile optimization */
    .stSpinner {
        padding: 1.5rem !important;
        text-align: center !important;
    }
    
    /* Alert messages mobile styling */
    .stAlert {
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        font-size: 0.9rem !important;
    }
    
    @media screen and (max-width: 768px) {
        .stAlert {
            padding: 0.75rem !important;
            font-size: 0.85rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("🤖 AI Sales Assistant")
    st.markdown("**Ask me anything about your leads in plain English!**")
    
    # Initialize session state variables
    if 'chatbot_session_id' not in st.session_state:
        st.session_state.chatbot_session_id = str(uuid.uuid4())
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        
    if 'last_message_id' not in st.session_state:
        st.session_state.last_message_id = None
        
    if 'processing_message' not in st.session_state:
        st.session_state.processing_message = False
        
    if 'last_input_time' not in st.session_state:
        st.session_state.last_input_time = 0
    
    # Initialize chatbot
    api_key = "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"
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = LeadAnalysisChatbot(api_key, leads_data)
    else:
        # Update leads data if it's changed
        st.session_state.chatbot.update_leads_data(leads_data)
    
    # Welcome message for first-time users
    if not st.session_state.chat_messages:
        st.info("👋 **Welcome!** I'm here to help you understand your leads and boost your sales. Try clicking a suggestion below or ask me anything!")
    
    # Quick action buttons (mobile-optimized layout)
    st.markdown("### 🚀 Quick Actions")
    
    # Check if we're on mobile (approximation based on screen width detection)
    # For mobile, we'll use a single column layout with larger buttons
    
    # Mobile-first design: Single column on small screens, multiple columns on larger screens
    with st.container():
        # Use responsive columns that stack on mobile
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📅 Today's Top Leads", key="action_top_leads", use_container_width=True, help="Get your priority calls for today"):
                handle_quick_action("Which leads should I call first today?")
            if st.button("📞 Call Script", key="action_script", use_container_width=True, help="Get a sales script for calling"):
                handle_quick_action("Give me a sales script for calling prospects")
            if st.button("🎯 Daily Plan", key="action_plan", use_container_width=True, help="Get your complete daily action plan"):
                handle_quick_action("Give me my daily action plan")
        
        with col2:
            if st.button("💡 What to Say", key="action_talking_points", use_container_width=True, help="Get conversation starters"):
                handle_quick_action("What should I say when I call a potential customer?")
            if st.button("🔥 Quick Wins", key="action_quick_wins", use_container_width=True, help="Find easy sales opportunities"):
                handle_quick_action("Show me the best quick win opportunities")
            if st.button("📊 Score Guide", key="action_scores", use_container_width=True, help="Understand lead scoring"):
                handle_quick_action("What do the lead scores mean?")
        
        with col3:
            if st.button("🏭 Best Industries", key="action_industries", use_container_width=True, help="Which industries to focus on"):
                handle_quick_action("Which industries should I focus on?")
            if st.button("❓ Handle Objections", key="action_objections", use_container_width=True, help="Deal with common objections"):
                handle_quick_action("How do I handle common sales objections?")
            if st.button("📈 Follow-up Tips", key="action_followup", use_container_width=True, help="Follow-up strategies"):
                handle_quick_action("What's the best follow-up strategy?")
    
    # Mobile-specific additional quick actions in collapsible section
    with st.expander("📱 More Quick Actions", expanded=False):
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🚪 Get Past Receptionist", key="action_gatekeeper", use_container_width=True):
                handle_quick_action("How do I get past the receptionist?")
            if st.button("💰 Revenue Focus", key="action_revenue", use_container_width=True):
                handle_quick_action("Which leads are worth the most money?")
        
        with col_b:
            if st.button("⏰ Best Call Times", key="action_timing", use_container_width=True):
                handle_quick_action("When is the best time to call prospects?")
            if st.button("📝 Call Notes", key="action_notes", use_container_width=True):
                handle_quick_action("What should I write down during calls?")
    
    # Conversation examples
    with st.expander("💡 See Example Conversations"):
        st.markdown("""
        **Example Questions You Can Ask:**
        
        📋 **Planning:** "What should I do today?" → Get a complete daily action plan
        
        🎯 **Lead Analysis:** "Tell me about ABC Company" → Detailed lead breakdown
        
        📞 **Sales Scripts:** "How do I call a restaurant?" → Custom calling script
        
        💰 **Revenue Focus:** "Which leads are worth the most?" → Revenue prioritization
        
        🔍 **Understanding:** "Why is this lead scored 85?" → Score explanation
        
        **Tips for Better Results:**
        - Be specific: "plumbing companies" vs "businesses"  
        - Ask follow-ups: "Tell me more about that strategy"
        - Use your own words: "How do I not sound salesy?"
        """)
    
    st.markdown("---")
    
    # Chat history display with better formatting
    if st.session_state.chat_messages:
        st.markdown("### 💬 Conversation")
        
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                    <strong style="color: #0066cc;">👤 You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                    <strong style="color: #28a745;">🤖 AI Assistant:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat input with better UX
    st.markdown("### ✨ Ask Me Anything")
    
    # Input validation and debouncing
    current_time = time.time()
    
    # Create unique form to prevent resubmission
    with st.form(key=f"chat_form_{st.session_state.chatbot_session_id}", clear_on_submit=True):
        user_input = st.text_input(
            "Your message:",
            value="",
            placeholder="Type your question here... (e.g., 'Which leads should I call first?')",
            help="Ask me anything about your leads, sales strategy, or daily planning!",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            submit_button = st.form_submit_button("💬 Send Message", type="primary")
        
        with col2:
            if st.form_submit_button("🗑️ New Conversation"):
                reset_conversation()
        
        with col3:
            if st.session_state.processing_message:
                st.info("🤔 Thinking...")
    
    # Process form submission
    if submit_button and user_input:
        # Debouncing - prevent rapid submissions
        if current_time - st.session_state.last_input_time < 1.0:
            st.warning("⏱️ Please wait a moment before sending another message.")
            return
        
        st.session_state.last_input_time = current_time
        process_user_message(user_input.strip())

def handle_quick_action(question: str):
    """Handle quick action button clicks"""
    if not st.session_state.processing_message:
        process_user_message(question)

def process_user_message(user_message: str):
    """Process user message with proper session management"""
    if not user_message or user_message.strip() == "":
        st.warning("⚠️ Please enter a message before sending.")
        return
    
    # Check for duplicate messages
    if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
        if st.session_state.chat_messages[-1]["content"] == user_message:
            return  # Don't process duplicate
    
    # Set processing state
    st.session_state.processing_message = True
    
    # Generate unique message ID
    message_id = str(uuid.uuid4())
    
    # Add user message
    user_msg = {
        "role": "user", 
        "content": user_message,
        "id": message_id,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.chat_messages.append(user_msg)
    
    # Get AI response
    try:
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.chatbot.chat_response(user_message, message_id)
    except Exception as e:
        response = f"I'm having trouble right now. Please try again in a moment. (Error: {str(e)[:50]}...)"
    
    # Add AI response
    ai_msg = {
        "role": "assistant", 
        "content": response,
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.chat_messages.append(ai_msg)
    
    # Clear processing state
    st.session_state.processing_message = False
    st.session_state.last_message_id = message_id
    
    # Rerun to show new messages
    st.rerun()

def reset_conversation():
    """Reset the conversation with proper cleanup"""
    st.session_state.chat_messages = []
    st.session_state.last_message_id = None
    st.session_state.processing_message = False
    st.session_state.chatbot_session_id = str(uuid.uuid4())
    
    if 'chatbot' in st.session_state:
        st.session_state.chatbot.reset_conversation()
    
    st.success("� Conversation reset! Ask me anything about your leads.")
    st.rerun()
