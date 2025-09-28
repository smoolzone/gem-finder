import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import json
from deep_translator import GoogleTranslator
from typing import Dict, List
import os
from datetime import datetime, timedelta
from dateutil.parser import parse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Initialize session state for profile
if 'profile' not in st.session_state:
    st.session_state.profile = {'city': '', 'interests': []}

# Initialize OpenRouter client
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")
llama_model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0.5,
)

# Initialize DuckDuckGo search tool
web_search_tool = DuckDuckGoSearchResults(max_results=5)

# Translation setup
translator_en = GoogleTranslator(source='de', target='en')
translator_de = GoogleTranslator(source='en', target='de')

# Default UI texts
UI_TEXTS = {
    'en': {
        'title': 'GemFinder: Discover Positive Vibes by City',
        'subtitle': 'Find restaurants, concerts, pubs, gatherings & more tailored to you.',
        'profile_title': 'Your Profile',
        'city_label': 'Your City',
        'interests_label': 'Interests (select all that apply)',
        'interests_options': ['Restaurants', 'Concerts', 'Pubs', 'Gatherings', 'Events', 'Outdoor Activities'],
        'submit_profile': 'Save Profile',
        'prompt_label': 'What are you in the mood for? (e.g., "cozy dinner spot")',
        'search_button': 'Search with GemFinder AI',
        'results_title': 'Your Personalized Suggestions',
        'no_results': 'No suggestions found. Try a different prompt or check your connection!',
        'mode_toggle': 'Theme',
        'lang_toggle': 'Language',
        'dark': 'Dark (GemFinder Style)', 'light': 'Light',
        'en': 'English', 'de': 'German',
        'error_openrouter': 'OpenRouter API error. Please check your API key or try again later.',
        'error_search': 'Search failed. Please check your connection or try a different prompt.'
    },
    'de': {
        'title': 'GemFinder: Entdecke positive Vibes in deiner Stadt',
        'subtitle': 'Finde Restaurants, Konzerte, Pubs, Treffen & mehr, maßgeschneidert für dich.',
        'profile_title': 'Dein Profil',
        'city_label': 'Deine Stadt',
        'interests_label': 'Interessen (wähle alle zutreffenden aus)',
        'interests_options': ['Restaurants', 'Konzerte', 'Pubs', 'Treffen', 'Events', 'Outdoor-Aktivitäten'],
        'submit_profile': 'Profil speichern',
        'prompt_label': 'Worauf hast du Lust? (z.B. "gemütliches Abendessen")',
        'search_button': 'Mit GemFinder AI suchen',
        'results_title': 'Deine personalisierten Vorschläge',
        'no_results': 'Keine Vorschläge gefunden. Versuche einen anderen Prompt oder überprüfe deine Verbindung!',
        'mode_toggle': 'Thema',
        'lang_toggle': 'Sprache',
        'dark': 'Dunkel (GemFinder-Stil)', 'light': 'Hell',
        'en': 'Englisch', 'de': 'Deutsch',
        'error_openrouter': 'OpenRouter API-Fehler. Überprüfe deinen API-Schlüssel oder versuche es später erneut.',
        'error_search': 'Suche fehlgeschlagen. Überprüfe deine Verbindung oder versuche einen anderen Prompt.'
    }
}

# Custom CSS with minimalist, chic design
GROK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    :root {
        --bg-color: #2e2e2e;
        --text-color: #e0e0e0;
        --input-bg: #3a3a3a;
        --input-border: #e63946;
        --button-bg: #e63946;
        --button-hover: #d00000;
        --header-color: #e63946;
        --container-bg: #3a3a3a;
        --shadow: 0 3px 8px rgba(230, 57, 70, 0.2);
        --accent-color: #f4a261;
    }

    .light-mode {
        --bg-color: #ffffff;
        --text-color: #333333;
        --input-bg: #f5f5f5;
        --input-border: #d00000;
        --button-bg: #d00000;
        --button-hover: #a30000;
        --header-color: #d00000;
        --container-bg: #f5f5f5;
        --shadow: 0 3px 8px rgba(208, 0, 0, 0.15);
        --accent-color: #e76f51;
    }

    .main, .stApp {
        background: var(--bg-color);
        color: var(--text-color);
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
    }

    h1 {
        color: var(--header-color);
        font-size: 2.2em;
        font-weight: 600;
        text-align: center;
        animation: fade-in 1.5s ease-in-out;
    }

    @keyframes fade-in {
        0% { opacity: 0; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }

    .stTextInput > div > div > input {
        background: var(--input-bg);
        color: var(--text-color);
        border: 1px solid var(--input-border);
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color);
        box-shadow: var(--shadow);
    }

    .stSelectbox > div > div > select {
        background: var(--input-bg);
        color: var(--text-color);
        border: 1px solid var(--input-border);
        border-radius: 8px;
        padding: 10px;
    }

    .stButton > button {
        background: var(--button-bg);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }

    .stButton > button:hover {
        background: var(--button-hover);
        transform: translateY(-1px);
        box-shadow: 0 5px 12px rgba(230, 57, 70, 0.3);
    }

    .stForm {
        background: var(--container-bg);
        padding: 20px;
        border-radius: 12px;
        box-shadow: var(--shadow);
        margin-bottom: 15px;
    }

    .sidebar .stSelectbox {
        margin-bottom: 20px;
    }

    label {
        font-weight: 600;
        color: var(--text-color);
    }

    .stSpinner > div {
        border-color: var(--header-color) transparent transparent transparent !important;
    }

    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        background: var(--container-bg);
        color: var(--text-color);
        box-shadow: var(--shadow);
    }

    .stMarkdown p {
        font-size: 1em;
        line-height: 1.4;
    }
</style>
"""

def translate_text(text: str, lang: str) -> str:
    """Translate text based on current language."""
    if lang == 'de':
        return translator_de.translate(text) if text in UI_TEXTS['en'] else text
    return text

def get_ui_texts(lang: str) -> Dict:
    """Get translated UI texts."""
    base = UI_TEXTS['en'].copy()
    for key, val in UI_TEXTS[lang].items():
        base[key] = val
    return base

def generate_search_query(user_profile: Dict, prompt: str, lang: str) -> str:
    """Use OpenRouter to generate a refined search query."""
    profile_str = json.dumps(user_profile)
    # Parse and correct date if possible
    try:
        input_date = parse(prompt, fuzzy=True, default=datetime(2025, 9, 1))
        if "friday" in prompt.lower():
            # Find the closest Friday
            days_to_friday = (4 - input_date.weekday()) % 7
            corrected_date = input_date + timedelta(days=days_to_friday)
            prompt = prompt.replace("September 29 2025", corrected_date.strftime("%B %d %Y"))
    except ValueError:
        pass  # Fallback to original prompt if date parsing fails
    system_prompt = f"""
    You are GemFinder, a helpful AI. User profile: {profile_str}.
    User prompt: {prompt}.
    Generate a concise DuckDuckGo search query for local positive events/places in the user's city.
    Focus on interests: {', '.join(user_profile['interests'])}.
    Make it specific, e.g., "best cozy restaurants in Berlin concerts September 2025".
    Respond with ONLY the search query string.
    """
    try:
        response = llama_model.invoke([{"role": "system", "content": system_prompt}])
        return response.content.strip()
    except Exception as e:
        st.error(f"{get_ui_texts(lang)['error_openrouter']} Error: {e}")
        return f"{prompt} in {user_profile['city']} September 2025"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def search_web(query: str, lang: str) -> List[Dict]:
    """Search DuckDuckGo using LangChain's DuckDuckGoSearchResults with retries."""
    try:
        results = web_search_tool.invoke(query)
        parsed_results = []
        try:
            results_list = json.loads(results) if isinstance(results, str) else results
            for result in results_list[:5]:
                parsed_results.append({
                    "title": result.get("title", "No title"),
                    "href": result.get("link", ""),
                    "body": result.get("snippet", "")
                })
        except json.JSONDecodeError:
            for line in results.split('\n')[:5]:
                if line.strip():
                    parsed_results.append({
                        "title": line[:50] + "...",
                        "href": "",
                        "body": line
                    })
        return parsed_results
    except Exception as e:
        st.error(f"{get_ui_texts(lang)['error_search']} Error: {e}")
        return []

def summarize_results(results: List[Dict], user_profile: Dict, prompt: str, lang: str) -> str:
    """Use OpenRouter to summarize search results into personalized suggestions."""
    results_str = json.dumps(results[:3])
    profile_str = json.dumps(user_profile)
    system_prompt = f"""
    You are GemFinder, witty and helpful. Summarize these web search results into 3-5 personalized suggestions.
    User profile: {profile_str}. Prompt: {prompt}.
    Focus on positive, local vibes (restaurants, concerts, etc.). Include links, brief descriptions.
    Output in {lang.upper()}: Bullet points, engaging tone.
    If no specific events are found, suggest general venues or activities matching the prompt.
    """
    try:
        response = llama_model.invoke([{"role": "system", "content": system_prompt + "\nResults: " + results_str}])
        return response.content.strip()
    except Exception as e:
        st.error(f"{get_ui_texts(lang)['error_openrouter']} Error: {e}")
        return f"No specific events found for '{prompt}'. Try exploring popular music venues in {user_profile['city']} like MEO Arena or LAV - Lisboa Ao Vivo."

# Streamlit App
def main():
    st.set_page_config(page_title="GemFinder", layout="wide")
    st.markdown(GROK_CSS, unsafe_allow_html=True)

    # Initialize session state for theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'

    # Sidebar for toggles
    with st.sidebar:
        lang = st.selectbox(
            get_ui_texts('en')['lang_toggle'],
            ['en', 'de'],
            format_func=lambda x: get_ui_texts('en' if x == 'en' else 'de')[x],
            key="lang_select",
            label="Select Language"
        )
        ui_texts = get_ui_texts(lang)
        st.session_state.theme = st.selectbox(
            ui_texts['mode_toggle'],
            ['dark', 'light'],
            format_func=lambda x: ui_texts[x],
            key="theme_select",
            label="Select Theme",
            on_change=lambda: st.markdown(f'<div class="{st.session_state.theme}-mode"></div>', unsafe_allow_html=True)
        )

    # Header with animated title
    st.markdown(f'<h1>{ui_texts["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-style: italic;">{ui_texts["subtitle"]}</p>', unsafe_allow_html=True)

    # Profile Form
    with st.form(key='profile_form'):
        st.markdown(f'<h3 style="color: var(--header-color);">{ui_texts["profile_title"]}</h3>', unsafe_allow_html=True)
        
        st.markdown(f'<label for="city_input">{ui_texts["city_label"]}</label>', unsafe_allow_html=True)
        city = st.text_input(
            label=ui_texts["city_label"],
            value=st.session_state.profile['city'],
            key="city_input",
            placeholder=ui_texts['city_label'],
            help="Enter your city (e.g., Lisbon)",
            label_visibility="collapsed"
        )

        st.markdown(f'<label for="interests_select">{ui_texts["interests_label"]}</label>', unsafe_allow_html=True)
        interests = st.multiselect(
            label=ui_texts["interests_label"],
            options=ui_texts['interests_options'],
            default=st.session_state.profile['interests'],
            key="interests_select",
            help="Select one or more interests",
            label_visibility="collapsed"
        )

        submit = st.form_submit_button(ui_texts['submit_profile'])
        if submit:
            if not city:
                st.error("Please enter a city.")
            else:
                st.session_state.profile = {'city': city.strip(), 'interests': interests}
                st.success("Profile saved! 🎉")

    # Main Search
    if st.session_state.profile['city']:
        with st.form(key='search_form'):
            st.markdown(f'<label for="prompt_input">{ui_texts["prompt_label"]}</label>', unsafe_allow_html=True)
            prompt = st.text_input(
                label=ui_texts["prompt_label"],
                key="prompt_input",
                placeholder=ui_texts['prompt_label'],
                help="Enter what you're in the mood for (e.g., live music next Friday)",
                label_visibility="collapsed"
            )
            search = st.form_submit_button(ui_texts['search_button'])
            if search and prompt:
                with st.spinner('GemFinder is searching the web...'):
                    search_query = generate_search_query(st.session_state.profile, prompt, lang)
                    st.info(f"🔍 Refined Query: {search_query}")

                    results = search_web(search_query, lang)

                    if results:
                        summary = summarize_results(results, st.session_state.profile, prompt, lang)
                        st.subheader(ui_texts['results_title'])
                        st.markdown(summary)
                    else:
                        st.warning(ui_texts['no_results'])
    else:
        st.info("Set your profile first to get started!")

if __name__ == "__main__":
    main()