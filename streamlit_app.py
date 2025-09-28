import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import json
from deep_translator import GoogleTranslator
from typing import Dict, List
import os

# Initialize OpenRouter client
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
llama_model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0.5,
)

# Initialize DuckDuckGo search tool
web_search_tool = DuckDuckGoSearchResults(max_results=5)  # Structured JSON output, max 5 results

# Translation setup
translator_en = GoogleTranslator(source='de', target='en')
translator_de = GoogleTranslator(source='en', target='de')

# Default UI texts
UI_TEXTS = {
    'en': {
        'title': 'Grok Local Goodies: Discover Positive Vibes by City',
        'subtitle': 'Find restaurants, concerts, pubs, gatherings & more tailored to you.',
        'profile_title': 'Your Profile',
        'city_label': 'Your City',
        'interests_label': 'Interests (select all that apply)',
        'interests_options': ['Restaurants', 'Concerts', 'Pubs', 'Gatherings', 'Events', 'Outdoor Activities'],
        'submit_profile': 'Save Profile',
        'prompt_label': 'What are you in the mood for? (e.g., "cozy dinner spot")',
        'search_button': 'Search with Grok AI',
        'results_title': 'Your Personalized Suggestions',
        'no_results': 'No suggestions found. Try a different prompt!',
        'mode_toggle': 'Theme',
        'lang_toggle': 'Language',
        'dark': 'Dark (Grok Style)', 'light': 'Light',
        'en': 'English', 'de': 'German',
        'error_openrouter': 'OpenRouter API error. Check your API key or connection.',
        'error_search': 'Search failed. Check your connection.'
    },
    'de': {
        'title': 'Grok Lokale Highlights: Entdecke positive Vibes in deiner Stadt',
        'subtitle': 'Finde Restaurants, Konzerte, Pubs, Treffen & mehr, maßgeschneidert für dich.',
        'profile_title': 'Dein Profil',
        'city_label': 'Deine Stadt',
        'interests_label': 'Interessen (wähle alle zutreffenden aus)',
        'interests_options': ['Restaurants', 'Konzerte', 'Pubs', 'Treffen', 'Events', 'Outdoor-Aktivitäten'],
        'submit_profile': 'Profil speichern',
        'prompt_label': 'Worauf hast du Lust? (z.B. "gemütliches Abendessen")',
        'search_button': 'Mit Grok AI suchen',
        'results_title': 'Deine personalisierten Vorschläge',
        'no_results': 'Keine Vorschläge gefunden. Versuche einen anderen Prompt!',
        'mode_toggle': 'Thema',
        'lang_toggle': 'Sprache',
        'dark': 'Dunkel (Grok-Stil)', 'light': 'Hell',
        'en': 'Englisch', 'de': 'Deutsch',
        'error_openrouter': 'OpenRouter API-Fehler. Überprüfe deinen API-Schlüssel oder die Verbindung.',
        'error_search': 'Suche fehlgeschlagen. Überprüfe deine Verbindung.'
    }
}

# Custom CSS for Grok look
GROK_CSS = """
<style>
    .main { background-color: #0f0f23; color: #ffffff; }
    .stApp { background-color: #0f0f23; }
    .stTextInput > div > div > input { background-color: #1a1a2e; color: #ffffff; border: 1px solid #16213e; }
    .stSelectbox > div > div > select { background-color: #1a1a2e; color: #ffffff; border: 1px solid #16213e; }
    .stButton > button { background-color: #0f3460; color: #ffffff; border: none; }
    .stButton > button:hover { background-color: #16213e; }
    h1 { color: #00d4ff; font-family: 'Arial Black', sans-serif; }
    .light-mode .main { background-color: #ffffff; color: #000000; }
    .light-mode .stTextInput > div > div > input { background-color: #f0f0f0; color: #000000; }
    .light-mode .stSelectbox > div > div > select { background-color: #f0f0f0; color: #000000; }
    .light-mode .stButton > button { background-color: #e0e0ff; color: #000000; }
    .light-mode h1 { color: #0066cc; }
    label { font-weight: bold; }
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
    system_prompt = f"""
    You are Grok, a helpful AI. User profile: {profile_str}.
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
        return f"{prompt} in {user_profile['city']}"

def search_web(query: str) -> List[Dict]:
    """Search DuckDuckGo using LangChain's DuckDuckGoSearchResults."""
    try:
        results = web_search_tool.invoke(query)
        # DuckDuckGoSearchResults returns a JSON-like string; parse it
        parsed_results = []
        try:
            results_list = json.loads(results) if isinstance(results, str) else results
            for result in results_list[:5]:  # Limit to 5 results
                parsed_results.append({
                    "title": result.get("title", "No title"),
                    "href": result.get("link", ""),
                    "body": result.get("snippet", "")
                })
        except json.JSONDecodeError:
            # Fallback: treat as string and split
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
    results_str = json.dumps(results[:3])  # Limit to top 3 for brevity
    profile_str = json.dumps(user_profile)
    system_prompt = f"""
    You are Grok, witty and helpful. Summarize these web search results into 3-5 personalized suggestions.
    User profile: {profile_str}. Prompt: {prompt}.
    Focus on positive, local vibes (restaurants, concerts, etc.). Include links, brief descriptions.
    Output in {lang.upper()}: Bullet points, engaging tone.
    """
    try:
        response = llama_model.invoke([{"role": "system", "content": system_prompt + "\nResults: " + results_str}])
        return response.content.strip()
    except Exception as e:
        st.error(f"{get_ui_texts(lang)['error_openrouter']} Error: {e}")
        return "Suggestions could not be generated."

# Streamlit App
def main():
    st.set_page_config(page_title="Grok Local Goodies", layout="wide")
    st.markdown(GROK_CSS, unsafe_allow_html=True)

    # Sidebar for toggles
    with st.sidebar:
        lang = st.selectbox(
            get_ui_texts('en')['lang_toggle'],
            ['en', 'de'],
            format_func=lambda x: get_ui_texts('en' if x == 'en' else 'de')[x],
            key="lang_select"
        )
        ui_texts = get_ui_texts(lang)
        theme = st.selectbox(
            ui_texts['mode_toggle'],
            ['dark', 'light'],
            format_func=lambda x: ui_texts[x],
            key="theme_select"
        )
        if theme == 'light':
            st.markdown('<style>.light-mode {}</style>'.format(GROK_CSS.replace('0f0f23', 'ffffff').replace('#1a1a2e', 'f0f0f0').replace('#0f3460', 'e0e0ff').replace('#16213e', 'cccccc').replace('#00d4ff', '#0066cc')), unsafe_allow_html=True)

    # Header
    st.title(ui_texts['title'])
    st.markdown(f"*{ui_texts['subtitle']}*")

    # Session state for profile
    if 'profile' not in st.session_state:
        st.session_state.profile = {'city': '', 'interests': []}

    # Profile Form with accessibility fixes
    with st.form(key='profile_form'):
        st.markdown(f"### {ui_texts['profile_title']}")
        
        # City input with accessibility
        st.markdown(f'<label for="city_input">{ui_texts["city_label"]}</label>', unsafe_allow_html=True)
        city = st.text_input(
            label=ui_texts["city_label"],
            value=st.session_state.profile['city'],
            key="city_input",
            placeholder=ui_texts['city_label'],
            help="Enter your city (e.g., Berlin)",
            label_visibility="collapsed"
        )

        # Interests multiselect with accessibility
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
                help="Enter what you're in the mood for (e.g., cozy dinner spot)",
                label_visibility="collapsed"
            )
            search = st.form_submit_button(ui_texts['search_button'])
            if search and prompt:
                with st.spinner('Grok is searching the web...'):
                    # Step 1: Generate refined query
                    search_query = generate_search_query(st.session_state.profile, prompt, lang)
                    st.info(f"🔍 Refined Query: {search_query}")

                    # Step 2: Web search
                    results = search_web(search_query)

                    # Step 3: Summarize
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