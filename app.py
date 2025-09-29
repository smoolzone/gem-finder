import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json
from deep_translator import GoogleTranslator
from typing import Dict, List
import time
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# Initialize OpenRouter client
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"] 

llama_model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.3-70b-instruct:free",
    temperature=0.5,
)

# Initialize DuckDuckGo search tool - lighter config
wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)  # Reduced for less load
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

# Translation setup
translator_en = GoogleTranslator(source='de', target='en')
translator_de = GoogleTranslator(source='en', target='de')

# Default UI texts
UI_TEXTS = {
    'en': {
        'title': 'Local Goodies: Discover Positive Vibes by City',
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
        'error_search': 'Search failed. Check your connection.',
        'search_rate_limit': 'Search service is temporarily unavailable due to rate limits. Please try again in 5-10 minutes.'
    },
    'de': {
        'title': 'Lokale Highlights: Entdecke positive Vibes in deiner Stadt',
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
        'error_search': 'Suche fehlgeschlagen. Überprüfe deine Verbindung.',
        'search_rate_limit': 'Suchdienst ist vorübergehend aufgrund von Ratenlimits nicht verfügbar. Bitte versuche es in 5-10 Minuten erneut.'
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

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=4, min=10, max=60))
def search_web(query: str, lang: str) -> List[Dict]:
    """Search DuckDuckGo with longer exponential backoff."""
    try:
        results_str = web_search_tool.invoke(query)
        # Simple parsing for SearchRun output
        parsed_results = []
        lines = results_str.split('\n')
        for i in range(0, len(lines), 3):  # Assume title, snippet, url pattern
            if i + 2 < len(lines):
                title = lines[i].strip()
                body = lines[i+1].strip()
                href = lines[i+2].strip() if lines[i+2] else ""
                if title:
                    parsed_results.append({"title": title[:50] + "..." if len(title) > 50 else title, "href": href, "body": body})
        return parsed_results[:3]
    except Exception as e:
        # Check for rate limit in error
        if "ratelimit" in str(e).lower() or "202" in str(e):
            raise Exception("Rate limit hit")
        st.error(f"{get_ui_texts(lang)['error_search']} Error: {e}")
        raise

def fallback_summary(user_profile: Dict, prompt: str, lang: str) -> str:
    """LLM-generated suggestions without search (fallback)."""
    profile_str = json.dumps(user_profile)
    language_full = 'English' if lang == 'en' else 'German'
    system_prompt = f"""
    You are Grok, witty and helpful. Generate 3-5 personalized outdoor activity suggestions for tonight in {user_profile['city']}.
    User profile: {profile_str}. Prompt: {prompt}.
    Focus on positive, local vibes. Assume current date is September 28, 2025.
    Output in {language_full}: Bullet points, engaging tone. Include made-up but realistic links/descriptions.
    """
    try:
        response = llama_model.invoke([{"role": "system", "content": system_prompt}])
        return response.content.strip()
    except Exception as e:
        st.error(f"{get_ui_texts(lang)['error_openrouter']} Error: {e}")
        return "Suggestions could not be generated."

def summarize_results(results: List[Dict], user_profile: Dict, prompt: str, lang: str) -> str:
    """Use OpenRouter to summarize search results into personalized suggestions."""
    results_str = json.dumps(results[:3])  # Limit to top 3 for brevity
    profile_str = json.dumps(user_profile)
    language_full = 'English' if lang == 'en' else 'German'
    system_prompt = f"""
    You are Grok, witty and helpful. Summarize these web search results into 3-5 personalized suggestions.
    User profile: {profile_str}. Prompt: {prompt}.
    Focus on positive, local vibes (restaurants, concerts, etc.). Include links, brief descriptions.
    Output in {language_full}: Bullet points, engaging tone.
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

    # Sidebar for toggles
    with st.sidebar:
        lang = st.selectbox(
            get_ui_texts('en')['lang_toggle'],
            ['en', 'de'],
            format_func=lambda x: get_ui_texts('en' if x == 'en' else 'de')[x],
            key="lang_select"
        )
        ui_texts = get_ui_texts(lang)

        # Detect language change and translate interests
        interests_map = {
            'en_to_de': {
                'Restaurants': 'Restaurants',
                'Concerts': 'Konzerte',
                'Pubs': 'Pubs',
                'Gatherings': 'Treffen',
                'Events': 'Events',
                'Outdoor Activities': 'Outdoor-Aktivitäten'
            }
        }
        interests_map['de_to_en'] = {v: k for k, v in interests_map['en_to_de'].items()}
        if 'previous_lang' not in st.session_state:
            st.session_state.previous_lang = lang
        if st.session_state.previous_lang != lang:
            direction = 'en_to_de' if lang == 'de' else 'de_to_en'
            st.session_state.profile['interests'] = [interests_map[direction].get(i, i) for i in st.session_state.profile['interests']]
            st.session_state.previous_lang = lang

        theme = st.selectbox(
            ui_texts['mode_toggle'],
            ['dark', 'light'],
            format_func=lambda x: ui_texts[x],
            key="theme_select"
        )

    # Apply theme CSS
    css = GROK_CSS
    if theme == 'light':
        css = GROK_CSS.replace('#0f0f23', '#ffffff').replace('#1a1a2e', '#f0f0f0').replace('#0f3460', '#e0e0ff').replace('#16213e', '#cccccc').replace('#00d4ff', '#0066cc')
    st.markdown(css, unsafe_allow_html=True)

    # Header
    st.title(ui_texts['title'])
    st.markdown(f"*{ui_texts['subtitle']}*")

    # Session state for profile
    if 'profile' not in st.session_state:
        st.session_state.profile = {'city': '', 'interests': []}

    # Profile Form with accessibility fixes
    with st.form(key='profile_form'):
        st.markdown(f"### {ui_texts['profile_title']}")
        
        # City input
        city = st.text_input(
            ui_texts['city_label'],
            value=st.session_state.profile['city'],
            key="city_input",
            placeholder=ui_texts['city_label'],
            help="Enter your city (e.g., Berlin)"
        )

        # Interests multiselect
        interests = st.multiselect(
            ui_texts['interests_label'],
            ui_texts['interests_options'],
            default=st.session_state.profile['interests'],
            key="interests_select",
            help="Select one or more interests"
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
            prompt = st.text_input(
                ui_texts['prompt_label'],
                key="prompt_input",
                placeholder=ui_texts['prompt_label'],
                help="Enter what you're in the mood for (e.g., cozy dinner spot)"
            )
            search = st.form_submit_button(ui_texts['search_button'])
            if search and prompt:
                with st.spinner('Grok is searching the web...'):
                    # Step 1: Generate refined query
                    search_query = generate_search_query(st.session_state.profile, prompt, lang)
                    st.info(f"🔍 Refined Query: {search_query}")

                    # Step 2: Web search with fallback
                    results = []
                    use_fallback = False
                    try:
                        results = search_web(search_query, lang)
                    except RetryError as re:
                        if "ratelimit" in str(re).lower():
                            st.warning(ui_texts['search_rate_limit'])
                            use_fallback = True
                        
                    except Exception as e:
                        st.error(f"Unexpected search error: {e}")
                        use_fallback = True

                    # Step 3: Summarize (with fallback)
                    if use_fallback or not results:
                        summary = fallback_summary(st.session_state.profile, prompt, lang)
                    else:
                        summary = summarize_results(results, st.session_state.profile, prompt, lang)
                    
                    st.subheader(ui_texts['results_title'])
                    st.markdown(summary)
    else:
        st.info("Set your profile first to get started!")

if __name__ == "__main__":
    main()