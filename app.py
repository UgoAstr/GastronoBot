import streamlit as st
import time

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="🍳 GastronoBot — Assistant Culinaire",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── Custom CSS ───────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&family=Playfair+Display:wght@600;700&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* ── Header ── */
    .chef-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .chef-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .chef-header p {
        font-family: 'DM Sans', sans-serif;
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0;
    }

    /* ── Chat bubbles ── */
    .stChatMessage {
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 16px !important;
        margin-bottom: 0.6rem !important;
    }

    /* ── Input bar ── */
    .stChatInput > div {
        border-radius: 24px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.06) !important;
    }
    .stChatInput textarea {
        font-family: 'DM Sans', sans-serif !important;
        color: #e2e8f0 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #16213e;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Streamlit top bar ── */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ── Divider ── */
    .subtle-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(246,211,101,0.3), transparent);
        margin: 0.5rem 4rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Header ───────────────────────────
st.markdown("""
<div class="chef-header">
    <h1>🍳 GastronoBot </h1>
    <p>Votre assistant culinaire intelligent — posez vos questions sur les recettes !</p>
</div>
<div class="subtle-divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")

    lang = st.radio(
        "Langue de conversation",
        ["🇫🇷 Français", "🇬🇧 English"],
        index=0,
    )

    top_k = st.slider(
        "Nombre de documents (top_k)",
        min_value=1,
        max_value=10,
        value=2,
        help="Nombre d'extraits récupérés par la recherche hybride. (on peut tourner seulemeent avec K de 1 à 3 pour le moment à cause de la limite de token du LLM)",
    )

    st.markdown("---")
    st.markdown(
        "**Comment ça marche ?**\n\n"
        "1. Votre question est convertie en embedding\n"
        "2. Recherche hybride (sémantique + BM25)\n"
        "3. Les extraits pertinents sont envoyés au LLM\n"
        "4. Le LLM génère une réponse sourcée"
    )

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────── Load RAG backend ───────────────────────────
# On importe le backend RAG une seule fois grâce au cache
@st.cache_resource(show_spinner="Chargement du modèle et des données…")
def load_rag_backend():
    """Charge les modèles, embeddings et index BM25."""
    from rag_backend import generation_gorq, generation_gorq_fr
    return generation_gorq, generation_gorq_fr

try:
    generation_gorq, generation_gorq_fr = load_rag_backend()
    rag_ready = True
except Exception as e:
    rag_ready = False
    st.error(f"⚠️ Impossible de charger le backend RAG : {e}")

# ─────────────────────────── Chat state ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    avatar = "🍳" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ─────────────────────────── Chat input ───────────────────────────
if prompt := st.chat_input("Posez votre question culinaire…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🍳"):
        if not rag_ready:
            response = "❌ Le backend RAG n'est pas disponible. Vérifiez que tous les fichiers sont présents."
            st.markdown(response)
        else:
            with st.spinner("Recherche en cours… 🔍"):
                try:
                    is_french = lang.startswith("🇫🇷")
                    if is_french:
                        response = generation_gorq_fr(prompt, top=top_k)
                    else:
                        response = generation_gorq(prompt, top=top_k)
                except Exception as e:
                    response = f"❌ Erreur lors de la génération : {e}"

            # Streaming-style display
            placeholder = st.empty()
            displayed = ""
            for char in response:
                displayed += char
                placeholder.markdown(displayed + "▌")
                time.sleep(0.008)
            placeholder.markdown(displayed)

    st.session_state.messages.append({"role": "assistant", "content": response})
