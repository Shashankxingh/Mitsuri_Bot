import os
import asyncio
import threading
import logging
from datetime import datetime
import pytz 

# NEW: Import Search Tool
from duckduckgo_search import DDGS

from dotenv import load_dotenv
from flask import Flask
from pymongo import MongoClient
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest

# AI Clients
from groq import AsyncGroq
from openai import AsyncOpenAI 

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mitsuri")
logging.getLogger("werkzeug").disabled = True
logging.getLogger("httpx").setLevel(logging.WARNING)

# ================= SETUP =================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
OWNER_ID = 8162412883 # Replace with your ID if different

# ================= WEB SERVER =================
app = Flask(__name__)
@app.route("/")
def home(): return "Mitsuri Agent Online 🌸"

def run_web():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

threading.Thread(target=run_web, daemon=True).start()

# ================= CONFIG =================
HISTORY_LIMIT = 10 
IST = pytz.timezone('Asia/Kolkata')

# Base Persona
BASE_SYSTEM_PROMPT = (
    "You are Mitsuri Kanroji from Demon Slayer. "
    "Personality: warm, cheerful, romantic, sweet. "
    "Speak in Hinglish (Hindi + English). "
    "Keep replies short, cute, friendly. "
    "Use emojis sparingly (🌸💖🍡)."
)

AI_MODELS = {
    "groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "cerebras": ["llama3.1-8b", "llama3.1-70b"], 
    "sambanova": ["Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-70B-Instruct"],
}

AI_PROVIDER = "groq"
AI_MODEL = "llama-3.1-8b-instant"

# ================= DATABASE =================
mongo = MongoClient(MONGO_URI)
db = mongo["mitsuri"]
users = db["users"]

def get_history(chat_id):
    data = users.find_one({"chat_id": chat_id}, {"history": 1})
    return data["history"] if data and "history" in data else []

def update_history(chat_id, role, content):
    users.update_one(
        {"chat_id": chat_id},
        {"$push": {"history": {"$each": [{"role": role, "content": content}], "$slice": -HISTORY_LIMIT}}},
        upsert=True
    )

# ================= AI CLIENTS =================
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
cerebras_client = AsyncOpenAI(api_key=os.getenv("CEREBRAS_API_KEY"), base_url="https://api.cerebras.ai/v1")
sambanova_client = AsyncOpenAI(api_key=os.getenv("SAMBANOVA_API_KEY"), base_url="https://api.sambanova.ai/v1")

# ================= 🔍 THE SEARCH TOOL =================

def search_web(query):
    """Real-time search using DuckDuckGo"""
    try:
        # Get top 3 results
        results = DDGS().text(query, max_results=3)
        if not results: return None
        
        # Format results into a clean string
        formatted = "\n".join([f"Source: {r['title']}\nSummary: {r['body']}\nLink: {r['href']}\n" for r in results])
        return formatted
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return None

# ================= 🧠 THE AGENT BRAIN =================

async def get_response(messages, model, temperature=0.7):
    """Helper to call the selected AI provider"""
    try:
        if AI_PROVIDER == "groq":
            r = await groq_client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=1000)
            return r.choices[0].message.content
        elif AI_PROVIDER == "cerebras":
            r = await cerebras_client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=1000)
            return r.choices[0].message.content
        elif AI_PROVIDER == "sambanova":
            r = await sambanova_client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=1000)
            return r.choices[0].message.content
    except Exception as e:
        logger.error(f"AI Error: {e}")
        return None

async def ask_ai_agent(chat_id: int, user_prompt: str):
    # 1. Get Conversation History
    history = get_history(chat_id)
    
    # 2. DECISION STEP (Router)
    # We ask a fast/cheap model if we need to search
    # If the user says "Hi", we don't search. If "Bitcoin price", we search.
    
    router_prompt = (
        f"User said: '{user_prompt}'. "
        "Does this require searching the internet for real-time info (news, weather, sports, prices, specific facts)? "
        "Reply ONLY with 'SEARCH' or 'CHAT'."
    )
    
    # Use Groq 8b for routing (it's fastest)
    try:
        decision = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": router_prompt}],
            max_tokens=5
        )
        intent = decision.choices[0].message.content.strip().upper()
    except:
        intent = "CHAT" # Fallback

    logger.info(f"Intent: {intent} | Prompt: {user_prompt}")

    # 3. TOOL STEP (If Search is needed)
    search_context = ""
    if "SEARCH" in intent:
        await asyncio.sleep(0.5) # Simulate thinking
        raw_search = await asyncio.to_thread(search_web, user_prompt)
        
        if raw_search:
            search_context = (
                f"\n\n[🔍 WEB SEARCH RESULTS - USE THIS DATA TO ANSWER]:\n{raw_search}\n"
                f"[End of Search Data]\n"
                f"Note: Synthesize this info. If user asks for links, provide them."
            )
    
    # 4. FINAL ANSWER STEP
    # Dynamic Date/Time
    now = datetime.now(IST)
    time_str = now.strftime("%Y-%m-%d %I:%M %p IST")
    
    system_instruction = BASE_SYSTEM_PROMPT + f"\n[Current Time: {time_str}]" + search_context

    final_messages = [{"role": "system", "content": system_instruction}] + history + [{"role": "user", "content": user_prompt}]
    
    response = await get_response(final_messages, AI_MODEL)
    
    if not response:
        return "Ahh~ Network issue lag raha hai 🥺 try again!"

    # 5. Save Memory
    update_history(chat_id, "user", user_prompt)
    update_history(chat_id, "assistant", response)

    return response

# ================= TELEGRAM HANDLERS =================

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text: return
    
    # Basic filters
    is_private = update.effective_chat.type == "private"
    is_reply = msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id
    if not (is_private or "mitsuri" in msg.text.lower() or is_reply): return

    # Typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Agent Logic
    reply = await ask_ai_agent(update.effective_chat.id, msg.text)
    await msg.reply_text(reply)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.update_one({"chat_id": update.effective_chat.id}, {"$set": {"history": []}})
    await update.message.reply_text("🌸 Mitsuri Online! Ask me anything (I can search the web too!) 💖")

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID: return
    kb = [[InlineKeyboardButton(p.upper(), callback_data=f"prov:{p}")] for p in AI_MODELS]
    await update.message.reply_text("Select Provider:", reply_markup=InlineKeyboardMarkup(kb))

async def ai_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_PROVIDER, AI_MODEL
    query = update.callback_query
    await query.answer()
    if query.from_user.id != OWNER_ID: return
    
    data = query.data
    if data.startswith("prov:"):
        prov = data.split(":")[1]
        kb = [[InlineKeyboardButton(m, callback_data=f"model:{prov}:{m}")] for m in AI_MODELS[prov]]
        await query.message.edit_text(f"Models for {prov}:", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("model:"):
        _, prov, mod = data.split(":", 2)
        AI_PROVIDER = prov; AI_MODEL = mod
        await query.message.edit_text(f"✅ Active: {prov} / {mod}")

# ================= MAIN =================

def main():
    request = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app_bot = ApplicationBuilder().token(BOT_TOKEN).request(request).build()

    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("ai", ai_cmd))
    app_bot.add_handler(CallbackQueryHandler(ai_buttons))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("🌸 Mitsuri Agent (Search Enabled) is running...")
    app_bot.run_polling()

if __name__ == "__main__":
    main()
