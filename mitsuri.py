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
from telegram.error import Forbidden, TimedOut, NetworkError
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

# Silence noisy libraries
logging.getLogger("werkzeug").disabled = True
logging.getLogger("httpx").setLevel(logging.WARNING)

# ================= LOAD ENV =================
load_dotenv()

# ⚠️ FILL THESE IN YOUR .ENV FILE
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
OWNER_ID = 8162412883  # Replace with your Telegram ID
ADMIN_GROUP_ID = -1002759296936 # Optional: For admin controls

# ================= FLASK (RENDER KEEPALIVE) =================
app = Flask(__name__)

@app.route("/")
def home():
    return "Mitsuri Agent (Search + Cast) Online 🌸"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_web, daemon=True).start()

# ================= CONFIGURATION =================
HISTORY_LIMIT = 10 
IST = pytz.timezone('Asia/Kolkata')

# Base Personality
BASE_SYSTEM_PROMPT = (
    "You are Mitsuri Kanroji from Demon Slayer. "
    "Personality: warm, cheerful, romantic, sweet. "
    "Speak in Hinglish (Hindi + English). "
    "Keep replies short, cute, friendly. "
    "Use emojis sparingly (🌸💖🍡)."
)

# AI Models (Corrected IDs)
AI_MODELS = {
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
    ],
    "cerebras": [
        "llama3.1-8b", 
        "llama3.1-70b", 
    ],
    "sambanova": [
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
    ],
}

# Default Provider
AI_PROVIDER = "groq"
AI_MODEL = "llama-3.1-8b-instant"

# ================= DATABASE =================
mongo = MongoClient(MONGO_URI)
db = mongo["mitsuri"]
users = db["users"]

def get_history(chat_id):
    """Fetch last 10 messages"""
    data = users.find_one({"chat_id": chat_id}, {"history": 1})
    return data["history"] if data and "history" in data else []

def update_history(chat_id, role, content):
    """Save message and trim history"""
    users.update_one(
        {"chat_id": chat_id},
        {
            "$push": {
                "history": {
                    "$each": [{"role": role, "content": content}],
                    "$slice": -HISTORY_LIMIT
                }
            }
        },
        upsert=True
    )

def save_user(update: Update):
    """Save user details for Broadcast"""
    try:
        users.update_one(
            {"chat_id": update.effective_chat.id},
            {"$set": {
                "first_name": update.effective_user.first_name,
                "username": update.effective_user.username
            }},
            upsert=True,
        )
    except:
        pass

# ================= AI CLIENTS =================
# 1. GROQ
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# 2. CEREBRAS (Via OpenAI Client for Stability)
cerebras_client = AsyncOpenAI(
    api_key=os.getenv("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)

# 3. SAMBANOVA (Via OpenAI Client for Stability)
sambanova_client = AsyncOpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1"
)

# ================= 🔍 SEARCH TOOL (PERPLEXITY LOGIC) =================

def search_web(query):
    """Searches DuckDuckGo and returns top 3 results"""
    try:
        logger.info(f"🔎 Searching for: {query}")
        results = DDGS().text(query, max_results=3)
        if not results: return None
        
        # Format results nicely with links
        formatted_data = ""
        for i, r in enumerate(results, 1):
            formatted_data += f"[{i}] {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}\n\n"
            
        return formatted_data
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return None

# ================= 🧠 AGENT LOGIC (THE BRAIN) =================

async def get_ai_response(messages, model, temperature=0.7):
    """Helper to call current provider"""
    try:
        client = None
        if AI_PROVIDER == "groq": client = groq_client
        elif AI_PROVIDER == "cerebras": client = cerebras_client
        elif AI_PROVIDER == "sambanova": client = sambanova_client

        r = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return r.choices[0].message.content
    except Exception as e:
        logger.error(f"AI Error ({AI_PROVIDER}): {e}")
        return None

async def ask_agent(chat_id: int, user_prompt: str):
    # 1. Get History
    history = get_history(chat_id)
    
    # 2. DECIDE: Do we need to search?
    # We use a fast model (Groq 8b) to act as the "Router"
    router_prompt = (
        f"User said: '{user_prompt}'. "
        "Does this require real-time info (news, weather, sports, stock prices, facts) that an AI wouldn't know? "
        "Reply ONLY with 'SEARCH' or 'CHAT'."
    )
    
    intent = "CHAT"
    try:
        # Quick router check
        decision = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": router_prompt}],
            max_tokens=5
        )
        intent = decision.choices[0].message.content.strip().upper()
    except:
        pass # If router fails, default to CHAT

    # 3. EXECUTE: Search if needed
    search_context = ""
    if "SEARCH" in intent:
        # Simulate typing action while searching
        await asyncio.sleep(0.5) 
        
        # Run search in background thread to not block bot
        raw_search_data = await asyncio.to_thread(search_web, user_prompt)
        
        if raw_search_data:
            search_context = (
                f"\n\n[🔍 SEARCH RESULTS]:\n{raw_search_data}\n"
                f"[INSTRUCTION]: Answer the user using these results. Cite the links if relevant."
            )

    # 4. FINAL ANSWER
    # Add Time & Date
    now = datetime.now(IST)
    time_str = now.strftime("%Y-%m-%d %I:%M %p IST")
    
    final_system_prompt = BASE_SYSTEM_PROMPT + f"\n[Current Time: {time_str}]" + search_context

    messages = [{"role": "system", "content": final_system_prompt}] + history + [{"role": "user", "content": user_prompt}]
    
    response = await get_ai_response(messages, AI_MODEL)
    
    if not response:
        return "Ahh~ Network issue lag raha hai 🥺 (Try again)"

    # 5. Save Memory
    update_history(chat_id, "user", user_prompt)
    update_history(chat_id, "assistant", response)

    return response

# ================= COMMANDS =================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    users.update_one({"chat_id": update.effective_chat.id}, {"$set": {"history": []}})
    await update.message.reply_text("🌸 Mitsuri Online! Ask me anything (I can search the web too!) 💖")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text: return
    
    # Filter: Only private chats OR mentions in groups
    is_private = update.effective_chat.type == "private"
    is_mentioned = "mitsuri" in msg.text.lower() or (msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id)
    
    if not (is_private or is_mentioned): return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    save_user(update)
    
    reply = await ask_agent(update.effective_chat.id, msg.text)
    
    # Auto-split long messages
    if len(reply) > 4000:
        for x in range(0, len(reply), 4000):
            await msg.reply_text(reply[x:x+4000])
    else:
        await msg.reply_text(reply)

# ================= RESTORED: /CAST (BROADCAST) =================

async def cast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcasts a message to all users in the DB"""
    if update.effective_user.id != OWNER_ID:
        return

    msg = " ".join(context.args)
    if not msg:
        await update.message.reply_text("❌ Usage: /cast <message>")
        return

    await update.message.reply_text("📣 Sending broadcast...")
    logger.info("Broadcast started")

    count = 0
    total = users.count_documents({})
    
    # Iterate through all users
    cursor = users.find({}, {"chat_id": 1})
    
    for u in cursor:
        chat_id = u.get("chat_id")
        if not chat_id: continue
        
        try:
            await context.bot.send_message(chat_id, msg)
            count += 1
            await asyncio.sleep(0.05) # Prevent flood wait
        except Forbidden:
            # User blocked bot, remove from DB
            users.delete_one({"chat_id": chat_id})
        except Exception as e:
            logger.warning(f"Failed to send to {chat_id}: {e}")

    await update.message.reply_text(f"✅ Broadcast complete.\nSent to: {count}/{total} users.")

# ================= /AI ADMIN PANEL =================

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID: return
    kb = [[InlineKeyboardButton(p.upper(), callback_data=f"prov:{p}")] for p in AI_MODELS]
    await update.message.reply_text("🧠 Select Provider:", reply_markup=InlineKeyboardMarkup(kb))

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
        AI_PROVIDER = prov
        AI_MODEL = mod
        await query.message.edit_text(f"✅ Active: {prov} -> {mod}")

# ================= MAIN =================

def main():
    logger.info("Starting Mitsuri Agent...")
    
    request = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app_bot = ApplicationBuilder().token(BOT_TOKEN).request(request).build()

    # Handlers
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("ai", ai_cmd))
    app_bot.add_handler(CommandHandler("cast", cast)) # <--- RESTORED!
    app_bot.add_handler(CallbackQueryHandler(ai_buttons))
    
    # Chat Handler (Must be last)
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    
    app_bot.run_polling()

if __name__ == "__main__":
    main()
