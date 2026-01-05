import os
import asyncio
import threading
import logging
from datetime import datetime
import pytz 

# Search Tool
from ddgs import DDGS

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

logging.getLogger("werkzeug").disabled = True
logging.getLogger("httpx").setLevel(logging.WARNING)

# ================= LOAD ENV =================
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
OWNER_ID = 8162412883 # YOUR ID
ADMIN_GROUP_ID = -1002759296936

# ================= FLASK SERVER =================
app = Flask(__name__)

@app.route("/")
def home(): return "Mitsuri System (Ask Command) Online 🌸"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_web, daemon=True).start()

# ================= CONFIGURATION =================
HISTORY_LIMIT = 10 
IST = pytz.timezone('Asia/Kolkata')
DAILY_LIMIT = 5 # Free searches per day

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

def check_limit(user_id):
    """
    Returns True if user can search, False if limit reached.
    Always returns True for OWNER.
    """
    if user_id == OWNER_ID:
        return True, "Infinity"

    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    user = users.find_one({"chat_id": user_id})
    
    # Default structure if missing
    usage = user.get("usage", {"date": today_str, "count": 0})
    
    # Reset if new day
    if usage["date"] != today_str:
        usage = {"date": today_str, "count": 0}
    
    if usage["count"] >= DAILY_LIMIT:
        return False, usage["count"]
    
    return True, usage["count"]

def increment_usage(user_id):
    if user_id == OWNER_ID: return
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    
    # Logic to handle date reset + increment safely
    user = users.find_one({"chat_id": user_id})
    usage = user.get("usage", {"date": today_str, "count": 0})
    
    if usage["date"] != today_str:
        usage = {"date": today_str, "count": 1}
    else:
        usage["count"] += 1
        
    users.update_one({"chat_id": user_id}, {"$set": {"usage": usage}}, upsert=True)

# ================= AI CLIENTS =================
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
cerebras_client = AsyncOpenAI(api_key=os.getenv("CEREBRAS_API_KEY"), base_url="https://api.cerebras.ai/v1")
sambanova_client = AsyncOpenAI(api_key=os.getenv("SAMBANOVA_API_KEY"), base_url="https://api.sambanova.ai/v1")

# ================= 🔍 SEARCH TOOL =================

def search_web(query):
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return None
        formatted = ""
        for i, r in enumerate(results, 1):
            formatted += f"[{i}] {r['title']}\nSnippet: {r['body']}\nLink: {r['href']}\n\n"
        return formatted
    except Exception as e:
        logger.error(f"Search error: {e}")
        return None

async def get_ai_response(messages, model):
    try:
        client = None
        if AI_PROVIDER == "groq": client = groq_client
        elif AI_PROVIDER == "cerebras": client = cerebras_client
        elif AI_PROVIDER == "sambanova": client = sambanova_client
        
        r = await client.chat.completions.create(model=model, messages=messages, temperature=0.7, max_tokens=1000)
        return r.choices[0].message.content
    except Exception as e:
        logger.error(f"AI Error: {e}")
        return None

# ================= COMMANDS =================

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """The Search Command (/ask <query>)"""
    user_id = update.effective_user.id
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("❓ Usage: `/ask Bitcoin price` or `/ask Who is PM of India`", parse_mode="Markdown")
        return

    # 1. Check Limits
    allowed, count = check_limit(user_id)
    if not allowed:
        await update.message.reply_text(f"❌ Daily limit reached! ({count}/{DAILY_LIMIT})\nTry again tomorrow or ask the owner.")
        return

    status_msg = await update.message.reply_text(f"🔍 Searching... (Used: {count}/{DAILY_LIMIT if user_id != OWNER_ID else '∞'})")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # 2. Perform Search
    raw_search = await asyncio.to_thread(search_web, query)
    
    if not raw_search:
        await status_msg.edit_text("🥺 Couldn't find anything on the web.")
        return

    # 3. Ask AI with Data
    # We do NOT save /ask results to history to keep main chat clean, 
    # but you can change this if you want.
    
    now_str = datetime.now(IST).strftime("%Y-%m-%d %I:%M %p")
    system_prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"Context: User used /ask command.\n"
        f"Current Time: {now_str}\n"
        f"SEARCH RESULTS:\n{raw_search}\n"
        f"INSTRUCTION: Answer the user query using ONLY the search results. Cite sources."
    )
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    
    response = await get_ai_response(messages, AI_MODEL)
    
    # 4. Reply & Update Usage
    if response:
        increment_usage(user_id)
        await status_msg.delete() # Delete "Searching..." message
        await update.message.reply_text(f"{response}")
    else:
        await status_msg.edit_text("Network error while generating answer 🥺")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Normal Chat - Fast, No Search, Uses Memory"""
    msg = update.message
    if not msg or not msg.text: return
    
    is_private = update.effective_chat.type == "private"
    is_mentioned = "mitsuri" in msg.text.lower() or (msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id)
    
    if not (is_private or is_mentioned): return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get History
    history = get_history(update.effective_chat.id)
    
    # Dynamic Time
    now_str = datetime.now(IST).strftime("%I:%M %p")
    sys_prompt = f"{BASE_SYSTEM_PROMPT}\n[Time: {now_str}]"
    
    messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg.text}]
    
    response = await get_ai_response(messages, AI_MODEL)
    
    if response:
        update_history(update.effective_chat.id, "user", msg.text)
        update_history(update.effective_chat.id, "assistant", response)
        await msg.reply_text(response)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.update_one({"chat_id": update.effective_chat.id}, {"$set": {"history": []}}, upsert=True)
    await update.message.reply_text(
        "🌸 Mitsuri Online! 💖\n\n"
        "💬 **Chat:** Just talk to me normally!\n"
        "🌐 **Search:** Use `/ask <query>` (Limit: 5/day)", 
        parse_mode="Markdown"
    )

async def cast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID: return
    msg = " ".join(context.args)
    if not msg: return
    
    users_list = users.find({}, {"chat_id": 1})
    count = 0
    for u in users_list:
        try:
            await context.bot.send_message(u["chat_id"], msg)
            count += 1
            await asyncio.sleep(0.05)
        except: pass
    await update.message.reply_text(f"Broadcast sent to {count} users.")

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
        await query.message.edit_text(f"✅ Active: {prov} -> {mod}")

# ================= MAIN =================

def main():
    request = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app_bot = ApplicationBuilder().token(BOT_TOKEN).request(request).build()

    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("ask", ask_cmd))   # <--- NEW /ask COMMAND
    app_bot.add_handler(CommandHandler("cast", cast))
    app_bot.add_handler(CommandHandler("ai", ai_cmd))
    app_bot.add_handler(CallbackQueryHandler(ai_buttons))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("🌸 Mitsuri is running...")
    app_bot.run_polling()

if __name__ == "__main__":
    main()
