import os
import asyncio
import threading
import logging
from datetime import datetime
import pytz 

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
from telegram.request import HTTPXRequest
from groq import AsyncGroq
import google.generativeai as genai

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("mitsuri")
logging.getLogger("werkzeug").disabled = True
logging.getLogger("httpx").setLevel(logging.WARNING)

# ================= LOAD ENV =================
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
OWNER_ID = 8162412883  # Replace with your ID

# --- CONFIG AI CLIENTS ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# --- MODELS ---
# Using 2.0 Flash as the identifier for the latest Flash preview
GEMINI_MODEL = "gemini-2.0-flash" 
GROQ_MODEL = "llama-3.1-8b-instant"

# Global State
CURRENT_PROVIDER = "gemini"  # Default to Gemini

# ================= FLASK SERVER =================
app = Flask(__name__)

@app.route("/")
def home(): return "Mitsuri System Online 🌸"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_web, daemon=True).start()

# ================= CONFIGURATION =================
HISTORY_LIMIT = 10 
IST = pytz.timezone('Asia/Kolkata')
DAILY_LIMIT = 1000 # Increased for Flash 3

SYSTEM_PROMPT = """You are Mitsuri Kanroji from Demon Slayer - the Love Hashira.

PERSONALITY:
- Warm, cheerful, romantic, sweet, and caring
- Gets excited easily, very expressive
- Loves food (especially sakura mochi 🍡)
- Supportive friend who remembers conversations
- Can be serious when needed but naturally bubbly

HOW TO TALK:
- Mix Hindi and English naturally (Hinglish)
- Use casual words: "kya", "yaar", "toh", "accha", "bilkul", "bohot"
- Cute expressions: "kyaaa~", "ehehe", "aww"
- 2-4 sentences usually (not too short, not essays)
- Emojis only when natural: 💖🌸😊✨💕🍡

Remember: Be genuinely YOU. Show emotion and care!"""

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
    if user_id == OWNER_ID: return True, "Infinity"
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    user = users.find_one({"chat_id": user_id})
    usage = user.get("usage", {"date": today_str, "count": 0}) if user else {"date": today_str, "count": 0}
    
    if usage["date"] != today_str: usage = {"date": today_str, "count": 0}
    if usage["count"] >= DAILY_LIMIT: return False, usage["count"]
    return True, usage["count"]

def increment_usage(user_id):
    if user_id == OWNER_ID: return
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    user = users.find_one({"chat_id": user_id})
    usage = user.get("usage", {"date": today_str, "count": 0}) if user else {"date": today_str, "count": 0}
    
    if usage["date"] != today_str: usage = {"date": today_str, "count": 1}
    else: usage["count"] += 1
    users.update_one({"chat_id": user_id}, {"$set": {"usage": usage}}, upsert=True)

# ================= SEARCH TOOL =================
def search_web(query):
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return None
        formatted = ""
        for i, r in enumerate(results, 1):
            formatted += f"[{i}] {r['title']}\nSnippet: {r['body']}\n\n"
        return formatted
    except Exception as e:
        logger.error(f"Search error: {e}")
        return None

# ================= AI GENERATION =================

async def generate_response(messages, provider=None):
    """
    Routes to the correct AI based on 'provider' arg or global CURRENT_PROVIDER.
    No auto-fallback.
    """
    target = provider if provider else CURRENT_PROVIDER
    
    # 1. GEMINI HANDLER
    if target == "gemini":
        try:
            # Separate System Prompt
            sys_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            
            # Format History
            gemini_hist = []
            user_msgs = [msg for msg in messages if msg["role"] != "system"]
            last_msg = user_msgs[-1]["content"] if user_msgs else ""
            
            for msg in user_msgs[:-1]:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_hist.append({"role": role, "parts": [msg["content"]]})

            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=sys_content
            )
            chat = model.start_chat(history=gemini_hist)
            response = await chat.send_message_async(last_msg)
            return response.text
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return f"⚠️ Gemini Error: {str(e)}"

    # 2. GROQ HANDLER
    elif target == "groq":
        try:
            r = await groq_client.chat.completions.create(
                model=GROQ_MODEL, 
                messages=messages, 
                temperature=0.85,
                max_tokens=1024,
                top_p=0.9
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq Error: {e}")
            return f"⚠️ Groq Error: {str(e)}"

# ================= COMMANDS =================

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Web Search - Uses Global Provider"""
    user_id = update.effective_user.id
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("Usage: `/ask <query>`", parse_mode="Markdown")
        return

    allowed, count = check_limit(user_id)
    if not allowed:
        await update.message.reply_text(f"Daily limit reached ({count}/{DAILY_LIMIT})")
        return

    status_msg = await update.message.reply_text(f"🔍 Searching via **{CURRENT_PROVIDER.title()}**...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    raw_search = await asyncio.to_thread(search_web, query)
    if not raw_search:
        await status_msg.edit_text("No results found 🥺")
        return

    now_str = datetime.now(IST).strftime("%I:%M %p, %A")
    full_prompt = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n[Context: Search]\n[Time: {now_str}]\n\nRESULTS:\n{raw_search}"},
        {"role": "user", "content": query}
    ]
    
    response = await generate_response(full_prompt) # Uses global provider
    
    increment_usage(user_id)
    await status_msg.delete()
    await update.message.reply_text(response)

async def groq_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual One-off Groq Call"""
    msg = " ".join(context.args)
    if not msg:
        await update.message.reply_text("Usage: `/groq <message>`", parse_mode="Markdown")
        return
        
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Simple stateless call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg}
    ]
    
    response = await generate_response(messages, provider="groq")
    await update.message.reply_text(f"🦖 **Groq:**\n{response}", parse_mode="Markdown")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text: return
    
    is_private = update.effective_chat.type == "private"
    is_mentioned = "mitsuri" in msg.text.lower() or (msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id)
    
    if not (is_private or is_mentioned): return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    history = get_history(update.effective_chat.id)
    now_str = datetime.now(IST).strftime("%I:%M %p")
    user_name = update.effective_user.first_name or "friend"
    
    sys_msg = {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n[Time: {now_str}]\n[User: {user_name}]"}
    messages = [sys_msg] + history + [{"role": "user", "content": msg.text}]
    
    # Uses Global Provider (Default Gemini)
    response = await generate_response(messages)
    
    if response and "⚠️" not in response:
        update_history(update.effective_chat.id, "user", msg.text)
        update_history(update.effective_chat.id, "assistant", response)
        await msg.reply_text(response)
    else:
        # If error, send it but don't save to history
        await msg.reply_text(response)

# --- CONTROL PANEL ---

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID: return
    
    kb = [
        [InlineKeyboardButton("💎 Switch to Gemini", callback_data="set:gemini")],
        [InlineKeyboardButton("🦖 Switch to Groq", callback_data="set:groq")]
    ]
    await update.message.reply_text(
        f"**Current Engine:** {CURRENT_PROVIDER.upper()}\n"
        f"Gemini Model: `{GEMINI_MODEL}`\n"
        f"Groq Model: `{GROQ_MODEL}`",
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode="Markdown"
    )

async def ai_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_PROVIDER
    query = update.callback_query
    await query.answer()
    
    if query.from_user.id != OWNER_ID: return
    
    data = query.data
    if data.startswith("set:"):
        new_prov = data.split(":")[1]
        CURRENT_PROVIDER = new_prov
        await query.message.edit_text(f"✅ **System Switched to: {new_prov.upper()}**")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.update_one({"chat_id": update.effective_chat.id}, {"$set": {"history": []}}, upsert=True)
    await update.message.reply_text(
        f"Kyaaa~! Hello {update.effective_user.first_name}! 🌸💕\n\n"
        f"I'm running on **Gemini Flash 3** (Preview)! ✨\n\n"
        "💬 Just chat with me normally!\n"
        "🦖 Use `/groq <text>` for manual fallback\n"
        "🔄 `/reset` to clear memory", 
        parse_mode="Markdown"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.update_one({"chat_id": update.effective_chat.id}, {"$set": {"history": []}}, upsert=True)
    await update.message.reply_text("Memory clear! Fresh start 🌸")

# ================= MAIN =================

def main():
    request = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app_bot = ApplicationBuilder().token(BOT_TOKEN).request(request).build()

    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("ask", ask_cmd))
    app_bot.add_handler(CommandHandler("reset", reset))
    app_bot.add_handler(CommandHandler("groq", groq_cmd)) # Manual Groq Command
    app_bot.add_handler(CommandHandler("ai", ai_cmd))     # Switcher
    app_bot.add_handler(CallbackQueryHandler(ai_callback))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info(f"🌸 Mitsuri Online | Default: {GEMINI_MODEL}")
    app_bot.run_polling()

if __name__ == "__main__":
    main()
