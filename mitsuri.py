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
OWNER_ID = 8162412883
ADMIN_GROUP_ID = -1002759296936

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
DAILY_LIMIT = 5

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

EXAMPLES:
User: "I'm sad"
You: "Aww yaar, kya hua? 😞 Tell me what's bothering you... I'm here to listen 💕"

User: "What's 5+5?"
You: "That's 10! Easy peasy ✨ Need help with something harder?"

Remember: Be genuinely YOU, not a generic bot. Show emotion and care!"""

# ⚡ UPDATED AI PROVIDERS (Groq + Gemini)
AI_PROVIDER = "groq" # Default
CHAT_MODEL = "llama-3.1-8b-instant"
SEARCH_MODEL = "llama-3.3-70b-versatile"

AI_MODELS = {
    "groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash"],
}

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
    if user_id == OWNER_ID:
        return True, "Infinity"

    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    user = users.find_one({"chat_id": user_id})
    usage = user.get("usage", {"date": today_str, "count": 0}) if user else {"date": today_str, "count": 0}
    
    if usage["date"] != today_str:
        usage = {"date": today_str, "count": 0}
    
    if usage["count"] >= DAILY_LIMIT:
        return False, usage["count"]
    
    return True, usage["count"]

def increment_usage(user_id):
    if user_id == OWNER_ID: return
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    user = users.find_one({"chat_id": user_id})
    usage = user.get("usage", {"date": today_str, "count": 0}) if user else {"date": today_str, "count": 0}
    
    if usage["date"] != today_str:
        usage = {"date": today_str, "count": 1}
    else:
        usage["count"] += 1
        
    users.update_one({"chat_id": user_id}, {"$set": {"usage": usage}}, upsert=True)

# ================= AI CLIENTS =================
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

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

async def get_ai_response(messages, model, max_tokens=2000):
    try:
        # --- GROQ HANDLER ---
        if AI_PROVIDER == "groq":
            r = await groq_client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=0.85,
                max_tokens=max_tokens,
                top_p=0.9
            )
            return r.choices[0].message.content

        # --- GEMINI HANDLER ---
        elif AI_PROVIDER == "gemini":
            # 1. Extract System Prompt
            system_instruction = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            
            # 2. Configure Model
            gen_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction
            )

            # 3. Format History (Gemini uses 'user' and 'model' roles)
            chat_history = []
            last_msg = messages[-1]["content"] # The current prompt

            for msg in messages:
                if msg["role"] == "system": continue # Handled above
                if msg == messages[-1]: continue     # Skip current msg, handled in send_message
                
                # Map roles: 'assistant' -> 'model', 'user' -> 'user'
                role = "model" if msg["role"] == "assistant" else "user"
                chat_history.append({"role": role, "parts": [msg["content"]]})

            # 4. Generate
            chat = gen_model.start_chat(history=chat_history)
            response = await chat.send_message_async(last_msg)
            return response.text

    except Exception as e:
        logger.error(f"AI Error ({AI_PROVIDER}): {e}")
        return None

# ================= COMMANDS =================

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Web Search Command"""
    user_id = update.effective_user.id
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("Arre yaar, query toh batao! 😅", parse_mode="Markdown")
        return

    allowed, count = check_limit(user_id)
    if not allowed:
        await update.message.reply_text(f"Daily limit khatam! 😞 ({count}/{DAILY_LIMIT})")
        return

    status_msg = await update.message.reply_text(f"🔍 Searching web... ({count}/{DAILY_LIMIT if user_id != OWNER_ID else '∞'})")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    raw_search = await asyncio.to_thread(search_web, query)
    
    if not raw_search:
        await status_msg.edit_text("Hmm... kuch nahi mila 🥺")
        return

    now_str = datetime.now(IST).strftime("%I:%M %p, %A")
    system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"[Context: User used /ask for web search]\n"
        f"[Time: {now_str}]\n\n"
        f"SEARCH RESULTS:\n{raw_search}\n\n"
        f"Answer using search results. Mention sources. Keep your personality!"
    )
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    
    # Use Search Model
    response = await get_ai_response(messages, SEARCH_MODEL, max_tokens=1500)
    
    if response:
        increment_usage(user_id)
        await status_msg.delete()
        await update.message.reply_text(response)
    else:
        await status_msg.edit_text("Network issue 😓 Try again?")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Normal Chat Handler"""
    msg = update.message
    if not msg or not msg.text: return
    
    is_private = update.effective_chat.type == "private"
    is_mentioned = "mitsuri" in msg.text.lower() or (msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id)
    
    if not (is_private or is_mentioned): return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    history = get_history(update.effective_chat.id)
    now_str = datetime.now(IST).strftime("%I:%M %p")
    user_name = update.effective_user.first_name or "friend"
    
    sys_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"[Time: {now_str}]\n"
        f"[User's name: {user_name}]\n"
        f"[You have conversation history - use it!]"
    )
    
    messages = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": msg.text}]
    
    response = await get_ai_response(messages, CHAT_MODEL)
    
    if response:
        update_history(update.effective_chat.id, "user", msg.text)
        update_history(update.effective_chat.id, "assistant", response)
        await msg.reply_text(response)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.update_one({"chat_id": update.effective_chat.id}, {"$set": {"history": []}}, upsert=True)
    await update.message.reply_text(
        f"Kyaaa~! Nayi friend! 🌸💕\n\n"
        f"Main Mitsuri hoon! Nice to meet you {update.effective_user.first_name}! ✨\n\n"
        "💬 **Chat:** Just talk naturally!\n"
        "🔍 **Search:** `/ask <your question>` (5 free/day)\n"
        "🔄 **Reset:** `/reset` to clear memory\n\n"
        "Toh batao, kya haal hai? 😊", 
        parse_mode="Markdown"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    users.update_one({"chat_id": update.effective_chat.id}, {"$set": {"history": []}}, upsert=True)
    await update.message.reply_text("Memory clear! Fresh start karte hain 🌸")

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
    await update.message.reply_text(f"Sent to {count} users ✅")

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID: return
    
    info = f"⚡ Provider: {AI_PROVIDER}\n💬 Chat: {CHAT_MODEL}\n🔍 Search: {SEARCH_MODEL}"
    kb = [[InlineKeyboardButton(p.upper(), callback_data=f"prov:{p}")] for p in AI_MODELS]
    await update.message.reply_text(f"Current Settings:\n{info}\n\nSwitch Provider?", reply_markup=InlineKeyboardMarkup(kb))

async def ai_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_PROVIDER, SEARCH_MODEL, CHAT_MODEL
    query = update.callback_query
    await query.answer()
    if query.from_user.id != OWNER_ID: return
    data = query.data
    
    if data.startswith("prov:"):
        prov = data.split(":")[1]
        AI_PROVIDER = prov
        # Auto-set default models when switching provider
        if prov == "groq":
            CHAT_MODEL = "llama-3.1-8b-instant"
            SEARCH_MODEL = "llama-3.3-70b-versatile"
        elif prov == "gemini":
            CHAT_MODEL = "gemini-2.0-flash"
            SEARCH_MODEL = "gemini-2.0-flash"
            
        await query.message.edit_text(
            f"✅ Switched to **{prov.upper()}**\nDefault models set.",
            parse_mode="Markdown"
        )

# ================= MAIN =================

def main():
    request = HTTPXRequest(connect_timeout=20, read_timeout=20)
    app_bot = ApplicationBuilder().token(BOT_TOKEN).request(request).build()

    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("ask", ask_cmd))
    app_bot.add_handler(CommandHandler("reset", reset))
    app_bot.add_handler(CommandHandler("cast", cast))
    app_bot.add_handler(CommandHandler("ai", ai_cmd))
    app_bot.add_handler(CallbackQueryHandler(ai_buttons))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("🌸 Mitsuri running with GROQ + GEMINI!")
    app_bot.run_polling()

if __name__ == "__main__":
    main()
