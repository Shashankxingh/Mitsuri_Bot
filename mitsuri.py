import os
import asyncio
import threading
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
from telegram.error import Forbidden

from groq import AsyncGroq
from cerebras.cloud.sdk import Cerebras
from sambanova import SambaNova

# ================= LOAD ENV =================

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")

OWNER_ID = 8162412883
ADMIN_GROUP_ID = -1002759296936

# ================= FLASK (RENDER PORT FIX) =================

app = Flask(__name__)

@app.route("/")
def home():
    return "Mitsuri is alive 🌸"

def run_web():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_web, daemon=True).start()

# ================= SYSTEM PROMPT =================

SYSTEM_PROMPT = (
    "You are Mitsuri Kanroji from Demon Slayer. "
    "Personality: warm, cheerful, romantic, sweet. "
    "Speak in the language, user is comfortable. "
    "Keep replies very very short, cute, friendly. "
    "Use emojis sparingly (🌸💖🍡)."
)

# ================= AI MODELS =================

AI_MODELS = {
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "cerebras": [
        "llama3.1-8b",
        "llama-3.3-70b",
    ],
    "sambanova": [
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.3-70B-Instruct",
    ],
}

AI_PROVIDER = "groq"
AI_MODEL = "llama-3.1-8b-instant"

# ================= DATABASE =================

mongo = MongoClient(MONGO_URI)
db = mongo["mitsuri"]
users = db["users"]

def save_user(update: Update):
    users.update_one(
        {"chat_id": update.effective_chat.id},
        {"$set": {"chat_id": update.effective_chat.id}},
        upsert=True,
    )

# ================= AI CLIENTS =================

groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
cerebras = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
sambanova = SambaNova(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

async def ask_ai(prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if AI_PROVIDER == "groq":
        res = await groq.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=200,
        )
        return res.choices[0].message.content

    if AI_PROVIDER == "cerebras":
        res = await asyncio.to_thread(
            cerebras.chat.completions.create,
            model=AI_MODEL,
            messages=messages,
            max_tokens=200,
        )
        return res.choices[0].message.content

    if AI_PROVIDER == "sambanova":
        res = await asyncio.to_thread(
            sambanova.chat.completions.create,
            model=AI_MODEL,
            messages=messages,
            max_tokens=200,
        )
        return res.choices[0].message.content

    return "AI error."

# ================= COMMANDS =================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    await update.message.reply_text("🌸 Hii! Main Mitsuri hoon 💖")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – start\n"
        "/help – help\n"
        "/ai – AI control (owner only)\n"
        "/cast – broadcast (owner only)"
    )

# ================= /AI BUTTONS =================

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if (
        update.effective_user.id != OWNER_ID
        or update.effective_chat.id != ADMIN_GROUP_ID
    ):
        return

    keyboard = [
        [InlineKeyboardButton(p.upper(), callback_data=f"prov:{p}")]
        for p in AI_MODELS
    ]

    await update.message.reply_text(
        "🧠 Select AI Provider:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def ai_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_PROVIDER, AI_MODEL

    query = update.callback_query
    await query.answer()

    if (
        query.from_user.id != OWNER_ID
        or query.message.chat.id != ADMIN_GROUP_ID
    ):
        return

    data = query.data

    if data.startswith("prov:"):
        provider = data.split(":")[1]
        keyboard = [
            [InlineKeyboardButton(m, callback_data=f"model:{provider}:{m}")]
            for m in AI_MODELS[provider]
        ]

        await query.message.edit_text(
            f"Provider: {provider}\nChoose model:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    elif data.startswith("model:"):
        _, provider, model = data.split(":", 2)
        AI_PROVIDER = provider
        AI_MODEL = model

        await query.message.edit_text(
            f"✅ AI Updated\n\nProvider: {provider}\nModel: {model}"
        )

# ================= BROADCAST =================

async def cast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if (
        update.effective_user.id != OWNER_ID
        or update.effective_chat.id != ADMIN_GROUP_ID
    ):
        return

    msg = " ".join(context.args)
    if not msg:
        return

    for u in users.find({}, {"chat_id": 1}):
        try:
            await context.bot.send_message(u["chat_id"], msg)
        except Forbidden:
            users.delete_one({"chat_id": u["chat_id"]})
        except:
            pass

    await update.message.reply_text("📢 Broadcast sent")

# ================= CHAT HANDLER (NO FLOOD) =================

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    text = msg.text.lower()
    bot_username = context.bot.username.lower()
    chat_type = update.effective_chat.type

    should_reply = False

    if chat_type == "private":
        should_reply = True
    else:
        if "mitsuri" in text:
            should_reply = True
        elif f"@{bot_username}" in text:
            should_reply = True
        elif msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id:
            should_reply = True

    if not should_reply:
        return

    save_user(update)
    reply = await ask_ai(msg.text)

    try:
        await msg.reply_text(reply)
    except Forbidden:
        users.delete_one({"chat_id": update.effective_chat.id})

# ================= MAIN =================

def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("ai", ai_cmd))
    application.add_handler(CallbackQueryHandler(ai_buttons))
    application.add_handler(CommandHandler("cast", cast))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("🌸 Mitsuri is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
