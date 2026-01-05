import os
import asyncio
from dotenv import load_dotenv
from pymongo import MongoClient
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
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

# ================= CONFIG =================

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")

OWNER_ID = 8162412883
ADMIN_GROUP_ID = -1002759296936

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
    messages = [{"role": "user", "content": prompt}]

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
    await update.message.reply_text("🌸 Hi! I'm Mitsuri 💖")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – start bot\n"
        "/help – help\n"
        "/ai – AI control (owner only)\n"
        "/cast – broadcast (owner only)"
    )

# ================= /AI WITH BUTTONS =================

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if (
        update.effective_user.id != OWNER_ID
        or update.effective_chat.id != ADMIN_GROUP_ID
    ):
        return

    keyboard = [
        [InlineKeyboardButton(p.upper(), callback_data=f"prov:{p}")]
        for p in AI_MODELS.keys()
    ]

    await update.message.reply_text(
        "🧠 Select AI Provider:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def ai_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_PROVIDER, AI_MODEL

    query = update.callback_query
    await query.answer()

    # 🔒 SECURITY CHECK
    if (
        query.from_user.id != OWNER_ID
        or query.message.chat.id != ADMIN_GROUP_ID
    ):
        await query.answer("❌ Not allowed", show_alert=True)
        return

    data = query.data

    # Provider selection
    if data.startswith("prov:"):
        provider = data.split(":")[1]

        keyboard = [
            [InlineKeyboardButton(m, callback_data=f"model:{provider}:{m}")]
            for m in AI_MODELS[provider]
        ]

        await query.message.edit_text(
            f"📦 Provider: {provider}\nChoose model:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    # Model selection
    elif data.startswith("model:"):
        _, provider, model = data.split(":", 2)

        AI_PROVIDER = provider
        AI_MODEL = model

        await query.message.edit_text(
            f"✅ AI Updated!\n\n"
            f"Provider: {AI_PROVIDER}\n"
            f"Model: {AI_MODEL}"
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

    await update.message.reply_text("📢 Broadcast sent!")

# ================= CHAT HANDLER =================

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    chat_type = update.effective_chat.type
    bot_username = context.bot.username.lower()
    text = msg.text.lower()

    should_reply = False

    if chat_type == "private":
        should_reply = True
    else:
        if "mitsuri" in text:
            should_reply = True
        elif f"@{bot_username}" in text:
            should_reply = True
        elif (
            msg.reply_to_message
            and msg.reply_to_message.from_user.id == context.bot.id
        ):
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
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CallbackQueryHandler(ai_button))
    app.add_handler(CommandHandler("cast", cast))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("🌸 Mitsuri is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
