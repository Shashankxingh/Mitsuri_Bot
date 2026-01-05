import os
import asyncio
from dotenv import load_dotenv
from pymongo import MongoClient
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from groq import AsyncGroq
from cerebras.cloud.sdk import Cerebras
from sambanova import SambaNova

# ================= BASIC CONFIG =================

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")

OWNER_ID = 8162412883
ADMIN_GROUP_ID = -1002759296936

# ================= AI MODEL REGISTRY =================

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
    await update.message.reply_text(
        "🌸 Hi! I'm Mitsuri!\n"
        "Talk to me normally 💖\n\n"
        "Use /help for commands."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start – start bot\n"
        "/help – show help\n"
        "/ai – change AI (owner only)\n"
        "/cast – broadcast (owner only)"
    )

async def ai_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AI_PROVIDER, AI_MODEL

    if update.effective_user.id != OWNER_ID:
        return

    if len(context.args) != 2:
        text = "Usage:\n/ai <provider> <model>\n\nAvailable models:\n"
        for p, models in AI_MODELS.items():
            text += f"\n🔹 {p}:\n"
            for m in models:
                text += f"  - {m}\n"
        await update.message.reply_text(text)
        return

    provider = context.args[0].lower()
    model = context.args[1]

    if provider not in AI_MODELS:
        await update.message.reply_text("❌ Invalid provider.")
        return

    if model not in AI_MODELS[provider]:
        await update.message.reply_text("❌ Invalid model for this provider.")
        return

    AI_PROVIDER = provider
    AI_MODEL = model

    await update.message.reply_text(
        f"✅ AI updated!\n\nProvider: {AI_PROVIDER}\nModel: {AI_MODEL}"
    )

async def cast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        return

    if update.effective_chat.id != ADMIN_GROUP_ID:
        await update.message.reply_text("❌ Use this in admin group only.")
        return

    msg = " ".join(context.args)
    if not msg:
        return

    for u in users.find({}, {"chat_id": 1}):
        try:
            await context.bot.send_message(u["chat_id"], msg)
        except:
            pass

    await update.message.reply_text("📢 Broadcast sent!")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update)
    reply = await ask_ai(update.message.text)
    await update.message.reply_text(reply)

# ================= MAIN =================

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ai", ai_cmd))
    app.add_handler(CommandHandler("cast", cast))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("🌸 Mitsuri is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
