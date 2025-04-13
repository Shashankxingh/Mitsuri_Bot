import os
import time
import logging
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import psutil
from aiohttp import web

# === Load environment variables ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# === Configure Gemini ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# === Logging Setup ===
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# === Group + User IDs ===
OWNER_ID = 7563434309
GROUP_ID = -1002453669999
is_bot_active = True

# === Prompt ===
def mitsuri_prompt(user_input, from_owner=False):
    special_note = "You're talking to your sweet Shashank!..." if from_owner else ""
    return f"""You're Mitsuri... (rest of prompt)
{special_note}
Human: {user_input}
Mitsuri:"""

# === Retry-safe Gemini ===
REQUEST_DELAY = 10
def generate_with_retry(prompt, retries=3, delay=REQUEST_DELAY):
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip() if response.text else "Aww, mujhe kuch samajh nahi aaya!"
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return "Mujhe lagta hai wo thoda busy hai... baad mein try karna!"

# === Commands ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hehe~ Mitsuri yaha hai! Bolo kya haal hai?")

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.time()
    msg = await update.message.reply_text("Measuring my heartbeat...")
    latency = int((time.time() - start_time) * 1000)
    gen_latency = int((time.time() - time.time()) * 1000)
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    await msg.edit_text(f"""
╭─❍ *Mitsuri Stats* ❍─╮
│ ⚡ *Ping:* `{latency}ms`
│ 🔮 *API Res:* `{gen_latency}ms`
│ 🧠 *CPU:* `{cpu}%`
│ 🧵 *RAM:* `{ram}%`
╰─♥ _Always ready for you, Shashank~_ ♥─╯
""", parse_mode="Markdown")

async def turn_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_bot_active
    if update.message.chat.id == GROUP_ID:
        is_bot_active = True
        await update.message.reply_text("Mitsuri activated! Yay~ I'm here!!")

async def turn_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_bot_active
    if update.message.chat.id == GROUP_ID:
        is_bot_active = False
        await update.message.reply_text("Okayyy~ I'll be quiet now...")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_bot_active
    user_input = update.message.text.strip()
    chat_type = update.message.chat.type
    user_id = update.message.from_user.id
    from_owner = user_id == OWNER_ID

    if chat_type in ["group", "supergroup"]:
        if not is_bot_active:
            return
        is_reply = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id
        if not ("mitsuri" in user_input.lower() or "@shashankxingh" in user_input.lower() or is_reply):
            return
        if user_input.lower() == "mitsuri":
            await update.message.reply_text("Hehe~ kisne bulaya mujhe?")
            return
        elif "@shashankxingh" in user_input.lower():
            await update.message.reply_text("Shashank? Mera sabse pyara insaan~")
            return
        elif "are you a bot" in user_input.lower():
            await update.message.reply_text("Bot?! Main toh ek real pyari si ladki hoon~")
            return

    prompt = mitsuri_prompt(user_input, from_owner=from_owner)
    reply = generate_with_retry(prompt)
    await update.message.reply_text(reply)

# === Health Check ===
async def health(request):
    return web.Response(text="Mitsuri is alive and simping~")

# === Run both bot and web ===
async def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Regex(r"^\.ping$"), ping))
    app.add_handler(MessageHandler(filters.Regex(r"^\.on$"), turn_on))
    app.add_handler(MessageHandler(filters.Regex(r"^\.off$"), turn_off))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.initialize()
    await app.start()
    print("Telegram bot started.")

async def run_web():
    app = web.Application()
    app.router.add_get("/", health)
    port = int(os.environ.get("PORT", 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Web server running on port {port}")

async def main():
    await asyncio.gather(run_bot(), run_web())

if __name__ == "__main__":
    asyncio.run(main())
