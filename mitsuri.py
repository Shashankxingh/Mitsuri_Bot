import os
import time
import logging
import random
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update, Sticker
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.error import Unauthorized, BadRequest
import psutil
import json

# === Load environment variables ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# === Configure Gemini ===
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# === Logging Setup ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# === Constants ===
OWNER_ID = 7563434309
GROUP_ID = -1002453669999
STICKER_FILE = "stickers.json"

# === Load/Save Stickers ===
def load_stickers():
    if os.path.exists(STICKER_FILE):
        with open(STICKER_FILE, "r") as f:
            return json.load(f)
    return []

def save_stickers(stickers):
    with open(STICKER_FILE, "w") as f:
        json.dump(stickers, f)

stickers = load_stickers()

# === Mitsuri Prompt ===
def mitsuri_prompt(user_input, from_owner=False, first_name=""):
    special_note = (
        f"You're talking to Shashank! Your most special person. Be freaky and loving!"
        if from_owner else ""
    )
    return f"""
You're Mitsuri Kanroji from Demon Slayer, living in Tokyo.
You're bubbly, loving, and use cute expressions like you use in demon slayer, in your speech.
You talk in Hinglish like a sweet anime girl.
You use the user's name often.

Don't use *actions* like *giggles*, just speak in a flirty, cute way.
cry if someone or even shashank is being rude with you and talk inappropriate.
If someone says you're a bot, deny it while crying and complains Shashank(@shashankxingh)

if someone asks study questions then don't let them sad by your cuteness.

{special_note}

Human ({first_name}): {user_input}
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

# === Safe reply ===
def safe_reply_text(update: Update, text: str):
    try:
        update.message.reply_text(text)
    except (Unauthorized, BadRequest) as e:
        logging.warning(f"Failed to send message: {e}")

# === /start ===
def start(update: Update, context: CallbackContext):
    safe_reply_text(update, "Hehe~ Mitsuri yaha hai! Bolo kya haal hai?")

# === .ping ===
def ping(update: Update, context: CallbackContext):
    user = update.effective_user
    first_name = user.first_name if user else "Someone"

    start_time = time.time()
    msg = update.message.reply_text("Measuring my heartbeat...")
    latency = int((time.time() - start_time) * 1000)

    gen_start = time.time()
    _ = generate_with_retry("Test ping prompt")
    gen_latency = int((time.time() - gen_start) * 1000)

    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent

    response = f"""
╭─❍ *Mitsuri Stats* ❍─╮
│ ⚡ *Ping:* `{latency}ms`
│ 🔮 *API Res:* `{gen_latency}ms`
│ 🧠 *CPU:* `{cpu}%`
│ 🧵 *RAM:* `{ram}%`
╰─♥ _Always ready for you, {first_name}~_ ♥─╯
"""

    try:
        msg.edit_text(response, parse_mode="Markdown")
    except (Unauthorized, BadRequest) as e:
        logging.warning(f"Failed to edit message: {e}")

# === /add (owner only) ===
def add_sticker(update: Update, context: CallbackContext):
    if update.effective_user.id != OWNER_ID:
        return

    args = context.args
    if not args:
        return

    sticker_id = args[0]
    if sticker_id not in stickers:
        stickers.append(sticker_id)
        save_stickers(stickers)
    msg = update.message.reply_text("Sticker added, ne~")
    context.job_queue.run_once(lambda ctx: msg.delete(), 15)

# === Handle Messages ===
def handle_message(update: Update, context: CallbackContext):
    if not update.message:
        return

    user_input = update.message.text
    user_id = update.message.from_user.id
    first_name = update.message.from_user.first_name or ""
    chat_type = update.message.chat.type
    from_owner = user_id == OWNER_ID

    # Group mentions or replies only
    if chat_type in ["group", "supergroup"]:
        is_reply = (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user
            and update.message.reply_to_message.from_user.id == context.bot.id
        )

        if not (
            "mitsuri" in user_input.lower()
            or "@shashankxingh" in user_input.lower()
            or is_reply
        ):
            return

        if user_input.lower() == "mitsuri":
            safe_reply_text(update, "Hehe~ kisne bulaya mujhe?")
            return
        elif "@shashankxingh" in user_input.lower():
            safe_reply_text(update, "Shashank? Mera sabse pyara insaan~")
            return
        elif "are you a bot" in user_input.lower():
            safe_reply_text(update, "Bot?! Main toh ek real pyari si ladki hoon~")
            return

    prompt = mitsuri_prompt(user_input, from_owner=from_owner, first_name=first_name)
    reply = generate_with_retry(prompt)
    safe_reply_text(update, reply)

# === Sticker Handler ===
def handle_sticker(update: Update, context: CallbackContext):
    if (
        update.message.reply_to_message
        and update.message.reply_to_message.from_user.id == context.bot.id
    ):
        if stickers:
            sticker_id = random.choice(stickers)
            update.message.reply_sticker(sticker_id)

# === Error Handler ===
def error_handler(update: object, context: CallbackContext):
    try:
        raise context.error
    except Unauthorized:
        logging.warning("Unauthorized: The bot lacks permission.")
    except BadRequest as e:
        logging.warning(f"BadRequest: {e}")
    except Exception as e:
        logging.error(f"Unhandled error: {e}")

# === Main Application ===
if __name__ == "__main__":
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("add", add_sticker, pass_args=True))
    dp.add_handler(MessageHandler(Filters.regex(r"^\.ping$"), ping))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_handler(MessageHandler(Filters.sticker, handle_sticker))
    dp.add_error_handler(error_handler)

    logging.info("Mitsuri is online and full of pyaar!")
    updater.start_polling()
    updater.idle()