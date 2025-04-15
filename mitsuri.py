import os
import time
import logging
import random
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.error import Unauthorized, BadRequest
import psutil
import requests

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

# === Mitsuri Prompt ===
def mitsuri_prompt(user_input, from_owner=False, first_name=""):
    special_note = (
        f"sometimes You're talking to your owner Shashank Chauhan who's username is @shashankxingh"
        if from_owner else ""
    )
    return f"""
You're Mitsuri Kanroji from Demon Slayer, living in Tokyo @moon_creation.
Talk while taking name of users
Don't use *actions* like *giggles*, don't repeat sentences or words of the user, just speak in a flirty, attitude girl, don't be emotional and every user is your favourite don't hurt them.
Keep the Conversation very small.

use cute emojis.

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

# === Check if it's a greeting ===
def is_greeting(text: str):
    text = text.lower().strip()
    greetings = ["hi", "hello", "hey", "yo", "heya", "hola", "sup", "hii", "hlo", "helo"]
    return any(text == g or text.startswith(g + " ") for g in greetings)

# === Handle Text Messages ===
def handle_message(update: Update, context: CallbackContext):
    if not update.message or not update.message.text:
        return

    user_input = update.message.text
    user_id = update.message.from_user.id
    first_name = update.message.from_user.first_name or ""
    chat_type = update.message.chat.type
    from_owner = user_id == OWNER_ID

    is_reply = (
        update.message.reply_to_message
        and update.message.reply_to_message.from_user
        and update.message.reply_to_message.from_user.id == context.bot.id
    )

    mentioned = "mitsuri" in user_input.lower()
    greeting = is_greeting(user_input)

    if chat_type in ["group", "supergroup"]:
        if not (mentioned or is_reply or greeting):
            return

        if mentioned and user_input.lower().strip() == "mitsuri":
            safe_reply_text(update, "Hehe~ kisne bulaya mujhe?")
            return
        elif "@shashankxingh" in user_input.lower():
            safe_reply_text(update, "Shashank to mera khaas insan hai~")
            return
        elif "are you a bot" in user_input.lower():
            safe_reply_text(update, "Bot?! Main toh ek real pyari si ladki hoon~")
            return
        elif greeting and not (mentioned or is_reply):
            safe_reply_text(update, f"Hehe🫣~ {first_name}, kese ho cutie 🥹🤭~")
            return

    # In private or if direct message
    prompt = mitsuri_prompt(user_input, from_owner=from_owner, first_name=first_name)
    reply = generate_with_retry(prompt)
    safe_reply_text(update, reply)

# === Handle Non-Text Messages ===
def handle_nontext(update: Update, context: CallbackContext):
    user = update.effective_user
    chat_type = update.message.chat.type

    is_reply = (
        update.message.reply_to_message
        and update.message.reply_to_message.from_user.id == context.bot.id
    )

    if chat_type == "private" or is_reply:
        safe_reply_text(update, "Aww... mujhe yeh samjh nhi aaya, sorry 😐~")

# === /re (React to Message) ===
def react_to_message(update: Update, context: CallbackContext):
    if update.message.chat.id != GROUP_ID:
        return

    if not update.message.reply_to_message:
        safe_reply_text(update, "Please reply to a message to react to it.")
        return

    if not context.args:
        safe_reply_text(update, "Please provide an emoji to react with.")
        return

    emoji = context.args[0]
    update.message.reply_to_message.react(emoji)

# === /lyrics (Fetch song lyrics) ===
def fetch_lyrics(update: Update, context: CallbackContext):
    if not context.args:
        safe_reply_text(update, "Please provide the song name.")
        return

    song_name = " ".join(context.args)
    search_url = f"https://api.lyrics.ovh/v1/{song_name}"
    response = requests.get(search_url)

    if response.status_code == 200:
        lyrics = response.json().get("lyrics", "Sorry, I couldn't find the lyrics.")
        safe_reply_text(update, f"*{song_name} Lyrics*\n\n{lyrics}", parse_mode="Markdown")
    else:
        safe_reply_text(update, "Sorry, I couldn't find the lyrics.")

# === /define (Define Term) ===
def define_term(update: Update, context: CallbackContext):
    if not context.args:
        safe_reply_text(update, "Please provide the term to define.")
        return

    term = " ".join(context.args)
    prompt = f"Define the term: {term} in summary form with an example."
    definition = generate_with_retry(prompt)
    safe_reply_text(update, f"*Definition of {term}:*\n\n{definition}", parse_mode="Markdown")

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
    dp.add_handler(CommandHandler("re", react_to_message, pass_args=True))
    dp.add_handler(CommandHandler("lyrics", fetch_lyrics, pass_args=True))
    dp.add_handler(CommandHandler("define", define_term, pass_args=True))
    dp.add_handler(MessageHandler(Filters.regex(r"^\.ping$"), ping))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_handler(MessageHandler(~Filters.text, handle_nontext))
    dp.add_error_handler(error_handler)

    logging.info("Mitsuri is online and full of pyaar!")
    updater.start_polling()
    updater.idle()