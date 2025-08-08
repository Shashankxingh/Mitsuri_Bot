import os
import time
import datetime
import logging
import re
from dotenv import load_dotenv
from html import escape
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatMemberUpdated
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
    ChatMemberHandler,
    CallbackQueryHandler,
)
from telegram.error import Unauthorized, BadRequest
from pymongo import MongoClient
import openai

# === Load environment variables ===
load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")

# === Owner and group IDs ===
OWNER_ID = 8162412883
SPECIAL_GROUP_ID = -1002759296936  # ✅ Use full supergroup ID

# === OpenAI (DeepInfra) configuration ===
openai.api_key = DEEPINFRA_API_KEY
openai.api_base = "https://api.deepinfra.com/v1/openai"  # DeepInfra base URL

DEEPINFRA_MODEL = "meta-llama/Llama-3-8b-instruct"

# === MongoDB setup ===
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["MitsuriDB"]
chat_info_collection = db["chat_info"]
chat_info_collection.create_index("chat_id", unique=True)

# === Logging setup ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# === Constants ===
REQUEST_DELAY = 2
BOT_START_TIME = time.time()
GROUP_COOLDOWN = {}

# === Utility Functions ===
def get_main_menu_buttons():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👤 Personal Chats", callback_data="show_personal_0")],
        [InlineKeyboardButton("👥 Group Chats", callback_data="show_groups_0")]
    ])

def save_chat_info(chat_id, user=None, chat=None):
    data = {"chat_id": chat_id}
    if user:
        data["name"] = user.first_name
        data["username"] = user.username
        data["user_id"] = user.id
    if chat and chat.type != "private":
        data["title"] = chat.title
        if chat.username:
            data["chat_username"] = chat.username
    chat_info_collection.update_one({"chat_id": chat_id}, {"$set": data}, upsert=True)

def build_prompt(last_two_messages, user_input, chosen_name):
    system_instructions = """
- Tum Mitsuri Kanroji ho demon slayer wali
- tumhe Hinglish me baatein karni hai
- Tum cute, cringe ho, childish ho
- baaton ko 1 ya 2 line me rakhna usse jyada nahi
- don't use actions like *giggles*, *blush* etc. instead use emojis
"""
    messages = [
        {"role": "system", "content": system_instructions.strip()}
    ]

    for role, msg in last_two_messages:
        if role == "user":
            messages.append({"role": "user", "content": f"Human ({chosen_name}): {msg}"})
        elif role == "bot":
            messages.append({"role": "assistant", "content": msg})
    
    messages.append({"role": "user", "content": f"Human ({chosen_name}): {user_input}"})
    return messages

def generate_with_retry(prompt, retries=2, delay=REQUEST_DELAY):
    for attempt in range(retries):
        try:
            start = time.time()
            response = openai.ChatCompletion.create(
                model=DEEPINFRA_MODEL,
                messages=prompt,
            )
            duration = time.time() - start
            logging.info(f"DeepInfra response time: {round(duration, 2)}s")

            if not response.choices or not response.choices[0].message.content:
                return "I'm not sure how to respond to that right now."
            
            response_text = response.choices[0].message.content
            return response_text.strip()
        except Exception as e:
            logging.error(f"DeepInfra error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return "I'm having trouble responding at the moment."

def safe_reply_text(update: Update, text: str):
    try:
        update.message.reply_text(text, parse_mode="HTML")
    except (Unauthorized, BadRequest) as e:
        logging.warning(f"Failed to send message: {e}")

def format_uptime(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# === Command Handlers ===
def start(update: Update, context: CallbackContext):
    if update.message:
        safe_reply_text(update, "Hello. Mitsuri is here. How can I help you today?")

def ping(update: Update, context: CallbackContext):
    if not update.message:
        return
    user = update.message.from_user
    name = escape(user.first_name or user.username or "User")
    msg = update.message.reply_text("Checking latency...")

    try:
        start_api_time = time.time()
        deepinfra_reply = openai.ChatCompletion.create(
            model=DEEPINFRA_MODEL,
            messages=[{"role": "user", "content": "Just say pong."}]
        ).choices[0].message.content.strip()
        api_latency = round((time.time() - start_api_time) * 1000)
        uptime = format_uptime(time.time() - BOT_START_TIME)
        group_link = "https://t.me/mitsuri_homie"

        reply = (
            f"╭───[ 🌸 <b>Mitsuri Ping Report</b> ]───\n"
            f"├ Hello <b>{name}</b>\n"
            f"├ Group: <a href='{group_link}'>@the_jellybeans</a>\n"
            f"├ Ping: <b>{deepinfra_reply}</b>\n"
            f"├ API Latency: <b>{api_latency} ms</b>\n"
            f"├ Uptime: <b>{uptime}</b>\n"
            f"╰─ I'm here and responsive."
        )

        context.bot.edit_message_text(
            chat_id=msg.chat_id,
            message_id=msg.message_id,
            text=reply,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as e:
        logging.error(f"/ping error: {e}")
        msg.edit_text("Something went wrong while checking ping.")

# (The rest of your code: show_chats, _send_chat_list, show_callback, track_bot_added_removed, handle_message, error_handler remains the same)

# === Main ===
if __name__ == "__main__":
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("ping", ping))
    dp.add_handler(CommandHandler("show", show_chats))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_handler(ChatMemberHandler(track_bot_added_removed, ChatMemberHandler.MY_CHAT_MEMBER))
    dp.add_handler(CallbackQueryHandler(show_callback))
    dp.add_error_handler(error_handler)

    updater.start_polling()
    updater.idle()