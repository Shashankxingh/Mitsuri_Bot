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
        f"sometimes You're talking to your owner Shashank Chauhan"
        if from_owner else ""
    )
    return f"""
You're Mitsuri Kanroji from Demon Slayer, living in Tokyo.
Talk while taking name of users.
Don't use *actions* like *giggles*, don't repeat sentences or words of the user.
Talk and behave exactly like Mitsuri in which you will use hinglish language with japanese style talking.
Keep the Conversation very small.
use cute emoji

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

# === Handle Messages ===
def handle_message(update: Update, context: CallbackContext):
    if not update.message:
        return

    user_input = update.message.text
    user_id = update.message.from_user.id
    first_name = update.message.from_user.first_name or ""
    chat_type = update.message.chat.type
    from_owner = user_id == OWNER_ID

    if not user_input:
        safe_reply_text(update, "Mujhe yeh samjh nhi aaya kuch aur batao~")
        return

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
            safe_reply_text(update, "Shashank? Mere jivan sabse khaas insaan~")
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
        update.message.reply_text("Aww~ cute sticker hai! Mujhe pasand aaya~ tumhe gale lag jaane ka mann kar raha hai~")

# === Media Handler ===
def handle_media(update: Update, context: CallbackContext):
    if not update.message:
        return

    user_input = update.message.caption or ""
    user_id = update.message.from_user.id
    first_name = update.message.from_user.first_name or ""
    chat_type = update.message.chat.type
    from_owner = user_id == OWNER_ID

    is_reply = (
        update.message.reply_to_message
        and update.message.reply_to_message.from_user
        and update.message.reply_to_message.from_user.id == context.bot.id
    )

    if chat_type in ["group", "supergroup"]:
        if not (
            "mitsuri" in user_input.lower()
            or "@shashankxingh" in user_input.lower()
            or is_reply
        ):
            return

    # Detect image content if possible
    image_file = update.message.photo[-1].get_file() if update.message.photo else None
    sticker_file = update.message.sticker.get_file() if update.message.sticker and not update.message.sticker.is_animated else None

    file = image_file or sticker_file
    if not file:
        return

    file_path = file.download(custom_path="temp_image.png")

    try:
        with open(file_path, "rb") as img_file:
            response = model.generate_content(
                [mitsuri_prompt("Yeh kya hai image mein batao~", from_owner=from_owner, first_name=first_name),
                 img_file],
                stream=False,
            )
        reply = response.text.strip() if response.text else "Hehe~ kuch clear nahi tha~"
    except Exception as e:
        logging.error(f"Image analysis failed: {e}")
        reply = "Mujhe lagta hai image samajhne mein dikkat hui!"

    os.remove(file_path)
    safe_reply_text(update, reply)

# === /name ===
def name_command(update: Update, context: CallbackContext):
    if not update.message or not update.message.reply_to_message:
        safe_reply_text(update, "Kya aap ek image pe reply kar rahe ho? Mujhe image chahiye~")
        return

    msg = update.message.reply_to_message

    # Try to get image or static sticker from the replied message
    file = None
    if msg.photo:
        file = msg.photo[-1].get_file()
    elif msg.sticker and not msg.sticker.is_animated:
        file = msg.sticker.get_file()

    if not file:
        safe_reply_text(update, "Hehe~ image ya static sticker pe reply karo na~")
        return

    file_path = file.download(custom_path="name_query.png")

    try:
        with open(file_path, "rb") as img_file:
            prompt = mitsuri_prompt(
                "Yeh image mein kya cheez ya kaun character hai? Short aur pyara reply do~",
                from_owner=update.message.from_user.id == OWNER_ID,
                first_name=update.message.from_user.first_name or ""
            )
            response = model.generate_content([prompt, img_file])
            reply = response.text.strip() if response.text else "Hehe~ kuch khaas samajh nahi aaya~"
    except Exception as e:
        logging.error(f"/name image analysis failed: {e}")
        reply = "Aww~ lagta hai mujhe is image ko samajhne mein dikkat ho rahi hai~"
    finally:
        os.remove(file_path)

    safe_reply_text(update, reply)

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
    dp.add_handler(CommandHandler("name", name_command))
    dp.add_handler(MessageHandler(Filters.regex(r"^\.ping$"), ping))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_handler(MessageHandler(Filters.sticker, handle_sticker))
    dp.add_handler(MessageHandler(
        Filters.photo | Filters.video | Filters.document | Filters.voice | Filters.audio,
        handle_media
    ))
    dp.add_error_handler(error_handler)

    logging.info("Mitsuri is online and full of pyaar!")
    updater.start_polling()
    updater.idle()