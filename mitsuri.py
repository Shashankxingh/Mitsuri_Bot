"""
🌸 Mitsuri Bot - Complete Single File Version (FIXED)
All bugs fixed, thread-safe, production-ready!

Usage:
    1. Set environment variables in .env file
    2. python mitsuri_bot_single_file.py
"""

import asyncio
import datetime
import hashlib
import html
import logging
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from threading import Thread
from typing import Optional

import certifi
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from flask import Flask
from groq import AsyncGroq
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from sambanova import SambaNova
from telegram import Update, constants, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ==================== CONFIGURATION ====================

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
OWNER_ID = os.getenv("OWNER_ID")

# Admin group (now required, no hardcoded default)
ADMIN_GROUP_ID_STR = os.getenv("ADMIN_GROUP_ID")
if ADMIN_GROUP_ID_STR:
    ADMIN_GROUP_ID = int(ADMIN_GROUP_ID_STR)
else:
    ADMIN_GROUP_ID = None

# Model Configuration
MODEL_LARGE = os.getenv("MODEL_LARGE", "llama-3.3-70b-versatile")
MODEL_SMALL = os.getenv("MODEL_SMALL", "llama-3.1-8b-instant")

CEREBRAS_MODEL_LARGE = os.getenv("CEREBRAS_MODEL_LARGE", "llama-3.3-70b")
CEREBRAS_MODEL_SMALL = os.getenv("CEREBRAS_MODEL_SMALL", "llama3.1-8b")

SAMBANOVA_MODEL_LARGE = os.getenv("SAMBANOVA_MODEL_LARGE", "Meta-Llama-3.3-70B-Instruct")
SAMBANOVA_MODEL_SMALL = os.getenv("SAMBANOVA_MODEL_SMALL", "Meta-Llama-3.1-8B-Instruct")

PROVIDER_ORDER = [
    provider.strip().lower()
    for provider in os.getenv("PROVIDER_ORDER", "groq,cerebras,sambanova").split(",")
    if provider.strip()
]

# Performance & Rate Limiting
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "10"))
SMALL_TALK_MAX_TOKENS = int(os.getenv("SMALL_TALK_MAX_TOKENS", "4"))

MONGO_MAX_POOL_SIZE = int(os.getenv("MONGO_MAX_POOL_SIZE", "50"))
MONGO_MIN_POOL_SIZE = int(os.getenv("MONGO_MIN_POOL_SIZE", "10"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "6"))
MAX_HISTORY_STORED = int(os.getenv("MAX_HISTORY_STORED", "20"))

CACHE_COMMON_RESPONSES = os.getenv("CACHE_COMMON_RESPONSES", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# FIX: Safer broadcast rate limits
BROADCAST_BATCH_SIZE = int(os.getenv("BROADCAST_BATCH_SIZE", "25"))
BROADCAST_BATCH_DELAY = float(os.getenv("BROADCAST_BATCH_DELAY", "1.2"))

GROUP_COOLDOWN_SECONDS = int(os.getenv("GROUP_COOLDOWN_SECONDS", "3"))

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ==================== UTILITIES ====================


def format_text_to_html(text):
    """Convert text to safe HTML for Telegram with XSS protection."""
    if not text:
        return ""
    
    # FIX: Escape HTML entities first to prevent XSS
    text = html.escape(text)
    
    # Convert markdown-like formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    
    return text


def require_env():
    """Validate required environment variables."""
    missing = []
    
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not MONGO_URI:
        missing.append("MONGO_URI")
    if not OWNER_ID:
        missing.append("OWNER_ID")
    if not ADMIN_GROUP_ID_STR:
        missing.append("ADMIN_GROUP_ID")
    
    if missing:
        raise ValueError(
            "❌ Missing required environment variables: " + ", ".join(missing)
        )

    try:
        owner_id = int(OWNER_ID)
    except ValueError as exc:
        raise ValueError("❌ OWNER_ID must be a valid integer!") from exc
    
    if ADMIN_GROUP_ID is None:
        raise ValueError("❌ ADMIN_GROUP_ID must be set!")

    return owner_id


# ==================== AI PROVIDER ERRORS ====================


class ProviderError(Exception):
    """Base error for provider failures."""


class RateLimitError(ProviderError):
    """Provider is rate limiting."""


class TransientProviderError(ProviderError):
    """Temporary errors (timeouts, 5xx)."""


class PermanentProviderError(ProviderError):
    """Permanent errors (invalid auth, invalid request)."""


# ==================== AI PROVIDERS ====================


@dataclass
class ProviderResult:
    content: str
    provider: str


class Provider:
    name: str

    async def generate(self, messages, model, temperature, max_tokens, top_p):
        raise NotImplementedError


class GroqProvider(Provider):
    name = "groq"

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("❌ Missing GROQ_API_KEY")
        self.client = AsyncGroq(api_key=GROQ_API_KEY)

    async def generate(self, messages, model, temperature, max_tokens, top_p):
        try:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            content = completion.choices[0].message.content.strip()
            return ProviderResult(content=content, provider=self.name)
        except Exception as exc:
            logger.warning("Groq error: %s", exc)
            status = getattr(exc, "status_code", None)
            if status == 429 or "rate limit" in str(exc).lower():
                raise RateLimitError(str(exc)) from exc
            if status and status >= 500:
                raise TransientProviderError(str(exc)) from exc
            raise PermanentProviderError(str(exc)) from exc


class CerebrasProvider(Provider):
    name = "cerebras"

    def __init__(self):
        if not CEREBRAS_API_KEY:
            raise ValueError("❌ Missing CEREBRAS_API_KEY")
        self.client = Cerebras(api_key=CEREBRAS_API_KEY)

    async def generate(self, messages, model, temperature, max_tokens, top_p):
        try:
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )
            content = completion.choices[0].message.content.strip()
            return ProviderResult(content=content, provider=self.name)
        except Exception as exc:
            logger.warning("Cerebras error: %s", exc)
            status = getattr(exc, "status_code", None)
            if status == 429 or "rate limit" in str(exc).lower():
                raise RateLimitError(str(exc)) from exc
            if status and status >= 500:
                raise TransientProviderError(str(exc)) from exc
            raise PermanentProviderError(str(exc)) from exc


class SambaNovaProvider(Provider):
    name = "sambanova"

    def __init__(self):
        if not SAMBANOVA_API_KEY:
            raise ValueError("❌ Missing SAMBANOVA_API_KEY")
        self.client = SambaNova(
            api_key=SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1",
        )

    async def generate(self, messages, model, temperature, max_tokens, top_p):
        try:
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content.strip()
            return ProviderResult(content=content, provider=self.name)
        except Exception as exc:
            logger.warning("SambaNova error: %s", exc)
            status = getattr(exc, "status_code", None)
            if status == 429 or "rate limit" in str(exc).lower():
                raise RateLimitError(str(exc)) from exc
            if status and status >= 500:
                raise TransientProviderError(str(exc)) from exc
            raise PermanentProviderError(str(exc)) from exc


# ==================== AI FALLBACK SYSTEM ====================


class ProviderFallback:
    def __init__(self, providers, model_resolver, max_attempts=2, backoff_seconds=1):
        self.providers = providers
        self.model_resolver = model_resolver
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    async def generate(self, messages, use_large, temperature, max_tokens, top_p):
        last_error = None
        for provider in self.providers:
            model = self.model_resolver(provider.name, use_large)
            for attempt in range(self.max_attempts):
                try:
                    result = await provider.generate(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    logger.info("✅ AI response from %s", result.provider)
                    return result
                except RateLimitError as exc:
                    last_error = exc
                    logger.warning("⚠️ %s rate limited; skipping retries", provider.name)
                    break
                except TransientProviderError as exc:
                    last_error = exc
                    logger.warning(
                        "⚠️ %s transient error (attempt %s/%s)",
                        provider.name,
                        attempt + 1,
                        self.max_attempts,
                    )
                except PermanentProviderError as exc:
                    last_error = exc
                    logger.error("❌ %s permanent error: %s", provider.name, exc)
                    break

                await asyncio.sleep(self.backoff_seconds)

            logger.info("➡️ Falling back from %s to next provider", provider.name)

        if last_error:
            raise last_error
        raise RuntimeError("No providers configured")


def build_fallback(model_resolver):
    """Build AI provider fallback system."""
    providers = []
    for provider_name in PROVIDER_ORDER:
        try:
            if provider_name == "groq":
                providers.append(GroqProvider())
            elif provider_name == "cerebras":
                providers.append(CerebrasProvider())
            elif provider_name == "sambanova":
                providers.append(SambaNovaProvider())
        except ValueError as exc:
            logger.warning("Skipping %s provider: %s", provider_name, exc)
    return ProviderFallback(providers=providers, model_resolver=model_resolver)


# ==================== THREAD-SAFE IN-MEMORY CACHE ====================


class InMemoryCache:
    """
    Thread-safe in-memory cache (100% FREE, no Redis needed!).
    FIX: Added threading locks for all operations.
    """
    
    def __init__(self):
        # FIX: Thread locks for all shared data structures
        self._rate_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._common_lock = threading.Lock()
        self._cooldown_lock = threading.Lock()
        self._broadcast_lock = threading.Lock()
        
        self.rate_limits = defaultdict(list)
        self.response_cache = {}
        self.common_cache = {}
        self.group_cooldowns = {}
        self.broadcasts = {}
        self.last_cleanup = time.time()
        
        logger.info("✅ Thread-safe in-memory cache initialized (FREE mode)")
    
    async def initialize(self):
        logger.info("💾 Running in FREE mode (thread-safe in-memory cache)")
    
    async def close(self):
        pass
    
    async def check_rate_limit(self, user_id: int) -> bool:
        """Thread-safe rate limit check."""
        now = time.time()
        
        with self._rate_lock:
            timestamps = self.rate_limits[user_id]
            timestamps[:] = [ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW]
            
            if len(timestamps) >= RATE_LIMIT_MAX:
                return False
            
            timestamps.append(now)
        
        await self._cleanup_if_needed()
        return True
    
    async def check_group_cooldown(self, chat_id: int, cooldown_seconds: int) -> bool:
        """Thread-safe group cooldown check."""
        now = time.time()
        
        with self._cooldown_lock:
            last_time = self.group_cooldowns.get(chat_id, 0)
            
            if now - last_time < cooldown_seconds:
                return False
            
            self.group_cooldowns[chat_id] = now
            return True
    
    def _generate_cache_key(self, chat_id: int, message: str) -> str:
        msg_hash = hashlib.md5(message.lower().strip().encode()).hexdigest()[:16]
        return f"response:{chat_id}:{msg_hash}"
    
    async def get_cached_response(self, chat_id: int, message: str) -> Optional[str]:
        """Thread-safe cached response retrieval."""
        key = self._generate_cache_key(chat_id, message)
        
        with self._cache_lock:
            if key in self.response_cache:
                response, expiry = self.response_cache[key]
                
                if time.time() < expiry:
                    logger.info("💾 Cache HIT for chat %s", chat_id)
                    return response
                else:
                    del self.response_cache[key]
        
        return None
    
    async def cache_response(self, chat_id: int, message: str, response: str):
        """Thread-safe response caching."""
        key = self._generate_cache_key(chat_id, message)
        expiry = time.time() + CACHE_TTL_SECONDS
        
        with self._cache_lock:
            self.response_cache[key] = (response, expiry)
    
    async def get_common_response(self, message: str) -> Optional[str]:
        """Thread-safe common response retrieval."""
        normalized = message.lower().strip()
        key = hashlib.md5(normalized.encode()).hexdigest()[:16]
        
        with self._common_lock:
            if key in self.common_cache:
                response, expiry = self.common_cache[key]
                
                if time.time() < expiry:
                    logger.info("💾 Common response cache HIT")
                    return response
                else:
                    del self.common_cache[key]
        
        return None
    
    async def cache_common_response(self, message: str, response: str):
        """Thread-safe common response caching."""
        normalized = message.lower().strip()
        key = hashlib.md5(normalized.encode()).hexdigest()[:16]
        expiry = time.time() + 86400
        
        with self._common_lock:
            self.common_cache[key] = (response, expiry)
    
    async def _cleanup_if_needed(self):
        """Thread-safe periodic cleanup. FIX: Use list() to prevent iteration errors."""
        now = time.time()
        
        if now - self.last_cleanup < 300:
            return
        
        self.last_cleanup = now
        
        with self._cache_lock:
            expired_keys = [
                key for key, (_, expiry) in list(self.response_cache.items())
                if now > expiry
            ]
            for key in expired_keys:
                del self.response_cache[key]
        
        with self._common_lock:
            expired_common = [
                key for key, (_, expiry) in list(self.common_cache.items())
                if now > expiry
            ]
            for key in expired_common:
                del self.common_cache[key]
        
        with self._broadcast_lock:
            old_broadcasts = [
                bid for bid, data in list(self.broadcasts.items())
                if now - data.get("started", 0) > 3600
            ]
            for bid in old_broadcasts:
                del self.broadcasts[bid]
        
        with self._cooldown_lock:
            old_cooldowns = [
                chat_id for chat_id, last_time in list(self.group_cooldowns.items())
                if now - last_time > 3600
            ]
            for chat_id in old_cooldowns:
                del self.group_cooldowns[chat_id]
        
        with self._rate_lock:
            inactive_users = [
                user_id for user_id, timestamps in list(self.rate_limits.items())
                if not timestamps or now - timestamps[-1] > RATE_LIMIT_WINDOW * 2
            ]
            for user_id in inactive_users:
                del self.rate_limits[user_id]


cache = InMemoryCache()


# ==================== DATABASE OPERATIONS ====================


def create_mongo_client():
    """Create MongoDB client with connection pooling."""
    try:
        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsCAFile=certifi.where(),
            maxPoolSize=MONGO_MAX_POOL_SIZE,
            minPoolSize=MONGO_MIN_POOL_SIZE,
            retryWrites=True,
            retryReads=True,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
        )
        mongo_client.admin.command("ping")
        logger.info("✅ MongoDB connected with pool size: %d-%d", 
                   MONGO_MIN_POOL_SIZE, MONGO_MAX_POOL_SIZE)
        return mongo_client
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        logger.critical("❌ MongoDB connection failed: %s", exc)
        raise


def initialize_indexes(db):
    """Create database indexes for optimal performance."""
    logger.info("🔧 Creating database indexes...")
    
    chat_collection = db["chat_info"]
    history_collection = db["chat_history"]
    
    chat_collection.create_index([("chat_id", ASCENDING)], unique=True)
    chat_collection.create_index([("type", ASCENDING)])
    chat_collection.create_index([("last_active", DESCENDING)])
    
    history_collection.create_index([
        ("chat_id", ASCENDING),
        ("timestamp", DESCENDING)
    ])
    history_collection.create_index([("timestamp", ASCENDING)])
    
    logger.info("✅ Database indexes created successfully!")


def save_user(chat_collection, update):
    """Save or update user/chat information."""
    try:
        chat = update.effective_chat
        user = update.effective_user
        
        data = {
            "chat_id": chat.id,
            "type": chat.type,
            "last_active": datetime.datetime.utcnow(),
        }
        
        if user:
            data["username"] = user.username
            data["first_name"] = user.first_name

        chat_collection.update_one(
            {"chat_id": chat.id},
            {"$set": data},
            upsert=True
        )
    except Exception as exc:
        logger.error("❌ DB Error in save_user: %s", exc)


def get_chat_history(history_collection, chat_id):
    """Retrieve chat history with optimized query."""
    try:
        history_docs = (
            history_collection
            .find(
                {"chat_id": chat_id},
                {"_id": 0, "role": 1, "content": 1}
            )
            .sort("timestamp", DESCENDING)
            .limit(HISTORY_LIMIT)
        )

        history = []
        for doc in reversed(list(history_docs)):
            history.append((doc["role"], doc["content"]))
        
        return history
    except Exception as exc:
        logger.error("❌ Error retrieving history: %s", exc)
        return []


def save_chat_history(history_collection, chat_id, role, content):
    """Save chat history."""
    try:
        history_collection.insert_one({
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.utcnow(),
        })
    except Exception as exc:
        logger.error("❌ Error saving history: %s", exc)


def get_all_chat_ids(chat_collection, batch_size=100):
    """Generator that yields chat IDs in batches for broadcasting."""
    try:
        cursor = chat_collection.find(
            {},
            {"chat_id": 1, "_id": 0},
            no_cursor_timeout=True
        )
        
        batch = []
        for doc in cursor:
            batch.append(doc["chat_id"])
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
            
        cursor.close()
    except Exception as exc:
        logger.error("❌ Error fetching chat IDs: %s", exc)
        yield []


def get_stats(chat_collection, history_collection):
    """Get bot statistics."""
    try:
        user_count = chat_collection.count_documents({"type": "private"})
        group_count = chat_collection.count_documents({"type": {"$ne": "private"}})
        total_messages = history_collection.estimated_document_count()
        
        return {
            "users": user_count,
            "groups": group_count,
            "messages": total_messages
        }
    except Exception as exc:
        logger.error("❌ Error fetching stats: %s", exc)
        return {"users": 0, "groups": 0, "messages": 0}


# ==================== BACKGROUND WORKER ====================


class BackgroundWorker:
    """Handles background tasks like history cleanup."""
    
    def __init__(self, history_collection):
        self.history_collection = history_collection
        self.cleanup_task = None
        self.running = False
    
    async def start(self):
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("✅ Background worker started")
    
    async def stop(self):
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Background worker stopped")
    
    async def _cleanup_loop(self):
        while self.running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._cleanup_old_history()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("❌ Error in cleanup loop: %s", exc)
    
    async def _cleanup_old_history(self):
        try:
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
            
            result = await asyncio.to_thread(
                self.history_collection.delete_many,
                {"timestamp": {"$lt": cutoff_date}}
            )
            
            if result.deleted_count > 0:
                logger.info("🧹 Cleaned up %d old history entries", result.deleted_count)
        except Exception as exc:
            logger.error("❌ Error cleaning old history: %s", exc)


# ==================== BOT HANDLERS ====================


SMALL_TALK_PATTERNS = re.compile(
    r"^(hi|hello|hey|hii|yo|sup|how are you|how r u|good morning|good night|"
    r"good evening|hola|namaste|hey there|hi there|hlo|wassup|whats up|"
    r"how's it going)\b",
    re.IGNORECASE,
)


def is_small_talk(text):
    """Detect if message is small talk to use faster model."""
    tokens = re.findall(r"\w+|[^\w\s]", text)
    token_count = len(tokens)
    if token_count <= SMALL_TALK_MAX_TOKENS:
        return True
    return bool(SMALL_TALK_PATTERNS.match(text.strip()))


def resolve_model(provider_name, use_large):
    """Map provider names to their model strings."""
    if provider_name == "cerebras":
        return CEREBRAS_MODEL_LARGE if use_large else CEREBRAS_MODEL_SMALL
    if provider_name == "sambanova":
        return SAMBANOVA_MODEL_LARGE if use_large else SAMBANOVA_MODEL_SMALL
    return MODEL_LARGE if use_large else MODEL_SMALL


@dataclass
class BotState:
    chat_collection: object
    history_collection: object
    owner_id: int
    admin_group_id: int
    provider_fallback: object


def build_state(chat_collection, history_collection, owner_id, admin_group_id):
    """Build bot state with optimized components."""
    return BotState(
        chat_collection=chat_collection,
        history_collection=history_collection,
        owner_id=owner_id,
        admin_group_id=admin_group_id,
        provider_fallback=build_fallback(resolve_model),
    )


async def get_ai_response(state, history, user_input, user_name):
    """Get AI response with caching. FIX: Added response validation."""
    system_prompt = (
        "You are Mitsuri Kanroji from Demon Slayer. "
        "Personality: Romantic, bubbly, cheerful, and sweet. Use emojis sparingly (🍡, 💖). "
        "Language: Hinglish (mix of Hindi and English). "
        "Keep responses concise and natural - around 1-3 sentences. Be warm and friendly!"
    )

    if CACHE_COMMON_RESPONSES and is_small_talk(user_input):
        cached = await cache.get_common_response(user_input)
        if cached:
            return cached

    messages = [{"role": "system", "content": system_prompt}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": f"{user_input} (User: {user_name})"})

    use_large = not is_small_talk(user_input)
    
    try:
        result = await state.provider_fallback.generate(
            messages=messages,
            use_large=use_large,
            temperature=0.8,
            max_tokens=150,
            top_p=0.9,
        )
        
        # FIX: Validate response is not empty
        if not result or not result.content or not result.content.strip():
            logger.error("❌ Empty AI response received")
            return "Ah! Something went wrong... 😵‍💫 Please try again!"
        
        response = result.content.strip()
        
        if CACHE_COMMON_RESPONSES and is_small_talk(user_input):
            await cache.cache_common_response(user_input, response)
        
        return response
    except Exception as exc:
        logger.error("❌ Provider fallback failed: %s", exc)
        return "Ah! Something went wrong... 😵‍💫 Please try again!"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    logger.info("🚀 /start triggered by %s (ID: %s)", user.first_name, user.id)
    
    state = context.bot_data["state"]
    await asyncio.to_thread(save_user, state.chat_collection, update)

    welcome_msg = (
        "Kyaa~! 💖 Hii! I am <b>Mitsuri Kanroji</b>!\n\n"
        "I love making new friends! Let's chat and eat mochi together! 🍡\n\n"
        "Use /help to see what I can do~"
    )
    await update.message.reply_html(welcome_msg)


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ping command."""
    user = update.effective_user
    logger.info("🏓 /ping triggered by %s (ID: %s)", user.first_name, user.id)

    start_time = time.time()
    msg = await update.message.reply_text("🍡 Pinging...")
    end_time = time.time()
    bot_latency = (end_time - start_time) * 1000
    
    await msg.edit_text(
        f"🏓 <b>Pong!</b>\n\n"
        f"⚡ <b>Latency:</b> <code>{bot_latency:.2f}ms</code>",
        parse_mode="HTML",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    user = update.effective_user
    logger.info("ℹ️ /help requested by %s (ID: %s)", user.first_name, user.id)

    help_text = (
        "🌸 <b>Mitsuri's Help Menu</b> 🌸\n\n"
        "I am the Love Hashira! Here is what I can do:\n\n"
        "💬 <b>Chat:</b> Reply to me or mention me in groups!\n"
        "💌 <b>Private:</b> DM me to talk privately (smarter AI!).\n"
        "🗣️ <b>Language:</b> I speak Hinglish!\n"
        "⚡ <b>Utility:</b> Use /ping to check speed.\n\n"
        "<i>Just say 'Hi' to start chatting!</i> 💖"
    )

    state = context.bot_data["state"]
    if update.effective_user.id == state.owner_id:
        keyboard = [[InlineKeyboardButton("🔐 Admin Commands", callback_data="admin_help")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_html(help_text, reply_markup=reply_markup)
    else:
        await update.message.reply_html(help_text)


async def admin_button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin button press."""
    query = update.callback_query
    await query.answer()

    state = context.bot_data["state"]
    if query.from_user.id != state.owner_id:
        logger.warning("⚠️ Unauthorized admin button press by %s", query.from_user.id)
        return

    admin_text = (
        "<b>👑 Admin Commands</b>\n"
        "<i>(Only work in Admin Group)</i>\n\n"
        "• <code>/stats</code> - Check user counts\n"
        "• <code>/cast [msg]</code> - Broadcast message\n"
    )
    await query.message.reply_html(admin_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages."""
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    user = update.effective_user
    state = context.bot_data["state"]

    if not await cache.check_rate_limit(user.id):
        logger.warning("⚠️ Rate limit exceeded for user %s", user.id)
        return

    should_reply = False
    is_private = update.effective_chat.type == constants.ChatType.PRIVATE
    bot_username = context.bot.username

    if is_private:
        should_reply = True
    else:
        if (
            f"@{bot_username}" in text
            or (
                update.message.reply_to_message
                and update.message.reply_to_message.from_user.id == context.bot.id
            )
        ):
            should_reply = True
            text = text.replace(f"@{bot_username}", "").strip()
        elif "mitsuri" in text.lower():
            should_reply = True

    if not should_reply:
        return

    if not is_private:
        if not await cache.check_group_cooldown(chat_id, GROUP_COOLDOWN_SECONDS):
            logger.info("⏳ Cooldown active for group %s", chat_id)
            return

    model_type = "Small" if is_small_talk(text) else "Large"
    logger.info(
        "📩 [%s] Message from %s (ID: %s): %s...",
        model_type,
        user.first_name,
        user.id,
        text[:30],
    )

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    
    asyncio.create_task(
        asyncio.to_thread(save_user, state.chat_collection, update)
    )

    history = await asyncio.to_thread(get_chat_history, state.history_collection, chat_id)
    response = await get_ai_response(state, history, text, user.first_name)

    asyncio.create_task(
        asyncio.to_thread(save_chat_history, state.history_collection, chat_id, "user", text)
    )
    asyncio.create_task(
        asyncio.to_thread(save_chat_history, state.history_collection, chat_id, "assistant", response)
    )

    try:
        await update.message.reply_html(format_text_to_html(response))
        logger.info("📤 [%s] Sent reply to %s", model_type, chat_id)
    except Exception as exc:
        logger.error("❌ Failed to send HTML reply to %s: %s", chat_id, exc)
        try:
            await update.message.reply_text(response)
        except Exception as final_exc:
            logger.error("❌ Complete failure to send message to %s: %s", chat_id, final_exc)


def admin_group_only(func):
    """Decorator to restrict commands to admin group only."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        state = context.bot_data["state"]
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        if user_id != state.owner_id:
            logger.warning("⚠️ Unauthorized admin command attempt by %s", user_id)
            return
        
        if state.admin_group_id and chat_id != state.admin_group_id:
            await update.message.reply_text("⚠️ Admin commands only work in admin group!")
            return

        return await func(update, context, *args, **kwargs)

    return wrapper


@admin_group_only
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get bot statistics."""
    logger.info("📊 Admin requested stats")
    state = context.bot_data["state"]
    
    try:
        stats_data = await asyncio.to_thread(
            get_stats,
            state.chat_collection,
            state.history_collection
        )
        
        await update.message.reply_html(
            f"<b>📊 Mitsuri's Stats</b>\n\n"
            f"👤 <b>Users:</b> {stats_data['users']:,}\n"
            f"👥 <b>Groups:</b> {stats_data['groups']:,}\n"
            f"💬 <b>Total Messages:</b> {stats_data['messages']:,}"
        )
    except Exception as exc:
        logger.error("❌ Error fetching stats: %s", exc)
        await update.message.reply_text("Failed to fetch stats!")


@admin_group_only
async def cast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Broadcast message with safe rate limiting."""
    msg = " ".join(context.args)
    if not msg:
        await update.message.reply_text("Usage: /cast [Message]")
        return

    logger.info("📢 Starting broadcast: %s...", msg[:30])
    status_msg = await update.message.reply_text("🚀 Preparing broadcast...")
    state = context.bot_data["state"]
    
    formatted_msg = format_text_to_html(msg)
    
    success = 0
    failed = 0
    total = 0

    try:
        batch_num = 0
        
        for batch in get_all_chat_ids(state.chat_collection, BROADCAST_BATCH_SIZE):
            batch_num += 1
            total += len(batch)
            
            tasks = [
                context.bot.send_message(
                    chat_id=chat_id,
                    text=formatted_msg,
                    parse_mode="HTML",
                )
                for chat_id in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    success += 1
            
            if batch_num % 5 == 0:
                await status_msg.edit_text(
                    f"📤 Broadcasting...\n"
                    f"✅ Sent: {success:,}\n"
                    f"❌ Failed: {failed:,}\n"
                    f"📊 Progress: {success + failed:,} / ~{total:,}"
                )
            
            await asyncio.sleep(BROADCAST_BATCH_DELAY)
        
        logger.info("📢 Broadcast finished. Success: %d, Failed: %d", success, failed)
        
        await status_msg.edit_text(
            f"✅ <b>Broadcast Complete!</b>\n\n"
            f"📤 Sent: {success:,}\n"
            f"❌ Failed: {failed:,}\n"
            f"📊 Total: {total:,}",
            parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("❌ Broadcast error: %s", exc)
        await status_msg.edit_text("❌ Broadcast failed!")


# ==================== FLASK HEALTH CHECK ====================


app = Flask(__name__)


@app.route("/")
def health_check():
    return "Mitsuri is Alive! 🌸"


@app.route("/health")
def health_detailed():
    return {
        "status": "healthy",
        "service": "mitsuri-bot",
        "version": "2.0.1-fixed-single-file"
    }


def run_flask():
    """Run Flask server in background thread."""
    port = int(os.environ.get("PORT", 8080))
    import logging as flask_logging
    log = flask_logging.getLogger("werkzeug")
    log.disabled = True
    app.logger.disabled = True
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ==================== MAIN APPLICATION ====================


async def post_init(application):
    """Initialize async components after bot starts."""
    logger.info("🔧 Initializing async components...")
    
    await cache.initialize()
    
    state = application.bot_data["state"]
    background_worker = BackgroundWorker(state.history_collection)
    await background_worker.start()
    
    application.bot_data["background_worker"] = background_worker
    
    logger.info("✅ Async components initialized")


async def post_shutdown(application):
    """Cleanup async components on shutdown."""
    logger.info("🧹 Cleaning up async components...")
    
    if "background_worker" in application.bot_data:
        await application.bot_data["background_worker"].stop()
    
    await cache.close()
    
    # FIX: Close MongoDB connection
    if "mongo_client" in application.bot_data:
        application.bot_data["mongo_client"].close()
        logger.info("✅ MongoDB connection closed")
    
    logger.info("✅ Cleanup complete")


def main():
    """Main entry point."""
    owner_id = require_env()
    
    mongo_client = create_mongo_client()
    db = mongo_client["MitsuriDB"]
    chat_collection = db["chat_info"]
    history_collection = db["chat_history"]
    
    initialize_indexes(db)

    state = build_state(chat_collection, history_collection, owner_id, ADMIN_GROUP_ID)

    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logger.info("🌸 Mitsuri Bot is Starting...")
    logger.info("🧠 AI Models: Large=%s, Small=%s", MODEL_LARGE, MODEL_SMALL)
    logger.info("⚡ Performance Mode: OPTIMIZED + FIXED (Single File)")

    # FIX: concurrent_updates=False for thread safety
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(False)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    
    application.bot_data["state"] = state
    application.bot_data["mongo_client"] = mongo_client

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(CallbackQueryHandler(admin_button_callback, pattern="admin_help"))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("cast", cast))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🤖 Polling started. Mitsuri is ready! 💖")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
