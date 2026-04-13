import random

# Optional: add more personality variations
GREETINGS = [
    "Hey! 👋 What can I help you with today?",
    "Hi there 😊 Need outfit ideas or just chatting?",
    "Hello! Ready to style something awesome? 😎",
]

SMALL_TALK = [
    "That’s interesting 😊 Tell me more.",
    "Got it! What else is on your mind?",
    "Hmm, sounds cool 👀",
]

JOKES = [
    "Why did the shirt go to therapy? It had too many issues 😄",
    "Why don’t sneakers ever argue? They always lace things up nicely 😆",
    "Why did the wardrobe break up? Too many hang-ups 😂",
]


def lightweight_chat(text: str) -> str:
    t = text.lower().strip()

    # -------------------------
    # GREETINGS
    # -------------------------
    if any(x in t for x in ["hi", "hello", "hey"]):
        return random.choice(GREETINGS)

    # -------------------------
    # HOW ARE YOU
    # -------------------------
    if "how are you" in t:
        return "I’m doing great 😄 Ready to help you look your best!"

    # -------------------------
    # IDENTITY
    # -------------------------
    if "who are you" in t:
        return "I’m your AI stylist + assistant 👗 I can help with outfits, planning, or just chat!"

    # -------------------------
    # NAME
    # -------------------------
    if "your name" in t:
        return "You can call me your style buddy 😎"

    # -------------------------
    # JOKES
    # -------------------------
    if "joke" in t:
        return random.choice(JOKES)

    # -------------------------
    # FASHION-RELATED SMALL TALK
    # -------------------------
    if any(x in t for x in ["look good", "style", "fashion"]):
        return "Style is all about confidence 😎 Want me to suggest an outfit?"

    # -------------------------
    # EMOTIONAL RESPONSES
    # -------------------------
    if any(x in t for x in ["sad", "tired", "bored"]):
        return "Ah, one of those days huh 🥲 Maybe a fresh outfit can lift the mood?"

    if any(x in t for x in ["happy", "excited"]):
        return "Love that energy 🔥 Let’s match it with a killer outfit!"

    # -------------------------
    # DEFAULT SMART FALLBACK
    # -------------------------
    return random.choice(SMALL_TALK)
