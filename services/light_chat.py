def lightweight_chat(text: str) -> str:
    text = text.lower()

    if "how are you" in text:
        return "I'm doing great 😊 What about you?"

    if "who are you" in text:
        return "I'm your AI stylist + assistant. I can help with outfits or just chat!"

    if "joke" in text:
        return "Why did the shirt go to therapy? It had too many issues 😄"

    if "what is your name" in text:
        return "You can call me your style buddy 😎"

    if "hello" in text or "hi" in text:
        return "Hey! 👋 What can I help you with today?"

    return "That's interesting! Tell me more 😊"
