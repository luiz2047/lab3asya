import logging
from telegram import Update, Audio
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
from modules.intent_classifier import IntentClassifier
from modules.dialogue_manager import DialogueManager
from modules.voice_handler import VoiceHandler
from modules.ad_manager import AdManager
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)

intent_classifier = IntentClassifier()
dialogue_manager = DialogueManager()
voice_handler = VoiceHandler()
ad_manager = AdManager()


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Я ваш чат-бот. Чем могу помочь?")


async def handle_message(update: Update, context: CallbackContext):
    text = update.message.text
    intent = intent_classifier.classify(text)
    response = dialogue_manager.respond(text, intent)
    ad_response = ad_manager.check_for_ad(intent)
    await update.message.reply_text(response + '\n' + ad_response)


async def handle_voice(update: Update, context: CallbackContext):
    file = await update.message.voice.get_file()
    file_path = os.path.join('temp', f"{file.file_id}.ogg")
    await file.download_to_drive(file_path)
    print(file_path)
    text = await voice_handler.recognize_speech(file_path)
    os.remove(file_path)

    intent = intent_classifier.classify(text)
    response = dialogue_manager.respond(text, intent)
    ad_response = ad_manager.check_for_ad(intent)
    await update.message.reply_text(f"You said: {text}\n\n{response}\n{ad_response}")


def main():
    application = ApplicationBuilder().token(os.getenv("TELEGRAM_API_TOKEN")).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logging.info("Бот запущен...")
    application.run_polling()


if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    main()
