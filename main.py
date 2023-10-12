import logging
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.error import InvalidToken

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

DATABASE = None

# telegram_bot_token = str(input("Please enter your telegram bot token - ")).strip()
# openai_APIKEY = str(input("Please enter your openai API key - "))

load_dotenv()
telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
openai_APIKEY = os.environ.get('OPENAI_API_KEY')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, 
                                   text="I'm here to help, use me.")   # Input whatever message you want users to see when they start chating with the bot


async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loader = TextLoader("state.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents=documents)

    global DATABASE
    DATABASE = FAISS.from_documents(docs, OpenAIEmbeddings())

    await context.bot.send_message(chat_id=update.effective_chat.id, text="Document loaded!")


async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    complete_docs = DATABASE.similarity_search(update.message.text, k=4)
    # chain = load_qa_chain(llm=openai(), chain_type="stuff")
    chain = load_qa_chain(llm=OpenAI(temperature=1), chain_type="stuff")
    results = chain({"input documents": complete_docs, "question": update.message.text}, return_only_outputs=True)
    text = results["output_text"]

    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)



if __name__ == '__main__':
    try:
        application = ApplicationBuilder().token(telegram_bot_token).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CommandHandler('load', load))
        application.add_handler(CommandHandler('query', query))
        application.run_polling()
    except InvalidToken:
        print("You have entered an invalid token, check and try again.")