import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# Стало:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Настройки ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в .env. Убедитесь, что он там есть.")

# Файл с Налоговым Кодексом РК
PDF_PATH = "nk_rk_2025.pdf"
# Имя файла для сохранения векторной базы данных
DB_FAISS_PATH = "faiss_vector_db"

# --- Параметры разбиения текста ---
# Размер одного фрагмента (чанка)
CHUNK_SIZE = 1000
# Перекрытие между чанками для сохранения контекста
CHUNK_OVERLAP = 200


def create_vector_db():
    """
    Загружает PDF-документ, разбивает его на чанки,
    создает эмбеддинги и сохраняет их в FAISS.
    """
    if not os.path.exists(PDF_PATH):
        print(f"Ошибка: Файл {PDF_PATH} не найден. Убедитесь, что вы его скачали.")
        return

    print("1. Загрузка документа...")
    # 1. Загрузка документа с помощью LangChain
    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
    except Exception as e:
        print(f"Ошибка при загрузке PDF: {e}")
        return

    print(f"   Загружено страниц: {len(documents)}")

    print("2. Разбиение документа на фрагменты (чанки)...")
    # 2. Разбиение на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"   Общее количество чанков: {len(texts)}")

    print("3. Создание эмбеддингов (векторизация) и сохранение в FAISS...")
    # 3. Создание эмбеддингов
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    # 4. Сохранение в векторной базе данных FAISS
    # Это занимает некоторое время, так как каждый чанк отправляется в API OpenAI
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    print("\n✅ Индексация завершена!")
    print(f"   Векторная база данных сохранена в папке: {DB_FAISS_PATH}")
    print("   Теперь можно использовать ее в основном боте (Шаг 4).")


if __name__ == "__main__":
    create_vector_db()