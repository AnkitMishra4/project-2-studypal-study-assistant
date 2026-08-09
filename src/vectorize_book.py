import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CLASS_SUBJECT_NAME = os.getenv("CLASS_SUBJECT_NAME")

DEVICE = os.getenv("DEVICE", "cpu")

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

data_dir = f"{parent_dir}/data"
vector_db_dir = f"{parent_dir}/vector_db"
chapters_vector_db_dir = f"{parent_dir}/chapters_vector_db"

embedding = HuggingFaceEmbeddings(
    model_kwargs={"device": DEVICE}
)

text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)


def vectorize_book_and_store_to_db(class_subject_name, vector_db_name):

    book_dir = f"{data_dir}/{class_subject_name}"
    vector_db_path = f"{vector_db_dir}/{vector_db_name}"

    all_documents = []

    for filename in os.listdir(book_dir):

        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(book_dir, filename)

        print(f"\nProcessing: {filename}")

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            all_documents.extend(documents)

            print(f"Successfully processed: {filename}")

        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
            continue

    if not all_documents:
        print("No PDF documents were successfully loaded.")
        return

    text_chunks = text_splitter.split_documents(all_documents)

    Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory=vector_db_path
    )

    print(
        f"{class_subject_name} saved to vector db: "
        f"{vector_db_name}"
    )


def vectorize_chapters(class_subject_name):

    book_dir = f"{data_dir}/{class_subject_name}"

    for chapter in os.listdir(book_dir):

        if not chapter.lower().endswith(".pdf"):
            continue

        chapter_name = chapter[:-4]
        chapter_pdf_path = os.path.join(book_dir, chapter)

        print(f"\nProcessing chapter: {chapter_name}")

        try:

            loader = PyPDFLoader(chapter_pdf_path)

            documents = loader.load()

            texts = text_splitter.split_documents(documents)

            Chroma.from_documents(
                documents=texts,
                embedding=embedding,
                persist_directory=(
                    f"{chapters_vector_db_dir}/{chapter_name}"
                )
            )

            print(f"{chapter_name} chapter vectorized")

        except Exception as e:

            print(
                f"ERROR processing chapter "
                f"{chapter_name}: {e}"
            )

            continue