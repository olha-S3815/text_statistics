
import sqlite3
from pathlib import Path
from collections import Counter

import tokenize_uk
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer

TEXTS_DIR = Path("texts")
DATABASE_FILE = "frequency.db"

morph = pymorphy2.MorphAnalyzer(lang="uk")

#завантаження текстів
def load_documents():
    """Зчитування TXT-файлів з папки texts."""

    files = sorted(TEXTS_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            "У папці texts немає TXT-файлів."
        )

    documents = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "name": file_path.name,
            "text": text
        })

    return documents

#обробка слів
def analyse_document(text):
    """
    Токенізує текст, підраховує частоти слів
    і визначає для кожного слова лему та частину мови.
    """

    tokens = tokenize_uk.tokenize_words(text)

    words = [
        token.lower()
        for token in tokens
        if token and any(char.isalpha() for char in token)
    ]

    frequencies = Counter(words)

    results = []

    for form, frequency in frequencies.items():
        parsed = morph.parse(form)[0] #морф аналіз
        lemma = parsed.normal_form
        pos = parsed.tag.POS or "Інша/невизначена"

        results.append({
            "form": form,
            "lemma": lemma,
            "pos": pos,
            "freq": frequency
        })

    return results, len(words)

# 3. Створення бази даних

def create_database():
    """Створює таблиці в базі даних SQLite."""

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    # Таблиця з інформацією про документи
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Documents (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            total_words INTEGER
        )
    """)

    # Таблиця з частотами словоформ
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS WordFreq (
            word_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            form TEXT NOT NULL,
            lemma TEXT NOT NULL,
            pos TEXT,
            freq INTEGER NOT NULL,
            FOREIGN KEY (document_id) REFERENCES Documents(document_id)
        )
    """)

    cursor.execute("DELETE FROM WordFreq")
    cursor.execute("DELETE FROM Documents")
    
    connection.commit()
    connection.close()


def save_document(name, total_words, word_data):
    """Запису документу і його характеристик в базу."""

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO Documents (name, total_words)
        VALUES (?, ?)
        """,
        (name, total_words)
    )

    document_id = cursor.lastrowid

    for word in word_data:
        cursor.execute(
            """
            INSERT INTO WordFreq
            (document_id, form, lemma, pos, freq)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                document_id,
                word["form"],
                word["lemma"],
                word["pos"],
                word["freq"]
            )
        )

    connection.commit()
    connection.close()

#статистика SQL
def show_statistics():
    """Виводить статистику за всіма обробленими документами."""

    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    print("\n========== ЗАГАЛЬНА СТАТИСТИКА ==========")

    cursor.execute(""" 
        SELECT name, total_words
        FROM Documents
    """)

    print("\nКількість слів у документах:")
    for name, total in cursor.fetchall():
        print(f"{name}: {total}")

    cursor.execute("""
        SELECT pos, SUM(freq)
        FROM WordFreq
        GROUP BY pos
        ORDER BY SUM(freq) DESC
    """)

    print("\nКількість слововживань за частинами мови:")
    for pos, count in cursor.fetchall():
        print(f"{pos}: {count}")

    cursor.execute("""
        SELECT pos, COUNT(*)
        FROM WordFreq
        GROUP BY pos
        ORDER BY COUNT(*) DESC
    """)

    print("\nКількість різних словоформ за частинами мови:")
    for pos, count in cursor.fetchall():
        print(f"{pos}: {count}")

    cursor.execute("""
        SELECT form, SUM(freq)
        FROM WordFreq
        GROUP BY form
        ORDER BY SUM(freq) DESC
        LIMIT 10
    """)

    print("\n10 найчастотніших словоформ:")
    for form, count in cursor.fetchall():
        print(f"{form}: {count}")

    cursor.execute("""
        SELECT lemma, SUM(freq)
        FROM WordFreq
        GROUP BY lemma
        ORDER BY SUM(freq) DESC
        LIMIT 10
    """)

    print("\n10 найчастотніших лем:")
    for lemma, count in cursor.fetchall():
        print(f"{lemma}: {count}")

    cursor.execute("""
        SELECT Documents.name, COUNT(DISTINCT WordFreq.lemma)
        FROM Documents
        JOIN WordFreq
        ON Documents.document_id = WordFreq.document_id
        GROUP BY Documents.document_id
    """)

    print("\nКількість різних лем у документах:")
    for name, count in cursor.fetchall():
        print(f"{name}: {count}")

    connection.close()

#TF-IDF
def calculate_tfidf(documents):
    """Розрахунок TF-IDF для всіх завантажених документів."""

    texts = [document["text"] for document in documents]

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_uk.tokenize_words,
        token_pattern=None,
        lowercase=True
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out()

    print("\n========== TF-IDF ==========")

    for doc_index, document in enumerate(documents):
        print(f"\nДокумент: {document['name']}")

        scores = tfidf_matrix[doc_index].toarray()[0] #TF-IDF для поточного документа

        sorted_indices = scores.argsort()[::-1]

        print("10 слів із найвищими показниками TF-IDF:")

        count = 0

        for index in sorted_indices:
            if scores[index] > 0:
                print(
                    f"{feature_names[index]}: "
                    f"{scores[index]:.4f}"
                )

                count += 1

                if count == 10:
                    break


def main():
    documents = load_documents()

    create_database()

    print("Початок обробки текстів...")

    for document in documents:
        print(f"Обробка: {document['name']}")

        word_data, total_words = analyse_document(
            document["text"]
        )

        save_document(
            document["name"],
            total_words,
            word_data
        )

    show_statistics()

    calculate_tfidf(documents)

    print("\nОбробку завершено!")

if __name__ == "__main__":
    main()