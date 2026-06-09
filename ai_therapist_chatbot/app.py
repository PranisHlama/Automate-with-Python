import csv
import json
import re
from pathlib import Path

import requests
import zipfile
import faiss
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer


APP_DIR = Path(__file__).resolve().parent
METADATA_FILE = APP_DIR / "./mental-health-metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TOP_K = 2
SIMILARITY_THRESHOLD = 0.45

QUESTION_COLUMNS = (
    "question",
    "input",
    "prompt",
    "instruction",
    "user",
    "client",
    "human",
)
ANSWER_COLUMNS = (
    "answer",
    "output",
    "response",
    "assistant",
    "therapist",
    "completion",
)

CRISIS_PATTERNS = (
    r"\b(suicide|suicidal|self[-\s]?harm|overdose)\b",
    r"\b(kill|hurt|harm|cut)\s+(myself|me)\b",
    r"\b(kill|hurt|harm)\s+(someone|somebody|others?|another person)\b",
    r"\b(end my life|take my life|want to die|can'?t go on)\b",
    r"\b(do not want to live|don'?t want to live)\b",
)

CRISIS_RESPONSE = (
    "I'm really sorry you're dealing with this. If you might hurt yourself or "
    "someone else, call emergency services now. In the U.S. and its territories, "
    "you can call or text 988, or chat at https://988lifeline.org/chat/. If you "
    "can, move away from anything you could use to harm yourself and contact "
    "someone you trust right now."
)

st.set_page_config(page_title="Theraplex Support Chat")
st.title("Theraplex")


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")

def first_matching_value(row, column_map, candidates):
    for candidate in candidates:
        source_column = column_map.get(normalize_column_name(candidate))
        if source_column is None:
            continue

        value = row.get(source_column)
        if value is None:
            continue

        value = str(value).strip()
        if value:
            return value

    return ""


def clean_row(section, question, answer):
    question = str(question).strip()
    answer = str(answer).strip()

    if not question or not answer:
        return None

    return {
        "section": section,
        "question": question,
        "answer": answer,
    }


def download_dataset_from_croissant(metadata, app_dir):
    zip_url = None

    for item in metadata.get("distribution", []):
        content_url = item.get("content_url", "")
        encoding = item.get("encoding Format", "")

        if content_url.startswith("http") and encoding == "application/zip":
            zip_url = content_url
            break

    if not zip_url:
        return 
    
    zip_path =  app_dir / "archive.zip"

    if not zip_path.exists():
        response = requests.get(zip_url, timeout=120)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(app_dir)


def extract_referenced_dataset_names(metadata):
    names = set()

    def walk(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"contentUrl", "includes", "name"} and isinstance(nested, str):
                    if nested.endswith((".csv", ".json")):
                        names.add(Path(nested).name)
                walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(metadata)
    return sorted(names)


def extract_croissant_file_map(metadata):
    file_map = {}

    for item in metadata.get("distribution", []):
        if item.get("@type") != "cr:FileObject":
            continue

        content_url = item.get("contentUrl")
        if not isinstance(content_url, str) or not content_url.strip():
            continue

        file_map[item.get("@id", "")] = Path(content_url).name

    return file_map


def extract_croissant_record_sets(metadata):
    record_sets = {}

    for record_set in metadata.get("recordSet", []):
        file_object_id = None
        columns = []

        for field in record_set.get("field", []):
            source = field.get("source", {})
            field_file_object = source.get("fileObject", {}).get("@id")
            if field_file_object:
                file_object_id = field_file_object

            extract = source.get("extract", {})
            column = extract.get("column")
            if isinstance(column, str) and column.strip():
                columns.append(column.strip())

        record_set_name = record_set.get("name") or record_set.get("@id")
        if file_object_id and columns and record_set_name:
            record_sets[record_set_name] = {
                "file_object_id": file_object_id,
                "columns": columns,
            }

    return record_sets


def extract_qa_from_json(data, section="general"):
    rows = []

    def walk(value, current_section):
        if isinstance(value, dict):
            column_map = {normalize_column_name(key): key for key in value}
            question = first_matching_value(value, column_map, QUESTION_COLUMNS)
            answer = first_matching_value(value, column_map, ANSWER_COLUMNS)
            row = clean_row(current_section, question, answer)

            if row:
                rows.append(row)

            for key, nested in value.items():
                walk(nested, key)
        elif isinstance(value, list):
            for item in value:
                walk(item, current_section)

    walk(data, section)
    return rows


def load_rows_from_csv(path):
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return rows

        column_map = {
            normalize_column_name(column): column
            for column in reader.fieldnames
            if column
        }

        for record in reader:
            question = first_matching_value(record, column_map, QUESTION_COLUMNS)
            answer = first_matching_value(record, column_map, ANSWER_COLUMNS)
            row = clean_row(path.stem, question, answer)

            if row:
                rows.append(row)

    return rows


def load_rows_from_json(path):
    with path.open("r", encoding="utf-8") as file:
        return extract_qa_from_json(json.load(file), path.stem)


def dedupe_rows(rows):
    seen = set()
    unique_rows = []

    for row in rows:
        key = (row["question"].casefold(), row["answer"].casefold())
        if key in seen:
            continue

        seen.add(key)
        unique_rows.append(row)

    return unique_rows


@st.cache_data(show_spinner=False)
def load_therapy_rows(_file_signature):
    referenced_names = []
    croissant_files = {}
    croissant_record_sets = {}
    if METADATA_FILE.exists():
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
            referenced_names = extract_referenced_dataset_names(metadata)
            croissant_files = extract_croissant_file_map(metadata)
            croissant_record_sets = extract_croissant_record_sets(metadata)
    
        data = download_dataset_from_croissant(metadata, APP_DIR)
        print(data)

    local_files = []
    for file_name in referenced_names:
        candidate = APP_DIR / file_name
        if candidate.exists() and candidate != METADATA_FILE:
            local_files.append(candidate)

    local_files.extend(path for path in APP_DIR.glob("*.csv"))
    local_files.extend(
        path
        for path in APP_DIR.glob("*.json")
        if path.name != METADATA_FILE.name
    )

    rows = []
    load_errors = []
    for path in sorted(set(local_files)):
        try:
            if path.suffix.lower() == ".csv":
                rows.extend(load_rows_from_csv(path))
            elif path.suffix.lower() == ".json" and path.name != METADATA_FILE.name:
                rows.extend(load_rows_from_json(path))
        except (csv.Error, json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            load_errors.append(f"{path.name}: {exc}")

    metadata_sources = []
    for record_set_name, spec in croissant_record_sets.items():
        file_name = croissant_files.get(spec["file_object_id"])
        if file_name:
            metadata_sources.append(
                f"{record_set_name} -> {file_name} ({', '.join(spec['columns'])})"
            )

    return dedupe_rows(rows), referenced_names, load_errors, metadata_sources


def dataset_file_signature():
    paths = [METADATA_FILE]
    paths.extend(APP_DIR.glob("*.csv"))
    paths.extend(APP_DIR.glob("*.json"))

    signature = []
    for path in sorted(set(paths)):
        if not path.exists():
            continue

        stat = path.stat()
        signature.append((path.name, stat.st_mtime_ns, stat.st_size))

    return tuple(signature)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Building search index...")
def build_index(questions):
    if not questions:
        raise ValueError("No questions found in dataset.")

    embeddings = model.encode(
        list(questions),
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def is_crisis_message(message):
    normalized = message.casefold()
    return any(re.search(pattern, normalized) for pattern in CRISIS_PATTERNS)


def format_context(row):
    return (
        f"Section: {row['section']}\n"
        f"Question: {row['question']}\n"
        f"Answer: {row['answer']}"
    )


def rag_answer(user_query):
    if is_crisis_message(user_query):
        return CRISIS_RESPONSE, []

    user_embedding = model.encode(
        [user_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    result_count = min(TOP_K, len(rows))
    scores, indices = index.search(user_embedding, result_count)
    best_score = float(scores[0][0])

    if best_score < SIMILARITY_THRESHOLD:
        return "I'm sorry, I can't help with that based on the available dataset.", []

    contexts = []
    for row_index in indices[0]:
        if row_index < 0:
            continue

        contexts.append(format_context(rows[row_index]))

    context_text = "\n\n".join(contexts)
    prompt = f"""
You are a concise, supportive mental-health information assistant.

Rules:
- Use only the provided context.
- Keep the answer short and direct.
- Do not diagnose the user.
- If the answer is not in the context, say:
  "I'm sorry, I can't help with that based on the available dataset."
- If the user may be in immediate danger, tell them to contact emergency
  services or a crisis hotline.

Context:
{context_text}

Question:
{user_query}
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise, supportive mental-health information "
                        "assistant. You do not diagnose users."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.1,
                "num_predict": 120,
                "num_ctx": 2048,
            },
        )
    except ollama.ResponseError as exc:
        return (
            f"Ollama could not use the local `{OLLAMA_MODEL}` model: {exc}. "
            f"Run `ollama pull {OLLAMA_MODEL}` and try again.",
            contexts,
        )
    except Exception:
        return (
            "I found relevant source material, but could not connect to Ollama. "
            "Make sure the Ollama app or service is running, then try again.",
            contexts,
        )

    answer = response.get("message", {}).get("content", "").strip()
    if not answer:
        return "The local model returned an empty response. Please try again.", contexts

    return answer, contexts


rows, referenced_files, load_errors, metadata_sources = load_therapy_rows(
    dataset_file_signature()
)

if load_errors:
    with st.sidebar.expander("Dataset load warnings"):
        for error in load_errors:
            st.warning(error)

if not rows:
    st.warning(
        "Using the built-in starter Q&A set because the Croissant-referenced CSV "
        "files are not present in this folder."
    )

if referenced_files:
    with st.sidebar.expander("Referenced dataset files"):
        st.code("\n".join(referenced_files))

if metadata_sources:
    with st.sidebar.expander("Croissant field mappings"):
        st.code("\n".join(metadata_sources))

model = load_model()
questions = tuple(row["question"] for row in rows)
index = build_index(questions)


if "chat" not in st.session_state:
    st.session_state.chat = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_chat_index" not in st.session_state:
    st.session_state.current_chat_index = None


with st.sidebar:
    st.title("Chat History")
    st.caption(f"Loaded {len(rows):,} Q&A examples")

    if st.button("+ New Chat"):
        if st.session_state.chat:
            if st.session_state.current_chat_index is None:
                title = st.session_state.chat[0]["content"][:30]
                if len(st.session_state.chat[0]["content"]) > 30:
                    title += "..."

                st.session_state.chat_sessions.append(
                    {
                        "title": title,
                        "messages": [message.copy() for message in st.session_state.chat],
                    }
                )
            else:
                st.session_state.chat_sessions[
                    st.session_state.current_chat_index
                ]["messages"] = [message.copy() for message in st.session_state.chat]

        st.session_state.chat = []
        st.session_state.current_chat_index = None
        st.rerun()

    st.divider()

    if st.session_state.chat_sessions:
        for idx, session in enumerate(reversed(st.session_state.chat_sessions)):
            original_index = len(st.session_state.chat_sessions) - 1 - idx

            if st.button(session["title"], key=f"chat_{original_index}"):
                st.session_state.chat = [
                    message.copy() for message in session["messages"]
                ]
                st.session_state.current_chat_index = original_index
                st.rerun()
    else:
        st.caption("No chats yet.")

    st.divider()

    if st.button("Clear History"):
        st.session_state.chat = []
        st.session_state.chat_sessions = []
        st.session_state.current_chat_index = None
        st.rerun()


for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask your question here")

if user_input:
    st.session_state.chat.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = rag_answer(user_input)

        st.markdown(answer)

        if sources:
            with st.expander("Sources used"):
                for source in sources:
                    st.info(source)

    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    if st.session_state.current_chat_index is not None:
        st.session_state.chat_sessions[
            st.session_state.current_chat_index
        ]["messages"] = [message.copy() for message in st.session_state.chat]

    st.rerun()
