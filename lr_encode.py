"""
Step 1: Encode claims and reviews with the local MiniLM ONNX model,
then save encodings to a file for training.
"""
import json
import csv
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import onnxruntime as ort

# ----- Sample counts: claims (one class), not-a-claim from reviews + questions + dialogue -----
N_CLAIMS = 30_000
N_REVIEWS = 10_000
N_QUESTIONS = 10_000
N_DIALOGUE_DECL = 10_000
N_DIALOGUE_IMP = 10_000

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DATASETS_DIR = PROJECT_ROOT / "src" / "public" / "datasets"
CLAIMS_PATH = DATASETS_DIR / "verifiableClaims.jsonl"
REVIEWS_PATH = DATASETS_DIR / "Reviews.csv"
QUESTIONS_PATH = DATASETS_DIR / "questions.json"
SPAADIA_DIR = DATASETS_DIR / "spaadia_release_v02"
MODEL_DIR = PROJECT_ROOT / "src" / "public" / "models" / "all-MiniLM-L6-v2"
ENCODINGS_PATH = PROJECT_ROOT / "claim_encodings.npz"

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


def load_claims(path: Path, max_samples: int | None = None) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(texts) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "claim" in obj:
                texts.append(obj["claim"])
    return texts


def load_reviews(path: Path, max_samples: int | None = None) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_col = None
        for name in ("text", "Text", "review", "Review", "Review_Text", "content", "Comment", "comment"):
            if name in fieldnames:
                text_col = name
                break
        if text_col is None and fieldnames:
            text_col = fieldnames[0]
        if text_col is None:
            raise ValueError("Reviews CSV has no columns")
        for row in reader:
            if max_samples is not None and len(texts) >= max_samples:
                break
            t = (row.get(text_col) or "").strip()
            if t:
                texts.append(t)
    return texts


def load_questions(path: Path, max_samples: int | None = None) -> list[str]:
    """Load questions from SQuAD-style JSON (data[].paragraphs[].qas[].question)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for item in data.get("data", []):
        if max_samples is not None and len(texts) >= max_samples:
            break
        for para in item.get("paragraphs", []):
            if max_samples is not None and len(texts) >= max_samples:
                break
            for qa in para.get("qas", []):
                if "question" in qa:
                    q = (qa["question"] or "").strip()
                    if q:
                        texts.append(q)
                if max_samples is not None and len(texts) >= max_samples:
                    break
    return texts[:max_samples] if max_samples is not None else texts


def load_spaadia_sentences(
    dir_path: Path,
    max_decl: int | None = None,
    max_imp: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load declarative and imperative sentences from the Spaadia XML dialogues.

    Declaratives are identified by either a tag name ``decl`` or a ``mode``
    attribute containing ``"decl"``. Imperatives are identified by either a
    tag name ``imp`` or a ``mode`` attribute containing ``"imp"``.
    Up to ``max_decl`` and ``max_imp`` samples are returned (or all available
    if the corresponding limit is None).
    """
    declaratives: list[str] = []
    imperatives: list[str] = []

    if not dir_path.exists():
        return declaratives, imperatives

    for xml_path in sorted(dir_path.glob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()

        for elem in root.iter():
            mode = elem.attrib.get("mode", "")
            tag = elem.tag
            text_parts = list(elem.itertext())
            text = " ".join(" ".join(text_parts).split()).strip()
            if not text:
                continue

            added_any = False
            if max_decl is None or len(declaratives) < max_decl:
                if tag == "decl" or "decl" in mode:
                    declaratives.append(text)
                    added_any = True

            if max_imp is None or len(imperatives) < max_imp:
                if tag == "imp" or "imp" in mode:
                    imperatives.append(text)
                    added_any = True

            if (
                (max_decl is not None and len(declaratives) >= max_decl)
                and (max_imp is not None and len(imperatives) >= max_imp)
            ):
                break

        if (
            (max_decl is not None and len(declaratives) >= max_decl)
            and (max_imp is not None and len(imperatives) >= max_imp)
        ):
            break

    return declaratives, imperatives


def encode_with_onnx(texts: list[str], model_dir: Path) -> np.ndarray:
    """Encode texts using the local MiniLM ONNX model. Returns (n, 384) float32."""
    onnx_path = model_dir / "model.onnx"
    tokenizer_path = model_dir / "tokenizer.json"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not tokenizer_path.exists() or Tokenizer is None:
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path} (and pip install tokenizers)"
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    max_length = 256

    all_embeddings = []
    for i, text in enumerate(texts):
        if (i + 1) % 1000 == 0:
            print(f"  Encoded {i + 1}/{len(texts)}...")
        enc = tokenizer.encode(text, add_special_tokens=True)
        input_ids = enc.ids[:max_length]
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < max_length:
            padding = max_length - len(input_ids)
            input_ids = input_ids + [0] * padding
            attention_mask = attention_mask + [0] * padding

        feed = {}
        if "input_ids" in input_names:
            feed["input_ids"] = np.array([input_ids], dtype=np.int64)
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.array([attention_mask], dtype=np.int64)
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros((1, max_length), dtype=np.int64)

        out = session.run(None, feed)
        last_hidden = out[0]
        if last_hidden.ndim == 2:
            emb = last_hidden.flatten()
        else:
            mask = np.array(attention_mask, dtype=np.float32).reshape(1, -1, 1)
            summed = (last_hidden * mask).sum(axis=1)
            counts = np.maximum(mask.sum(axis=1), 1e-9)
            emb = (summed / counts).flatten()
        emb = emb / np.maximum(np.linalg.norm(emb), 1e-9)
        all_embeddings.append(emb)

    return np.array(all_embeddings, dtype=np.float32)


def main():
    if not CLAIMS_PATH.exists():
        raise FileNotFoundError(f"Claims file not found: {CLAIMS_PATH}")
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Reviews file not found: {REVIEWS_PATH}")
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_PATH}")

    print(
        f"Loading {N_CLAIMS} claims, {N_REVIEWS} reviews, "
        f"{N_QUESTIONS} questions, {N_DIALOGUE_DECL} declaratives, "
        f"{N_DIALOGUE_IMP} imperatives from dialogue..."
    )
    claim_texts = load_claims(CLAIMS_PATH, max_samples=N_CLAIMS)
    review_texts = load_reviews(REVIEWS_PATH, max_samples=N_REVIEWS)
    question_texts = load_questions(QUESTIONS_PATH, max_samples=N_QUESTIONS)
    dialogue_decl_texts, dialogue_imp_texts = load_spaadia_sentences(
        SPAADIA_DIR,
        max_decl=N_DIALOGUE_DECL,
        max_imp=N_DIALOGUE_IMP,
    )
    print(
        f"  Claims: {len(claim_texts)}, "
        f"Reviews: {len(review_texts)}, "
        f"Questions: {len(question_texts)}, "
        f"Declaratives: {len(dialogue_decl_texts)}, "
        f"Imperatives: {len(dialogue_imp_texts)}"
    )

    # claim=0, not a claim=1 (reviews + questions + dialogue sentences)
    X_raw = (
        claim_texts
        + review_texts
        + question_texts
        + dialogue_decl_texts
        + dialogue_imp_texts
    )
    y_labels = (
        ["claim"] * len(claim_texts)
        + ["not a claim"] * len(review_texts)
        + ["not a claim"] * len(question_texts)
        + ["not a claim"] * len(dialogue_decl_texts)
        + ["not a claim"] * len(dialogue_imp_texts)
    )
    class_names = np.array(["claim", "not a claim"])
    # Binary labels for logistic regression:
    #   0 = claim, 1 = not a claim (reviews + questions + dialogue)
    y = np.array(
        [0] * len(claim_texts)
        + [1] * len(review_texts)
        + [1] * len(question_texts)
        + [1] * len(dialogue_decl_texts)
        + [1] * len(dialogue_imp_texts),
        dtype=np.int32,
    )

    print("Encoding with MiniLM ONNX model...")
    onnx_path = MODEL_DIR / "model.onnx"
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not tokenizer_path.exists() or Tokenizer is None:
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path} (and pip install tokenizers)"
        )
    if ENCODINGS_PATH.exists():
        # Avoid accidentally reusing a previously generated (possibly incompatible) NPZ.
        ENCODINGS_PATH.unlink()
    X = encode_with_onnx(X_raw, MODEL_DIR)

    print(f"Saving encodings to {ENCODINGS_PATH}...")
    np.savez_compressed(
        ENCODINGS_PATH,
        X=X,
        y=y,
        class_names=class_names,
    )
    print(f"Done. Shape: X {X.shape}, y {y.shape}. Run lr_train.py to train.")


if __name__ == "__main__":
    main()
