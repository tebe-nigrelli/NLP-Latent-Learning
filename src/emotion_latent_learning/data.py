from __future__ import annotations

from .utils.common import *

# Focused module: datasets and collators.


class EmotionTextDataset(Dataset):
    """Base class for supervised emotion datasets used by the runtime."""

    @property
    def labels(self) -> torch.Tensor:
        raise NotImplementedError


class GoEmotionsExampleDataset(EmotionTextDataset):
    """Thin wrapper around the Hugging Face GoEmotions split."""

    def __init__(self, split: Any, num_labels: int) -> None:
        self.split = split
        self.num_labels = int(num_labels)
        cached: List[torch.Tensor] = []
        for row in split:
            label_vec = torch.zeros(self.num_labels, dtype=torch.float32)
            for idx in row.get("labels", []) or []:
                if 0 <= int(idx) < self.num_labels:
                    label_vec[int(idx)] = 1.0
            cached.append(label_vec)
        self._labels = torch.stack(cached, dim=0) if cached else torch.zeros((0, self.num_labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.split[int(idx)]
        return {"text": str(row.get("text", "")), "labels": self._labels[int(idx)]}

    @property
    def labels(self) -> torch.Tensor:
        return self._labels


class SemEvalEIRegDataset(EmotionTextDataset):
    """Dataset wrapper for merged SemEval EI-reg rows with continuous labels."""

    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = list(rows)
        if self.rows:
            labels = [torch.tensor(row["labels"], dtype=torch.float32) for row in self.rows]
            self._labels = torch.stack(labels, dim=0)
        else:
            self._labels = torch.zeros((0, 0), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[int(idx)]
        return {"text": str(row["text"]), "labels": self._labels[int(idx)]}

    @property
    def labels(self) -> torch.Tensor:
        return self._labels


class EmotionTextCollator:
    """Tokenize supervised text batches and stack labels."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [str(ex["text"]) for ex in examples]
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.stack([
            ex["labels"] if isinstance(ex["labels"], torch.Tensor) else torch.tensor(ex["labels"], dtype=torch.float32)
            for ex in examples
        ], dim=0).float()
        encoded["labels"] = labels
        encoded["texts"] = texts
        return encoded


class TextOnlyDataset(Dataset):
    """Map-style text-only dataset wrapper for denoising VAE ablations."""

    def __init__(self, split: Any, text_column: str = "text", max_samples: Optional[int] = None) -> None:
        self.split = split
        self.text_column = text_column
        n = len(split)
        if max_samples is not None:
            n = min(n, int(max_samples))
        self.indices = list(range(n))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.split[self.indices[int(idx)]]
        text = row.get(self.text_column, "") if isinstance(row, dict) else row[self.text_column]
        return {"text": str(text)}


class TextOnlyIterableDataset(torch.utils.data.IterableDataset):
    """Streaming text-only dataset wrapper for large HF datasets."""

    def __init__(self, iterable: Any, text_column: str = "text", max_samples: Optional[int] = None) -> None:
        super().__init__()
        self.iterable = iterable
        self.text_column = text_column
        self.max_samples = None if max_samples is None else int(max_samples)

    def __iter__(self):
        count = 0
        for row in self.iterable:
            if self.max_samples is not None and count >= self.max_samples:
                break
            if not isinstance(row, dict):
                continue
            text = str(row.get(self.text_column, "")).strip()
            if not text:
                continue
            count += 1
            yield {"text": text}


def _split_words(text: str) -> List[str]:
    return str(text).strip().split()


class TextDenoisingCollator:
    """Build corrupted-input/original-target batches for text-only VAE denoising.

    The input to the encoder is corrupted text. The target text is the original.
    `target_input_ids` encodes the original text so the text-only memory loss can
    match the decoder memory to the original encoder representation, while the
    copy loss teacher-forces the original text through T5's decoder.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        mask_fraction: float = 0.15,
        mean_span_length: float = 3.0,
        corruption: str = "span_mask",
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.mask_fraction = float(mask_fraction)
        self.mean_span_length = max(1.0, float(mean_span_length))
        self.corruption = str(corruption)
        self.rng = random.Random(int(seed))

    def _mask_token(self, span_idx: int) -> str:
        # T5 tokenizers define sentinel tokens; fall back to a generic marker for
        # other tokenizers so the ablation remains model-agnostic.
        token = f"<extra_id_{span_idx % 100}>"
        vocab = getattr(self.tokenizer, "get_vocab", lambda: {})()
        return token if token in vocab else "<mask>"

    def _corrupt_delete(self, words: List[str]) -> List[str]:
        if len(words) <= 1:
            return words
        keep = [w for w in words if self.rng.random() > self.mask_fraction]
        return keep if keep else [self.rng.choice(words)]

    def _corrupt_token_mask(self, words: List[str]) -> List[str]:
        if not words:
            return words
        return [self._mask_token(0) if self.rng.random() < self.mask_fraction else w for w in words]

    def _corrupt_span_mask(self, words: List[str]) -> List[str]:
        if len(words) <= 1 or self.mask_fraction <= 0.0:
            return words
        n = len(words)
        target_masked = max(1, int(round(n * min(max(self.mask_fraction, 0.0), 0.95))))
        out: List[str] = []
        i = 0
        masked = 0
        span_idx = 0
        while i < n:
            remaining = n - i
            remaining_to_mask = max(0, target_masked - masked)
            if remaining_to_mask <= 0:
                out.extend(words[i:])
                break
            # Slightly increase probability as we approach the end so very short
            # examples still receive at least one corruption.
            p = min(0.8, max(self.mask_fraction, remaining_to_mask / max(remaining, 1)))
            if self.rng.random() < p:
                span_len = max(1, int(round(self.rng.expovariate(1.0 / self.mean_span_length))))
                span_len = min(span_len, remaining, remaining_to_mask)
                out.append(self._mask_token(span_idx))
                span_idx += 1
                masked += span_len
                i += span_len
            else:
                out.append(words[i])
                i += 1
        return out if out else [self._mask_token(0)]

    def corrupt_text(self, text: str) -> str:
        words = _split_words(text)
        if not words:
            return str(text)
        mode = self.corruption.lower().strip()
        if mode in {"delete", "deletion", "token_delete"}:
            corrupted = self._corrupt_delete(words)
        elif mode in {"token_mask", "mask"}:
            corrupted = self._corrupt_token_mask(words)
        elif mode in {"none", "identity"}:
            corrupted = words
        else:
            corrupted = self._corrupt_span_mask(words)
        return " ".join(corrupted)

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [str(ex.get("text", "")).strip() for ex in examples]
        texts = [t if t else "." for t in texts]
        corrupted_texts = [self.corrupt_text(t) for t in texts]
        corrupted = self.tokenizer(
            corrupted_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": corrupted["input_ids"],
            "attention_mask": corrupted["attention_mask"],
            "target_input_ids": target["input_ids"],
            "target_attention_mask": target["attention_mask"],
            "texts": texts,
            "corrupted_texts": corrupted_texts,
        }


def labels_are_continuous(labels: torch.Tensor) -> bool:
    if labels.numel() == 0:
        return False
    labels = labels.detach().float()
    return bool(((labels > 0.0) & (labels < 1.0)).any().item())


def compute_pos_weight(labels: torch.Tensor, max_ratio: float = 20.0) -> torch.Tensor:
    labels = labels.detach().float()
    if labels.numel() == 0:
        return torch.ones((1,), dtype=torch.float32)
    positives = labels.sum(dim=0)
    negatives = labels.size(0) - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    return pos_weight.clamp(min=1.0, max=float(max_ratio)).float()


@dataclass
class SemEvalEIRegSplits:
    train: List[Dict[str, Any]]
    validation: List[Dict[str, Any]]
    test: List[Dict[str, Any]]
    emotion_names: Sequence[str]


def _read_semeval_file(path: Path, emotion: str) -> Dict[str, float]:
    rows: Dict[str, float] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        text_idx = 1 if len(header) > 1 else 0
        score_idx = len(header) - 1
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(text_idx, score_idx):
                continue
            text = parts[text_idx]
            try:
                score = float(parts[score_idx])
            except ValueError:
                continue
            rows[text] = score
    return rows


def _merge_semeval_rows(per_emotion: Dict[str, Dict[str, float]], emotions: Sequence[str]) -> List[Dict[str, Any]]:
    texts = sorted({text for mapping in per_emotion.values() for text in mapping.keys()})
    merged: List[Dict[str, Any]] = []
    for text in texts:
        labels = [float(per_emotion.get(emotion, {}).get(text, 0.0)) for emotion in emotions]
        merged.append({"text": text, "labels": labels})
    return merged


def load_semval2018_ei_reg(
    data_dir: Path,
    language: str = "En",
    emotions: Sequence[str] = ("anger", "fear", "joy", "sadness"),
    merge_by_text: bool = True,
) -> SemEvalEIRegSplits:
    del merge_by_text  # Files are always merged by text in this runtime wrapper.
    data_dir = Path(data_dir)
    split_suffixes = {
        "train": "train",
        "validation": "dev",
        "test": "test-gold",
    }
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, suffix in split_suffixes.items():
        per_emotion: Dict[str, Dict[str, float]] = {}
        for emotion in emotions:
            pattern = f"2018-EI-reg-{language}-{emotion}-{suffix}.txt"
            per_emotion[emotion] = _read_semeval_file(data_dir / pattern, emotion)
        split_rows[split_name] = _merge_semeval_rows(per_emotion, emotions)
    return SemEvalEIRegSplits(
        train=split_rows["train"],
        validation=split_rows["validation"],
        test=split_rows["test"],
        emotion_names=list(emotions),
    )


def semval_download_instructions(
    data_dir: Path,
    language: str = "En",
    emotions: Sequence[str] = ("anger", "fear", "joy", "sadness"),
) -> str:
    emotion_list = ", ".join(emotions)
    return (
        f"Place the SemEval-2018 EI-reg {language} files for {emotion_list} under {Path(data_dir)}. "
        "Expected names look like 2018-EI-reg-En-anger-train.txt, "
        "2018-EI-reg-En-anger-dev.txt, and 2018-EI-reg-En-anger-test-gold.txt."
    )


def download_semval2018_ei_reg(
    data_dir: Path,
    url: Optional[str] = None,
    language: str = "En",
    emotions: Sequence[str] = ("anger", "fear", "joy", "sadness"),
) -> None:
    # The official archive has been distributed in several layouts. To avoid a
    # brittle downloader, fail with an actionable message unless the user supplies
    # local files. This keeps the runtime deterministic on clusters without net.
    del url
    data_dir = Path(data_dir)
    required = [data_dir / f"2018-EI-reg-{language}-{emotion}-train.txt" for emotion in emotions]
    if all(path.exists() for path in required):
        return
    raise FileNotFoundError(semval_download_instructions(data_dir=data_dir, language=language, emotions=emotions))


def build_text_denoising_dataset(data_config: Any, supervised_train_split: Any = None) -> Optional[Dataset]:
    """Create the optional text-only dataset for the denoising ablation.

    If `text_denoising_dataset_repo` is unset, we reuse the supervised train
    split's text column as a cheap smoke-test source. For a real ablation, point
    it at a large unlabeled HF dataset and set the text column if needed.
    """

    if not getattr(data_config, "text_denoising_enabled", False):
        return None

    text_column = getattr(data_config, "text_denoising_text_column", "text")
    max_samples = getattr(data_config, "text_denoising_max_samples", None)
    repo = getattr(data_config, "text_denoising_dataset_repo", None)
    config = getattr(data_config, "text_denoising_dataset_config", None)
    split = getattr(data_config, "text_denoising_split", "train")
    streaming = bool(getattr(data_config, "text_denoising_streaming", False))

    if repo:
        kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
        if config:
            dataset = load_dataset(repo, config, **kwargs)
        else:
            dataset = load_dataset(repo, **kwargs)
        if streaming:
            return TextOnlyIterableDataset(dataset, text_column=text_column, max_samples=max_samples)
        return TextOnlyDataset(dataset, text_column=text_column, max_samples=max_samples)

    if supervised_train_split is None:
        raise ValueError(
            "text_denoising_enabled=True requires either text_denoising_dataset_repo "
            "or a supervised training split to reuse."
        )
    return TextOnlyDataset(supervised_train_split, text_column=text_column, max_samples=max_samples)
