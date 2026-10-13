import hashlib
import json
import os
import random
import sys

import make_corpus

START = 1695513600
WEEK = 7 * 86400


def build(out_dir, rows_per_week=12, train_weeks=52, test_weeks=12, challenge=15, seed=3):
    import thrember

    extractor = thrember.PEFeatureExtractor()
    rng = random.Random(seed)
    corpus = os.path.join(out_dir, "_corpus")
    make_corpus.main(corpus)
    pes = []
    for name in sorted(os.listdir(corpus)):
        with open(os.path.join(corpus, name), "rb") as handle:
            data = handle.read()
        if data[:2] == b"MZ" and 512 < len(data) < 1_000_000:
            pes.append(data)

    def record(week, label, family, file_type):
        base = pes[rng.randrange(len(pes))]
        if label == 1:
            payload = bytes(rng.getrandbits(8) for _ in range(rng.randrange(2048, 8192)))
        else:
            payload = (b"benign-resource " * rng.randrange(64, 256))
        data = base + payload
        raw = extractor.raw_features(data)
        raw["sha256"] = hashlib.sha256(data).hexdigest()
        raw["first_submission_date"] = START + week * WEEK + rng.randrange(WEEK)
        raw["label"] = label
        raw["family"] = family
        raw["file_type"] = file_type
        return json.dumps(raw)

    def write(name, lines):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")

    def week_rows(week):
        lines = []
        for j in range(rows_per_week):
            malicious = j % 4 == 0
            family = f"fam{week // 8:02d}" if malicious else None
            lines.append(record(week, 1 if malicious else 0, family, "Dot_Net"))
        lines.append(record(week, 0, None, "Win32"))
        lines.append(record(week, None, None, "Dot_Net"))
        return lines

    train_a, train_b, test = [], [], []
    for week in range(train_weeks):
        (train_a if week < train_weeks // 2 else train_b).extend(week_rows(week))
    for week in range(train_weeks, train_weeks + test_weeks):
        test.extend(week_rows(week))
    write("2023-09-24_Dot_Net_train.jsonl", train_a)
    write("2024-03-24_Dot_Net_train.jsonl", train_b)
    write("2024-09-22_Dot_Net_test.jsonl", test)
    write("challenge_malicious.jsonl", [record(train_weeks + rng.randrange(test_weeks), 1, "evasive", "Dot_Net") for _ in range(challenge)] + [record(55, 1, "other", "Win64")])

    for name in os.listdir(corpus):
        os.remove(os.path.join(corpus, name))
    os.rmdir(corpus)


if __name__ == "__main__":
    build(sys.argv[1])
