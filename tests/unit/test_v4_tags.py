"""v4 §3: speech act, status vocabulary, evidence span, confidence, entities."""
from engine.read.v4_tags import MODEL_VERSION, extract_entities, read_tags


def test_kind_is_the_speech_act_not_the_reader_kind():
    # v3 calls a meeting question kind=info; §3 has an explicit `question`.
    assert read_tags("Est-ce que le budget est validé ?").kind in (
        "question", "decision")
    assert read_tags("Est-ce que le budget est validé ?").kind == "question"
    assert read_tags("On valide le budget à 800 euros.").kind in (
        "decision", "fact")


def test_status_uses_the_v4_vocabulary():
    assert read_tags("Le budget est proposé à 1000 euros.").status == "proposed"
    assert read_tags("On valide le budget à 800 euros.").status == "confirmed"
    assert read_tags("Finalement on annule le budget.").status == "cancelled"
    assert read_tags("C'est fait, le devis est signé.").status == "completed"
    assert read_tags("euh donc voilà quoi").status == "unknown"


def test_every_tag_carries_an_evidence_span_inside_the_text():
    text = "Le budget est proposé à 1000 euros. On valide 800 euros."
    tags = read_tags(text)
    start, end, surface = tags.evidence
    assert 0 <= start < end <= len(text)
    assert text[start:end] == surface


def test_confidence_is_zero_when_nothing_was_read():
    assert read_tags("euh alors voilà").confidence == 0.0
    assert read_tags("On valide le budget à 800 euros.").confidence > 0.0
    assert read_tags("Le budget est proposé.").model_version == MODEL_VERSION


def test_entities_cover_the_seven_kinds_and_word_numbers():
    ents = extract_entities("Le projet Atlas prévoit cent euros en juin pour Marie")
    kinds = {e["type"] for e in ents}
    assert "amount" in kinds                     # ASR spells numbers as words
    assert "date" in kinds and "project" in kinds and "person" in kinds
    amount = next(e for e in ents if e["type"] == "amount")
    assert "cent" in amount["value"]


def test_slot_and_value_are_grounded():
    t = read_tags("On valide le budget à 800 euros.")
    assert t.slot == "budget" and "800" in t.value
    d = read_tags("La livraison est fixée au 12 juin.")
    assert d.slot == "date" and "juin" in d.value
    o = read_tags("Marie s'occupe du budget.")
    assert o.slot == "owner" and o.owner


def test_intent_is_read_not_hardcoded():
    assert read_tags("Compare le budget de mai et de juin.").intent == "compare"
    assert read_tags("Prépare une synthèse du dossier.").intent == "dossier"
    assert read_tags("Rédige un mail au prestataire.").intent == "email"
