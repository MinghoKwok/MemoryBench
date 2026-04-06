#!/usr/bin/env python3
"""
Build ComicScene v3 dev set.

Design goals:
- keep image pages as the primary evidence source
- enrich raw session text enough for lexical / dense / multimodal retrieval
- add text-only distractor sessions that reuse near-match entities and actions
- avoid external sidecar notes in the main benchmark path
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIALOG = REPO_ROOT / "Benchmark_Pipeline" / "data" / "dialog" / "ComicScene_Alley_Oop_Draft.json"
OUTPUT_DIALOG = REPO_ROOT / "Benchmark_Pipeline" / "data" / "dialog" / "ComicScene_Alley_Oop_V3_Dev.json"
OUTPUT_NOTES = REPO_ROOT / "Benchmark_Pipeline" / "data" / "dialog" / "ComicScene_Alley_Oop_V3_Dev_notes.json"

SOURCE_SESSION_IDS: List[str] = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D25",
    "D26",
    "D27",
    "D28",
    "D29",
    "D30",
    "D31",
    "D32",
    "D33",
    "D34",
    "D35",
]

SESSION_PLAN: List[str] = [
    "D1",
    "D2",
    "D3",
    "D4",
    "C1",
    "C2",
    "C3",
    "C4",
    "D25",
    "D26",
    "D27",
    "D28",
    "C5",
    "C6",
    "D29",
    "C7",
    "D30",
    "D31",
    "C8",
    "D32",
    "D33",
    "D34",
    "D35",
]

FOLLOW_UP_DIALOGUES: Dict[str, List[Tuple[str, str]]] = {
    "D1": [
        (
            "The rescue page is easy to confuse with the later crown pages. What is the opening image again?",
            "It opens with a woman pointing toward a drowning crowned figure, so the page starts as a water-rescue scene rather than a palace argument.",
        ),
        (
            "What makes that page stand apart from the later palace-crown material?",
            "The soaked ruler later points accusingly back at Alley, and the whole page stays centered on the water emergency instead of throne or chore business.",
        ),
    ],
    "D2": [
        (
            "That Dinny page opens mid-ride instead of with the crash itself, right?",
            "Yes. Alley is already riding Dinny in the first panel, and only later does Dinny slam into a tree.",
        ),
        (
            "So the joke there is a tree collision, not a water splash or bargain scene?",
            "Exactly. The page is about Dinny crashing into a tree and being sore afterward, not about splashing through water or shopping for another dinosaur.",
        ),
    ],
    "D3": [
        (
            "The dinosaur lot had several sales labels. Which comparison matters on that page?",
            "The cheap, bargain, and today's special labels all appear there, but the bargain animal is not larger than the cheap one.",
        ),
        (
            "What is the punch line of that dealership page after all those sale animals are lined up?",
            "Instead of ending up on one of the larger sale beasts, Alley test-drives the little baby dinosaur.",
        ),
    ],
    "D4": [
        (
            "The small-dinosaur test-drive page is different from the dealership lineup, right?",
            "Yes. This page is about motion: Alley rides the little dinosaur, then gets kicked off during the test drive.",
        ),
        (
            "And what image closes that page after the kick?",
            "The final close-up is the little dinosaur drooling, which is different from the earlier sales-lot labels.",
        ),
    ],
    "D25": [
        (
            "This bridge page is still before the palace takeover sequence, correct?",
            "Right. It belongs to the same comic run, but it is still before the throne offer, crowd pelting, chores, dinner, and crown-return pages.",
        ),
    ],
    "D26": [
        (
            "So this transition page still comes before Oop becomes the temporary ruler?",
            "Yes. It is another lead-in page before the crowd, cave labor, basin-running, clams dispute, and crown-return scenes.",
        ),
    ],
    "D27": [
        (
            "Where does the later palace argument really start?",
            "It starts on the throne-offer page where the crowned ruler tells Oop he ought to take the throne for a while.",
        ),
        (
            "And how does that throne-offer page end?",
            "It does not end with calm acceptance. By the end of the page Oop strikes the crowned figure.",
        ),
    ],
    "D28": [
        (
            "The crowd scene and the cave-work scene are actually the same page, right?",
            "Yes. The temporary ruler is pelted by the crowd outside, and later on that same page he is shown working inside a cave.",
        ),
        (
            "So that page is punishment and labor, not a lounging interlude?",
            "Exactly. It moves from public punishment to cave labor rather than staying on a relaxed palace tableau.",
        ),
    ],
    "D29": [
        (
            "The chores page begins more quietly than the crowd page, doesn't it?",
            "It starts with a reclining ruler beside round treats that are specifically said not to be bon-bons.",
        ),
        (
            "And what visual detail matters once the former ruler starts running?",
            "Later on the same page he runs with a basin on his head before rushing outside, so the page shifts from lounging to frantic movement.",
        ),
    ],
    "D30": [
        (
            "What is the complaint page after the chores material about?",
            "It is the clams-dispute page, with two accused subjects facing the seated ruler and raising their hands while the complaint is heard.",
        ),
        (
            "Does that page turn into cave labor or stay with the hearing?",
            "It stays focused on the seated hearing over the stolen clams rather than shifting into a work scene.",
        ),
    ],
    "D31": [
        (
            "The ulcer page changes setting within the same strip, correct?",
            "Yes. It begins with a fight outside and then shifts to a dinner table scene.",
        ),
        (
            "What happens to the temporary ruler once the dinner scene starts?",
            "He complains about food and ulcers and later gets hit at the dinner table.",
        ),
    ],
    "D32": [
        (
            "Which later palace page has the peek-a-boo trick?",
            "That is the page where Oop sneaks up behind the necklace-wearing man with a peek-a-boo.",
        ),
        (
            "And what follows the peek-a-boo on that same page?",
            "The peek-a-boo comes before Oop hits the necklace-wearing man during the meal sequence.",
        ),
    ],
    "D33": [
        (
            "How does the crown-return page progress?",
            "It includes an apology and the crown being handed back before Oop is shown again as an ordinary citizen.",
        ),
        (
            "What later detail on that same page makes it easy to distinguish from the earlier apology beat?",
            "After the crown-return sequence, Oop is later shown carrying a wooden board.",
        ),
    ],
    "D34": [
        (
            "This is a later palace aftermath page, but not the crown-return page with the wooden board, right?",
            "Right. It belongs to the same palace cast after the takeover arc, but it is not the apology-and-board page and not the chores page with the basin.",
        ),
    ],
    "D35": [
        (
            "And this other later palace page is also after the main takeover sequence?",
            "Yes. It is another later palace aftermath page, but not the clams hearing, not the ulcer dinner page, and not the board-carrying apology page.",
        ),
    ],
}

SYNTHETIC_SESSIONS: Dict[str, Dict[str, Any]] = {
    "C1": {
        "date": "1933-01-05",
        "dialogues": [
            (
                "Those early crown scenes are easy to muddle together. I vaguely remember a headpiece near a riverbank too.",
                "Yes, one unrelated page has a helmeted figure near a riverbank, which is different from the drowning crowned figure on the true water-rescue page.",
            ),
            (
                "So not every page with headgear and water is the rescue strip.",
                "Exactly. Some later or unrelated images echo the same motifs without having the woman pointing at a drowning ruler.",
            ),
        ],
    },
    "C2": {
        "date": "1933-01-06",
        "dialogues": [
            (
                "The dinosaur material is also hard to track because several animals are compared by size.",
                "Right. On one confusing page the premium display model seems largest and the trade-in beast seems smaller, which is not the same setup as the cheap-versus-bargain dealership page.",
            ),
            (
                "So a bigger-looking sales beast elsewhere does not mean Alley rode it.",
                "Exactly. Size language repeats across the comic even when the actual mount choice is different.",
            ),
        ],
    },
    "C3": {
        "date": "1933-01-07",
        "dialogues": [
            (
                "The small-dinosaur scenes blur together for me.",
                "That is the trap. One creature is panting or snorting in a different scene, while another page ends with an actual drooling close-up after a kick.",
            ),
            (
                "So just remembering an animal face at the end of a page is not enough.",
                "Right. The comic reuses close-ups, but the actions before them are different.",
            ),
        ],
    },
    "C4": {
        "date": "1933-01-08",
        "dialogues": [
            (
                "The early action pages and the later palace pages both have people pointing and arguing.",
                "Yes, which is why a model can mix the soaked ruler's accusation with later throne and complaint scenes if it only remembers the broad gesture of pointing.",
            ),
            (
                "So gestures alone are not enough to identify the right strip.",
                "Exactly. The event context around the gesture is what matters.",
            ),
        ],
    },
    "C5": {
        "date": "1933-01-30",
        "dialogues": [
            (
                "The palace pages keep flipping between power and embarrassment.",
                "Yes. One distractor scene has a figure working first and resting later, which is the reverse of the chores page where relaxation comes before the frantic running.",
            ),
            (
                "That reversal makes the later palace pages easy to confuse.",
                "Exactly. Several pages share the same cast and setting but reverse the order of work, rest, and punishment.",
            ),
        ],
    },
    "C6": {
        "date": "1933-01-31",
        "dialogues": [
            (
                "I also mix up the carrying gags.",
                "That is understandable. One page suggests a bucket carried on the shoulder around the middle panels, which is different from the chores page where the basin is balanced on the runner's head.",
            ),
            (
                "So container jokes repeat with slightly different poses.",
                "Yes, and those near-matches are exactly what make blind full-context reading unreliable.",
            ),
        ],
    },
    "C7": {
        "date": "1933-02-03",
        "dialogues": [
            (
                "The later complaint pages also blend together.",
                "Right. Not every palace hearing is the stolen-clams page; another complaint scene can involve tribute, taxes, or table talk without the same two accused figures raising their hands.",
            ),
            (
                "So a seated ruler hearing a complaint is not automatically the clams dispute.",
                "Exactly. The specifics of who is present and what is being argued over matter a lot.",
            ),
        ],
    },
    "C8": {
        "date": "1933-02-05",
        "dialogues": [
            (
                "By the end of the palace arc I can barely separate the apology material from the aftermath pages.",
                "That is the intended confusion. Some aftermath pages keep the same cast but do not include the crown hand-back or the wooden board that make the true return page distinctive.",
            ),
            (
                "So the final palace pages are deliberately close neighbors rather than cleanly separated scenes.",
                "Yes. The overlap in cast and setting makes retrieval precision more important than a broad full-context gist.",
            ),
        ],
    },
}

NOTE_TEXTS: Dict[str, str] = {
    session_id: " ".join(answer for _, answer in dialogues)
    for session_id, dialogues in FOLLOW_UP_DIALOGUES.items()
}

QA_ITEMS: List[Dict[str, Any]] = [
    {
        "point": [["X3"], ["Y1"]],
        "question": "In the opening panel of the water-rescue page, who is pointing toward the drowning figure: woman, crowned man, or shirtless man? Reply with only one of: woman, crowned man, shirtless man.",
        "answer": "woman",
        "session_id": ["D1"],
        "clue": ["D1:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the water-rescue page, is the crowned figure still in the water in the final panel? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D1"],
        "clue": ["D1:1"],
    },
    {
        "point": [["X3"], ["Y2"]],
        "question": "On the water-rescue page, after Alley pulls the crowned figure from the water, does the still-wet ruler later point accusingly at Alley? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D1"],
        "clue": ["D1:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the tree-crash page, is Alley already riding Dinny in the first panel? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D2"],
        "clue": ["D2:1"],
    },
    {
        "point": [["X4"], ["Y1"]],
        "question": "On the tree-crash page, is Dinny shown splashing through water after the crash? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D2"],
        "clue": ["D2:1"],
    },
    {
        "point": [["X3"], ["Y2"]],
        "question": "What does Dinny crash into on that page: a tree, a cave wall, or a wagon? Reply with only one of: a tree, a cave wall, a wagon.",
        "answer": "a tree",
        "session_id": ["D2"],
        "clue": ["D2:1"],
    },
    {
        "point": [["X4"], ["Y2"]],
        "question": "On the dealership page, is the dinosaur labeled bargain larger than the one labeled cheap? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D3"],
        "clue": ["D3:1"],
    },
    {
        "point": [["X3"], ["Y1"]],
        "question": "Which dinosaur does Alley test-drive on the dealership page: cheap, today's special, or the little baby? Reply with only one of: cheap, today's special, little baby.",
        "answer": "little baby",
        "session_id": ["D3"],
        "clue": ["D3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "On the dealership page, does Alley end up test-driving one of the larger sale animals instead of the little baby? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D3"],
        "clue": ["D3:1"],
    },
    {
        "point": [["X3"], ["Y2"]],
        "question": "On the test-drive page, does the small dinosaur kick Alley off before the final drooling close-up? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D4"],
        "clue": ["D4:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the test-drive page, is the final close-up a drooling view of the little dinosaur? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D4"],
        "clue": ["D4:1"],
    },
    {
        "point": [["X3"], ["Y1"]],
        "question": "Who suggests that Oop ought to take the ruler's place on the throne-offer page: crowned man, Oop, or guard? Reply with only one of: crowned man, Oop, guard.",
        "answer": "crowned man",
        "session_id": ["D27"],
        "clue": ["D27:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the throne-offer page, is the crowned ruler the one who gets hit by the end of the page? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D27"],
        "clue": ["D27:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "On the throne-offer page, does Oop calmly accept the throne without striking anyone? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D27"],
        "clue": ["D27:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the crowd-and-cave page, after the temporary ruler is pelted outside, is he later shown working inside a cave on that same page? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D28"],
        "clue": ["D28:1"],
    },
    {
        "point": [["X3"], ["Y2"]],
        "question": "Later on that same crowd-and-cave page, where is the temporary ruler shown working: in a cave, by the water, or inside a wagon? Reply with only one of: in a cave, by the water, inside a wagon.",
        "answer": "in a cave",
        "session_id": ["D28"],
        "clue": ["D28:1"],
    },
    {
        "point": [["X2"], ["Y1"]],
        "question": "Is the crowd-and-cave page just a lounging page with no later work scene? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D28"],
        "clue": ["D28:1"],
    },
    {
        "point": [["X4"], ["Y1"]],
        "question": "On the chores page, are the round treats beside the reclining ruler actually bon-bons? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D29"],
        "clue": ["D29:1"],
    },
    {
        "point": [["X3"], ["Y2"]],
        "question": "When the former temporary ruler runs on the chores page, where is the basin: on his head, in his hand, or on the ground? Reply with only one of: on his head, in his hand, on the ground.",
        "answer": "on his head",
        "session_id": ["D29"],
        "clue": ["D29:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "On the chores page, is the same figure shown relaxing indoors first and then later running outside? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D29"],
        "clue": ["D29:1"],
    },
    {
        "point": [["X3"], ["Y1"]],
        "question": "On the clams-dispute page, are the two accused subjects shown with their hands raised while facing the seated ruler? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D30"],
        "clue": ["D30:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the clams-dispute page, does the ruler stay seated while hearing about the stolen clams before he talks with the grand wiz? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D30"],
        "clue": ["D30:1"],
    },
    {
        "point": [["X2"], ["Y1"]],
        "question": "Is the clams-dispute page mainly about cave labor rather than a complaint over stolen clams? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D30"],
        "clue": ["D30:1"],
    },
    {
        "point": [["X2", "X3"]],
        "question": "On the ulcer dinner page, after the fight outside, does the temporary ruler later get hit at the dinner table? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D31"],
        "clue": ["D31:1"],
    },
    {
        "point": [["X3"], ["Y1"]],
        "question": "On the ulcer dinner page, does the page shift from an outdoor fight to a meal where the temporary ruler complains about food and ulcers? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D31"],
        "clue": ["D31:1"],
    },
    {
        "point": [["X3"], ["Y1"]],
        "question": "On the peek-a-boo page, does Oop hit the necklace-wearing man after sneaking up behind him? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D32"],
        "clue": ["D32:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the peek-a-boo page, does the peek-a-boo happen before Oop hits the necklace-wearing man? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D32"],
        "clue": ["D32:1"],
    },
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the apology-and-crown-return page, is the crown handed back before Oop is shown as an ordinary citizen again? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D33"],
        "clue": ["D33:1"],
    },
    {
        "point": [["X2"], ["Y1"]],
        "question": "On the apology-and-crown-return page, is Oop shown as an ordinary citizen again before the crown is handed back? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D33"],
        "clue": ["D33:1"],
    },
    {
        "point": [["X3"], ["Y1"]],
        "question": "On the apology-and-crown-return page, is Oop later shown carrying a wooden board after the crown-return sequence? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D33"],
        "clue": ["D33:1"],
    },
]

def load_source() -> Dict[str, Any]:
    with SOURCE_DIALOG.open('r', encoding='utf-8') as f:
        return json.load(f)


def _append_follow_ups(session: Dict[str, Any]) -> Dict[str, Any]:
    session_id = session['session_id']
    follow_ups = FOLLOW_UP_DIALOGUES.get(session_id, [])
    if not follow_ups:
        return session

    dialogues = list(session.get('dialogues', []))
    next_idx = len(dialogues) + 1
    for user_text, assistant_text in follow_ups:
        dialogues.append({
            'round': f'{session_id}:{next_idx}',
            'user': user_text,
            'assistant': assistant_text,
        })
        next_idx += 1
    session['dialogues'] = dialogues
    return session


def _build_synthetic_session(session_id: str) -> Dict[str, Any]:
    spec = SYNTHETIC_SESSIONS[session_id]
    dialogues = []
    for idx, (user_text, assistant_text) in enumerate(spec['dialogues'], start=1):
        dialogues.append({
            'round': f'{session_id}:{idx}',
            'user': user_text,
            'assistant': assistant_text,
        })
    return {
        'session_id': session_id,
        'date': spec['date'],
        'dialogues': dialogues,
    }


def build_dataset() -> Dict[str, Any]:
    source = load_source()
    source_sessions = {
        sess['session_id']: copy.deepcopy(sess)
        for sess in source.get('multi_session_dialogues', [])
        if sess.get('session_id') in SOURCE_SESSION_IDS
    }

    ordered_sessions: List[Dict[str, Any]] = []
    for session_id in SESSION_PLAN:
        if session_id in source_sessions:
            ordered_sessions.append(_append_follow_ups(source_sessions[session_id]))
        else:
            ordered_sessions.append(_build_synthetic_session(session_id))

    return {
        'character_profile': copy.deepcopy(source.get('character_profile', {})),
        'multi_session_dialogues': ordered_sessions,
        'human-annotated QAs': copy.deepcopy(QA_ITEMS),
    }


def build_notes(data: Dict[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for session in data['multi_session_dialogues']:
        session_id = session['session_id']
        note_text = NOTE_TEXTS.get(session_id, '').strip()
        if not note_text:
            continue
        round_id = session.get('dialogues', [{}])[0].get('round', '')
        if not round_id:
            continue
        entries.append({
            'session_id': session_id,
            'round_id': round_id,
            'text': note_text,
        })
    return {
        'dataset': 'comicscene_v3_dev',
        'source_dialog_json': str(OUTPUT_DIALOG),
        'notes': entries,
    }


def main() -> None:
    data = build_dataset()
    notes = build_notes(data)

    OUTPUT_DIALOG.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DIALOG.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with OUTPUT_NOTES.open('w', encoding='utf-8') as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

    total_sessions = len(data['multi_session_dialogues'])
    total_rounds = sum(len(sess['dialogues']) for sess in data['multi_session_dialogues'])
    total_qas = len(data['human-annotated QAs'])
    total_notes = len(notes['notes'])
    print(f'[INFO] Wrote {OUTPUT_DIALOG}')
    print(f'[INFO] Wrote {OUTPUT_NOTES}')
    print(f'[INFO] sessions={total_sessions} rounds={total_rounds} qas={total_qas} notes={total_notes}')


if __name__ == '__main__':
    main()
