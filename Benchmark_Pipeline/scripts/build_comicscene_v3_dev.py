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
    # Chaotic interleaving across 4 comic series
    "D1", "D2",                  # AO start
    "TC1", "TC2",                # TC intro
    "D3", "WL1",                 # AO + WL
    "CH1", "TC3", "WL2",         # CH + TC + WL
    "D4",                        # AO continues
    "CX1", "CX2",                # cross-series confusion
    "TC4", "TC5", "CH2", "CH3",  # TC + CH block
    "WL3", "WL4", "WL5",         # WL block
    "CX3", "CX4",                # more confusion
    "C1", "C2", "C3", "C4",      # AO distractors
    "D25", "D26", "D27", "D28",  # AO palace start
    "TC6", "TC7", "CH4",         # interrupt with other comics
    "C5", "C6",
    "D29", "WL6", "WL7", "TC8",  # palace + interruptions
    "CX5", "CX6",
    "C7", "D30", "CH5", "D31",   # palace continues
    "C8", "TC9", "D32", "WL8",
    "D33", "CH6", "D34",
    "TC10", "D35", "WL9",        # palace ends
    "CX7", "CX8", "CX9", "CX10", # heavy false memories
    "CH7", "WL10", "CH8",        # late other-comic sessions
    "CX11", "CX12", "CX13", "CX14", "CX15",  # final wave
]

# Treasure Comics image root (relative to image_root)
TC_IMAGE_DIR = "ComicScene_Alley_Oop_V3_Dev"

# Treasure Comics sessions (with images)
TC_SESSIONS: Dict[str, Dict[str, Any]] = {
    "TC1": {
        "date": "1933-01-10",
        "dialogues": [
            {
                "round": "TC1:1",
                "user": "I started reading another comic — Treasure Comics, about Paul Bunyan. Here is the first page.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_1.jpg"],
                "image_id": ["TC1:IMG_001"],
                "image_caption": ["Treasure Comics page 1."],
            },
            {
                "round": "TC1:2",
                "user": "This one is about a strongman at a circus. Very different from the caveman comic.",
                "assistant": "Noted. I will keep the two comics separate in memory.",
            },
        ],
    },
    "TC2": {
        "date": "1933-01-12",
        "dialogues": [
            {
                "round": "TC2:1",
                "user": "Here is page 2 of the Paul Bunyan story.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_2.jpg"],
                "image_id": ["TC2:IMG_001"],
                "image_caption": ["Treasure Comics page 2."],
            },
        ],
    },
    "TC3": {
        "date": "1933-01-14",
        "dialogues": [
            {
                "round": "TC3:1",
                "user": "Page 3 of the Paul Bunyan comic.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_3.jpg"],
                "image_id": ["TC3:IMG_001"],
                "image_caption": ["Treasure Comics page 3."],
            },
        ],
    },
    "TC4": {
        "date": "1933-01-16",
        "dialogues": [
            {
                "round": "TC4:1",
                "user": "Page 4 — things are getting exciting at the circus.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_4.jpg"],
                "image_id": ["TC4:IMG_001"],
                "image_caption": ["Treasure Comics page 4."],
            },
        ],
    },
    "TC5": {
        "date": "1933-01-18",
        "dialogues": [
            {
                "round": "TC5:1",
                "user": "Page 5 of Treasure Comics.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_5.jpg"],
                "image_id": ["TC5:IMG_001"],
                "image_caption": ["Treasure Comics page 5."],
            },
        ],
    },
    "TC6": {
        "date": "1933-02-20",
        "dialogues": [
            {
                "round": "TC6:1",
                "user": "Back to Paul Bunyan — here is page 6.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_6.jpg"],
                "image_id": ["TC6:IMG_001"],
                "image_caption": ["Treasure Comics page 6."],
            },
        ],
    },
    "TC7": {
        "date": "1933-02-22",
        "dialogues": [
            {
                "round": "TC7:1",
                "user": "Page 7 of the circus comic.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_7.jpg"],
                "image_id": ["TC7:IMG_001"],
                "image_caption": ["Treasure Comics page 7."],
            },
        ],
    },
    "TC8": {
        "date": "1933-02-24",
        "dialogues": [
            {
                "round": "TC8:1",
                "user": "Page 8 of Paul Bunyan.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_8.jpg"],
                "image_id": ["TC8:IMG_001"],
                "image_caption": ["Treasure Comics page 8."],
            },
        ],
    },
    "TC9": {
        "date": "1933-02-26",
        "dialogues": [
            {
                "round": "TC9:1",
                "user": "Now a completely different story from the same Treasure Comics book — page 12.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_12.jpg"],
                "image_id": ["TC9:IMG_001"],
                "image_caption": ["Treasure Comics page 12."],
            },
        ],
    },
    "TC10": {
        "date": "1933-02-28",
        "dialogues": [
            {
                "round": "TC10:1",
                "user": "And page 20 from Treasure Comics — a detective adventure.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Treasure_Comics_Page_20.jpg"],
                "image_id": ["TC10:IMG_001"],
                "image_caption": ["Treasure Comics page 20."],
            },
        ],
    },
}

# Champ Comics sessions (sports/action/superhero)
CH_PAGES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35]
CH_SESSIONS: Dict[str, Dict[str, Any]] = {
    f"CH{i+1}": {
        "date": f"1933-{(1 + i // 5):02d}-{((i % 5) * 6 + 2):02d}",
        "dialogues": [
            {
                "round": f"CH{i+1}:1",
                "user": f"Reading another comic now — Champ Comics. Here is page {p}." if i == 0 else f"Page {p} of Champ Comics.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Champ_Page_{p}.jpg"],
                "image_id": [f"CH{i+1}:IMG_001"],
                "image_caption": [f"Champ Comics page {p}."],
            },
        ],
    }
    for i, p in enumerate(CH_PAGES)
}

# Western Love sessions (cowboys/romance/western)
WL_PAGES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35]
WL_SESSIONS: Dict[str, Dict[str, Any]] = {
    f"WL{i+1}": {
        "date": f"1933-{(1 + i // 5):02d}-{((i % 5) * 6 + 4):02d}",
        "dialogues": [
            {
                "round": f"WL{i+1}:1",
                "user": f"Now reading Western Love comics. Here is page {p}." if i == 0 else f"Page {p} of Western Love.",
                "assistant": "Stored for later memory questions.",
                "input_image": [f"../image/{TC_IMAGE_DIR}/Western_Love_Page_{p}.jpg"],
                "image_id": [f"WL{i+1}:IMG_001"],
                "image_caption": [f"Western Love page {p}."],
            },
        ],
    }
    for i, p in enumerate(WL_PAGES)
}

# Cross-series distractor sessions (text-only, mixing all 4 comics)
CROSS_SERIES_SESSIONS: Dict[str, Dict[str, Any]] = {
    "CX1": {
        "date": "1933-01-20",
        "dialogues": [
            (
                "Wait, was the big strong guy in the circus the same character as Alley Oop? They are both very muscular.",
                "They are different characters from different comics. The circus strongman is Paul Bunyan, and the caveman is Alley Oop.",
            ),
        ],
    },
    "CX2": {
        "date": "1933-01-22",
        "dialogues": [
            (
                "Both comics have animals. The dinosaur lot and the circus animals — I keep mixing them up.",
                "The dinosaur dealership is in Alley Oop. The circus has different animals. They are separate stories.",
            ),
        ],
    },
    "CX3": {
        "date": "1933-01-24",
        "dialogues": [
            (
                "Both stories have a strong character who gets into fights. It is hard to keep them straight.",
                "Paul Bunyan fights in the circus setting. Alley Oop fights in prehistoric and palace settings. The contexts are different.",
            ),
        ],
    },
    "CX4": {
        "date": "1933-03-25",
        "dialogues": [
            (
                "I think Paul Bunyan was the one who became temporary king and wore a crown.",
                "That sounds like it could be from either story. Kings and crowns appear in many comics.",
            ),
        ],
    },
    "CX5": {
        "date": "1933-03-26",
        "dialogues": [
            (
                "The circus animals were labeled with price tags just like the dinosaur dealership, right?",
                "Price labels on animals is a detail that could appear in either story. It is easy to mix up.",
            ),
        ],
    },
    "CX6": {
        "date": "1933-03-27",
        "dialogues": [
            (
                "Paul Bunyan struck someone when he was offered a position, just like Oop struck someone when offered the throne.",
                "Strong characters reacting aggressively is a common motif. Whether both stories have that exact scene is worth checking.",
            ),
        ],
    },
    "CX7": {
        "date": "1933-03-28",
        "dialogues": [
            (
                "I think the cowboys in Western Love had a circus tent at one point.",
                "Circus tents are a Treasure Comics motif. Western Love is a different setting.",
            ),
        ],
    },
    "CX8": {
        "date": "1933-03-29",
        "dialogues": [
            (
                "The Champ Comics character — was he the one who became temporary king?",
                "The temporary king storyline is from Alley Oop. Champ Comics is a different series with different characters.",
            ),
        ],
    },
    "CX9": {
        "date": "1933-03-30",
        "dialogues": [
            (
                "I am pretty sure I saw a dinosaur in the Western Love comic.",
                "Dinosaurs are an Alley Oop element. Western Love is set in the American West.",
            ),
        ],
    },
    "CX10": {
        "date": "1933-03-31",
        "dialogues": [
            (
                "All four comics had a strong character in a plaid shirt, right?",
                "The plaid shirt is a Paul Bunyan trait in Treasure Comics. Other comics may not feature it.",
            ),
        ],
    },
    "CX11": {
        "date": "1933-04-02",
        "dialogues": [
            (
                "I keep mixing up which comic was in black and white versus color.",
                "The visual style differs across comics. Some are color, some black-and-white.",
            ),
        ],
    },
    "CX12": {
        "date": "1933-04-03",
        "dialogues": [
            (
                "Wait, was there a treasure chest in the Champ comic or the Treasure comic?",
                "The name Treasure Comics suggests treasure themes there. Champ Comics is sports-themed.",
            ),
        ],
    },
    "CX13": {
        "date": "1933-04-04",
        "dialogues": [
            (
                "I think the woman from Western Love also appeared in the Alley Oop palace scenes.",
                "Western Love and Alley Oop have separate casts. Crossovers would be unusual.",
            ),
        ],
    },
    "CX14": {
        "date": "1933-04-05",
        "dialogues": [
            (
                "Did all four comics have the same artist? They look similar to me.",
                "Each comic has its own art style. They are from different publishers and eras.",
            ),
        ],
    },
    "CX15": {
        "date": "1933-04-06",
        "dialogues": [
            (
                "By now I cannot remember which strong man was the boxer and which was the lumberjack.",
                "Paul Bunyan is the lumberjack-circus strongman in Treasure Comics. Champ Comics features other action characters.",
            ),
        ],
    },
}

FOLLOW_UP_DIALOGUES: Dict[str, List[Tuple[str, str]]] = {
    # Step 1: Sanitized follow-ups — visual facts removed, thematic context kept.
    "D1": [
        (
            "The rescue page is easy to confuse with the later crown pages. What is the opening image again?",
            "It opens with someone pointing toward a drowning crowned figure, so the page starts as a water-rescue scene rather than a palace argument.",
        ),
        (
            "What makes that page stand apart from the later palace-crown material?",
            "The soaked ruler later confronts Alley, and the whole page stays centered on the water emergency instead of throne or chore business.",
        ),
    ],
    "D2": [
        (
            "That Dinny page opens mid-ride instead of with the crash itself, right?",
            "Yes. Alley is already riding Dinny in the first panel, and only later does Dinny have a collision.",
        ),
        (
            "So the joke there is a collision, not a water splash or bargain scene?",
            "Exactly. The page is about Dinny's collision and being sore afterward, not about splashing through water or shopping for another dinosaur.",
        ),
    ],
    "D3": [
        (
            "The dinosaur lot had several sales labels. Which comparison matters on that page?",
            "Several labels appear on the lot, and the sizes are not what you might expect from the names.",
        ),
        (
            "What is the punch line of that dealership page after all those sale animals are lined up?",
            "Instead of ending up on one of the larger sale beasts, Alley test-drives the smallest animal on the lot.",
        ),
    ],
    "D4": [
        (
            "The small-dinosaur test-drive page is different from the dealership lineup, right?",
            "Yes. This page is about motion: Alley rides the little dinosaur, then gets kicked off during the test drive.",
        ),
        (
            "And what image closes that page after the kick?",
            "The final close-up shows the small dinosaur's face, which is different from the earlier sales-lot labels.",
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
            "Later on the same page he runs carrying a basin before rushing outside, so the page shifts from lounging to frantic movement.",
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
        "date": "1933-03-15",
        "dialogues": [
            (
                "Looking back at the palace pages from several weeks ago, the palace pages keep flipping between power and embarrassment.",
                "Yes. One distractor scene has a figure working first and resting later, which is the reverse of the chores page where relaxation comes before the frantic running.",
            ),
            (
                "That reversal makes the later palace pages easy to confuse.",
                "Exactly. Several pages share the same cast and setting but reverse the order of work, rest, and punishment.",
            ),
        ],
    },
    "C6": {
        "date": "1933-03-16",
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
        "date": "1933-03-17",
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
        "date": "1933-03-18",
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
    # --- Conflict-introducing sessions (Step 3) ---
    "C9": {
        "date": "1933-03-20",
        "dialogues": [
            (
                "I think the crowd-and-cave page started with the temporary ruler lounging on the throne before the crowd showed up.",
                "That could be how you remember it. Memory for the order of events on busy pages can shift over time.",
            ),
        ],
    },
    "C10": {
        "date": "1933-03-21",
        "dialogues": [
            (
                "I am pretty sure Alley chose the today's special dinosaur at the dealership.",
                "That is one of the labels that appeared on the lot. It is easy to mix up which animal Alley actually rode.",
            ),
        ],
    },
    "C11": {
        "date": "1933-03-22",
        "dialogues": [
            (
                "On the crown-return page, Oop was already shown as a regular citizen before the crown was handed back, was he not?",
                "The crown-return page has several beats that are easy to reorder in memory.",
            ),
        ],
    },
    "C12": {
        "date": "1933-03-23",
        "dialogues": [
            (
                "On the water-rescue page, I think the crowned man was the one pointing at the drowning person.",
                "The opening panel has multiple figures and it is easy to confuse who is doing what.",
            ),
        ],
    },
    # --- More conflict sessions for MMA advantage ---
    "C13": {
        "date": "1933-03-25",
        "dialogues": [
            (
                "I recall the clams-dispute page had the ruler standing up and shouting at the accused, not sitting down.",
                "Throne and hearing scenes do blend together. The ruler's posture is easy to misremember.",
            ),
        ],
    },
    "C14": {
        "date": "1933-03-26",
        "dialogues": [
            (
                "On the chores page, the figure was relaxing outdoors and then ran inside, right?",
                "The indoor-outdoor ordering on that page is one of those details that flips in memory.",
            ),
        ],
    },
    "C15": {
        "date": "1933-03-27",
        "dialogues": [
            (
                "I think the ulcer dinner page only showed the dinner scene without any outdoor fight beforehand.",
                "Some pages do jump straight to the meal. Whether there is an earlier fight panel depends on the specific strip.",
            ),
        ],
    },
    "C16": {
        "date": "1933-03-28",
        "dialogues": [
            (
                "The temporary ruler's situation actually improved over the palace arc — he started off punished but ended up relaxing by the dinner page.",
                "That is one way to read the arc. Whether his situation improved or worsened depends on which pages you compare.",
            ),
        ],
    },
}

# Append conflict sessions to the session plan
SESSION_PLAN.extend(["C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16"])

NOTE_TEXTS: Dict[str, str] = {
    session_id: " ".join(answer for _, answer in dialogues)
    for session_id, dialogues in FOLLOW_UP_DIALOGUES.items()
}

QA_ITEMS: List[Dict[str, Any]] = [
    {"point": [["X2"], ["Y3"]], "question": "How did Oop react when offered the throne?", "options": {"A": "He calmly accepted", "B": "He ran away", "C": "He struck someone", "D": "He asked for time to think"}, "answer": "C", "session_id": ["D27"], "clue": ["D27:1"]},
    {"point": [["X3"], ["Y2"]], "question": "Where did the temporary ruler end up working after the crowd scene?", "options": {"A": "By the water", "B": "Inside a wagon", "C": "In a cave", "D": "In the palace kitchen"}, "answer": "C", "session_id": ["D28"], "clue": ["D28:1"]},
    {"point": [["X3"], ["Y1"]], "question": "During the clams hearing, what were the two accused doing while facing the ruler?", "options": {"A": "Kneeling on the ground", "B": "Raising their hands", "C": "Running away", "D": "Sitting with their heads down"}, "answer": "B", "session_id": ["D30"], "clue": ["D30:1"]},
    {"point": [["X3"], ["Y1"]], "question": "After the crown was returned, what was Oop later seen carrying?", "options": {"A": "A wooden board", "B": "A spear", "C": "A basket of food", "D": "A stone tablet"}, "answer": "A", "session_id": ["D33"], "clue": ["D33:1"]},
    {"point": [["X2"], ["Y3"]], "question": "In the overall story, when did the water rescue happen relative to Oop becoming temporary ruler?", "options": {"A": "Before he became ruler", "B": "After he became ruler", "C": "At the same time", "D": "There was no water rescue"}, "answer": "A", "session_id": ["D1", "D27"], "clue": ["D1:1", "D27:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Which of these events came last in the palace: the crowd pelting, the dinner fight, or the crown return?", "options": {"A": "The crowd pelting", "B": "The dinner fight", "C": "The crown return", "D": "They all happened at the same time"}, "answer": "C", "session_id": ["D28", "D31", "D33"], "clue": ["D28:1", "D31:1", "D33:1"]},
    {"point": [["X2", "X4"], ["Y2"]], "question": "The dinosaur Alley rode during the test drive — was it the same one labeled today's special at the dealership?", "options": {"A": "Yes, it was today's special", "B": "No, it was a different animal", "C": "There was no today's special label", "D": "Alley rode multiple dinosaurs"}, "answer": "B", "session_id": ["D3", "D4"], "clue": ["D3:1", "D4:1"]},
    {"point": [["X4"], ["Y3"]], "question": "Was there actually a dinosaur labeled premium at the dealership, or am I making that up?", "options": {"A": "Yes, there was a premium label", "B": "No, there was no premium label", "C": "The label said deluxe, not premium", "D": "All dinosaurs had the same label"}, "answer": "B", "session_id": ["D3"], "clue": ["D3:1"]},
    {"point": [["X3"], ["Y3"]], "question": "I recall the ruler standing up during the clams hearing. Was he actually standing or seated?", "options": {"A": "Standing and shouting", "B": "Seated on a throne", "C": "Walking around the room", "D": "Lying down"}, "answer": "B", "session_id": ["D30"], "clue": ["D30:1"]},
    {"point": [["X2"], ["Y3"]], "question": "I think the temporary ruler started off relaxing in the crowd scene before things went bad. Is that right?", "options": {"A": "Yes, he was relaxing first", "B": "No, the crowd was already pelting him", "C": "There was no crowd scene", "D": "The ruler was not in that scene"}, "answer": "B", "session_id": ["D28"], "clue": ["D28:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Which happened first in the story: Dinny crashing into something, or the dinosaur dealership visit?", "options": {"A": "Dinny's crash came first", "B": "The dealership came first", "C": "They happened at the same time", "D": "Neither of those happened"}, "answer": "A", "session_id": ["D2", "D3"], "clue": ["D2:1", "D3:1"]},
    {"point": [["X2"], ["Y3"]], "question": "In the palace arc, which event came first: the dinner fight or the crowd pelting?", "options": {"A": "The dinner fight", "B": "The crowd pelting", "C": "They happened on the same page", "D": "Neither happened"}, "answer": "B", "session_id": ["D28", "D31"], "clue": ["D28:1", "D31:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Did all the dinosaur scenes happen before the palace takeover arc, or were they mixed in?", "options": {"A": "All dinosaur scenes came first", "B": "They were mixed in with palace scenes", "C": "Dinosaur scenes came after the palace", "D": "There were no dinosaur scenes"}, "answer": "A", "session_id": ["D2", "D3", "D4", "D27"], "clue": ["D2:1", "D3:1", "D4:1", "D27:1"]},
    {"point": [["X4"], ["Y2"]], "question": "Thinking back to the very first page, was the drowning figure wearing a crown or not?", "options": {"A": "Yes, wearing a crown", "B": "No crown visible", "C": "Wearing a helmet instead", "D": "Cannot tell from the image"}, "answer": "A", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "I think Oop rescued the king by pulling him out of a cave. Is that what happened?", "options": {"A": "Yes, from a cave", "B": "No, from water — he was drowning", "C": "Oop did not rescue anyone", "D": "The king rescued himself"}, "answer": "B", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X4"], ["Y3"]], "question": "Was there a dinosaur labeled 'bargain' at the dealership that was bigger than the 'cheap' one?", "options": {"A": "Yes, the bargain was bigger", "B": "No, the bargain was not bigger than the cheap one", "C": "There was no bargain label", "D": "They were the same size"}, "answer": "B", "session_id": ["D3"], "clue": ["D3:1"]},
    {"point": [["X2"], ["Y3"]], "question": "I think the crown was taken from the king by force. Was it actually given willingly?", "options": {"A": "Taken by force", "B": "Given willingly by the king", "C": "The crown was never transferred", "D": "Oop made his own crown"}, "answer": "B", "session_id": ["D27"], "clue": ["D27:1"]},
    {"point": [["X4"], ["Y2"]], "question": "In the clams hearing scene, was the ruler sitting on a fancy throne or a simple seat?", "options": {"A": "A simple seat on the ground", "B": "A fancy golden throne", "C": "Standing — there was no seat", "D": "Lying in a hammock"}, "answer": "A", "session_id": ["D30"], "clue": ["D30:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Over the course of the palace story, how did things change for the temporary ruler?", "options": {"A": "His situation got progressively worse", "B": "His situation improved over time", "C": "Nothing really changed", "D": "He gained more power"}, "answer": "A", "session_id": ["D28", "D29", "D31"], "clue": ["D28:1", "D29:1", "D31:1"]},
    {"point": [["X2"], ["Y3"]], "question": "I feel like the temporary ruler's situation actually got better over time in the palace. Did it really get worse or better?", "options": {"A": "It got worse", "B": "It got better", "C": "It stayed the same", "D": "He was never a temporary ruler"}, "answer": "A", "session_id": ["D28", "D29", "D31"], "clue": ["D28:1", "D29:1", "D31:1"]},
    {"point": [["X4"], ["Y1"]], "question": "How many sale labels were there at the dinosaur lot?", "options": {"A": "2", "B": "3", "C": "4", "D": "5"}, "answer": "B", "session_id": ["D3"], "clue": ["D3:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Did the original king seem grateful or hostile toward Oop right after being rescued from the water?", "options": {"A": "Grateful — he thanked Oop", "B": "Hostile — he confronted Oop", "C": "Indifferent", "D": "Oop did not rescue anyone"}, "answer": "B", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Did the clams hearing happen before or after the chores scene with the basin?", "options": {"A": "Before the chores", "B": "After the chores", "C": "At the same time", "D": "There was no clams hearing"}, "answer": "B", "session_id": ["D29", "D30"], "clue": ["D29:1", "D30:1"]},
    {"point": [["X2"], ["Y3"]], "question": "By the end of the palace arc, was the temporary ruler punished more or less than at the beginning?", "options": {"A": "More — punishments escalated", "B": "Less — things calmed down", "C": "About the same throughout", "D": "He was never punished"}, "answer": "A", "session_id": ["D28", "D29", "D31"], "clue": ["D28:1", "D29:1", "D31:1"]},
    # =================================================================
    # Cross-series entity disambiguation (caption-proof: X3/X4)
    # =================================================================
    {"point": [["X2"], ["Y2"]], "question": "The character who rode a dinosaur during a test drive — was that from the caveman comic or the circus comic?", "options": {"A": "The caveman comic (Alley Oop)", "B": "The circus comic (Treasure Comics)", "C": "Both comics had dinosaur rides", "D": "Neither comic had a dinosaur ride"}, "answer": "A", "session_id": ["D4"], "clue": ["D4:1"]},
    {"point": [["X2"], ["Y2"]], "question": "Which comic featured a character in a red plaid shirt — the caveman story or the circus story?", "options": {"A": "The caveman story", "B": "The circus story (Paul Bunyan)", "C": "Both had plaid shirts", "D": "Neither had plaid shirts"}, "answer": "B", "session_id": ["TC3"], "clue": ["TC3:1"]},
    {"point": [["X2"], ["Y2"]], "question": "Animals kept in cages appeared in which comic — the one with dinosaurs or the one with the circus?", "options": {"A": "The dinosaur comic", "B": "The circus comic", "C": "Both comics", "D": "Neither comic"}, "answer": "B", "session_id": ["TC5"], "clue": ["TC5:1"]},
    {"point": [["X2"], ["Y3"]], "question": "A character got crowned as temporary king in one of the comics. Which one?", "options": {"A": "The caveman comic (Alley Oop)", "B": "The circus comic (Paul Bunyan)", "C": "Both comics", "D": "Neither comic"}, "answer": "A", "session_id": ["D27"], "clue": ["D27:1"]},
    {"point": [["X4"], ["Y2"]], "question": "In the circus comic, what pattern was on Paul Bunyan's shirt in most scenes?", "options": {"A": "Red and black plaid/checkered", "B": "Solid blue", "C": "Striped", "D": "He was shirtless throughout"}, "answer": "A", "session_id": ["TC3", "TC4"], "clue": ["TC3:1", "TC4:1"]},
    {"point": [["X4"], ["Y2"]], "question": "In the very first circus comic page, was Paul Bunyan wearing his plaid shirt or was he shirtless?", "options": {"A": "Shirtless", "B": "Wearing the plaid shirt", "C": "Wearing a suit", "D": "He was not on the first page"}, "answer": "A", "session_id": ["TC1"], "clue": ["TC1:1"]},
    {"point": [["X3"], ["Y2"]], "question": "In the circus comic, were there police officers shown in any scene?", "options": {"A": "Yes, officers in dark uniforms appeared", "B": "No police in the circus comic", "C": "Only in the caveman comic", "D": "Police appeared in both comics"}, "answer": "A", "session_id": ["TC5", "TC8"], "clue": ["TC5:1", "TC8:1"]},
    {"point": [["X2"], ["Y3"]], "question": "I think Paul Bunyan was the one who became temporary king. Is that right?", "options": {"A": "Yes, Paul Bunyan became king", "B": "No, that was Oop from the caveman comic", "C": "Neither character became king", "D": "Both characters became king in their own stories"}, "answer": "B", "session_id": ["D27"], "clue": ["D27:1"]},
    # =================================================================
    # Within-series recall after interruption (tests memory persistence)
    # =================================================================
    {"point": [["X2"], ["Y2"]], "question": "Going back to the very first comic you read — in the caveman water rescue scene, was someone drowning?", "options": {"A": "Yes, a crowned figure was drowning", "B": "No, everyone was on dry land", "C": "That was from the circus comic", "D": "There was no water rescue scene"}, "answer": "A", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X4"], ["Y2"]], "question": "In the caveman comic's dealership scene, were the dinosaurs displayed outdoors or indoors?", "options": {"A": "Outdoors, lined up in the open", "B": "Indoors in a building", "C": "In cages", "D": "That scene was in the circus comic"}, "answer": "A", "session_id": ["D3"], "clue": ["D3:1"]},
    {"point": [["X3"], ["Y3"]], "question": "The scene with a fire breaking out — was that in the circus comic or the caveman comic?", "options": {"A": "The circus comic", "B": "The caveman comic", "C": "Both comics had fire scenes", "D": "Neither comic had fire"}, "answer": "A", "session_id": ["TC4"], "clue": ["TC4:1"]},
    # =================================================================
    # Cross-series temporal ordering
    # =================================================================
    {"point": [["X2"], ["Y3"]], "question": "Did you start reading the circus comic before or after the caveman palace arc?", "options": {"A": "Before the palace arc", "B": "After the palace arc", "C": "At the same time", "D": "I never read a circus comic"}, "answer": "A", "session_id": ["TC1", "D27"], "clue": ["TC1:1", "D27:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Was the caveman water rescue scene or the circus introduction the first thing you read?", "options": {"A": "The caveman water rescue came first", "B": "The circus introduction came first", "C": "They were on the same day", "D": "Neither happened"}, "answer": "A", "session_id": ["D1", "TC1"], "clue": ["D1:1", "TC1:1"]},
    # =================================================================
    # Cross-series false memory detection
    # =================================================================
    {"point": [["X2"], ["Y3"]], "question": "I think the circus had animals with price tags just like the dinosaur dealership. Is that true?", "options": {"A": "Yes, both had price-tagged animals", "B": "No, only the dinosaur dealership had price labels", "C": "The circus had price labels but not the dealership", "D": "Neither had price labels"}, "answer": "B", "session_id": ["D3", "TC5"], "clue": ["D3:1", "TC5:1"]},
    {"point": [["X2"], ["Y3"]], "question": "I remember Paul Bunyan striking someone when offered a job, just like Oop struck someone when offered the throne. Did Paul Bunyan actually do that?", "options": {"A": "Yes, both characters struck someone", "B": "No, only Oop struck someone when offered power", "C": "Paul Bunyan struck someone but Oop did not", "D": "Neither struck anyone"}, "answer": "B", "session_id": ["D27", "TC4"], "clue": ["D27:1", "TC4:1"]},
    # =================================================================
    # 4-comic series disambiguation (caption-proof)
    # =================================================================
    {"point": [["X2"], ["Y2"]], "question": "Which comic was set in the American West with cowboys?", "options": {"A": "Alley Oop", "B": "Treasure Comics (Paul Bunyan)", "C": "Champ Comics", "D": "Western Love"}, "answer": "D", "session_id": ["WL1"], "clue": ["WL1:1"]},
    {"point": [["X2"], ["Y2"]], "question": "Which comic featured a sports/action superhero called The Human Meteor?", "options": {"A": "Alley Oop", "B": "Treasure Comics", "C": "Champ Comics", "D": "Western Love"}, "answer": "C", "session_id": ["CH1"], "clue": ["CH1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Of the 4 comics you read, which had dinosaurs?", "options": {"A": "Champ Comics", "B": "Alley Oop", "C": "Western Love", "D": "Treasure Comics"}, "answer": "B", "session_id": ["D3"], "clue": ["D3:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Which comic had a circus strongman as the main character?", "options": {"A": "Champ Comics", "B": "Western Love", "C": "Treasure Comics (Paul Bunyan)", "D": "Alley Oop"}, "answer": "C", "session_id": ["TC1"], "clue": ["TC1:1"]},
    {"point": [["X4"], ["Y2"]], "question": "The Champ Comics cover featured which color theme?", "options": {"A": "Mostly red and yellow with a hero in green", "B": "Black and white only", "C": "Pastel pink", "D": "Sepia tones"}, "answer": "A", "session_id": ["CH1"], "clue": ["CH1:1"]},
    {"point": [["X4"], ["Y2"]], "question": "In the Western Love first page, what is the name of the story shown?", "options": {"A": "The Girl from Ghost Town", "B": "Paul Bunyan's Adventure", "C": "Alley Oop in Love", "D": "The Champ Returns"}, "answer": "A", "session_id": ["WL1"], "clue": ["WL1:1"]},
    {"point": [["X3"], ["Y2"]], "question": "Were the cowboys in Western Love shown on horseback in the early pages?", "options": {"A": "Yes, on horses", "B": "No, only on foot", "C": "Riding wagons", "D": "There were no cowboys"}, "answer": "A", "session_id": ["WL1", "WL2"], "clue": ["WL1:1", "WL2:1"]},
    # =================================================================
    # Cross-series false memory rejection (uses CX sessions)
    # =================================================================
    {"point": [["X2"], ["Y3"]], "question": "I think the cowboys in Western Love had a circus tent in their story. Did they?", "options": {"A": "Yes, there was a circus tent", "B": "No, that was Treasure Comics", "C": "Both comics had circus tents", "D": "Neither had circus tents"}, "answer": "B", "session_id": ["WL1", "TC1"], "clue": ["WL1:1", "TC1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Was the Champ Comics character the one who became a temporary king?", "options": {"A": "Yes, the Champ became king", "B": "No, that was Oop from Alley Oop", "C": "Both characters became king", "D": "Neither became king"}, "answer": "B", "session_id": ["CH1", "D27"], "clue": ["CH1:1", "D27:1"]},
    {"point": [["X2"], ["Y3"]], "question": "I remember seeing a dinosaur in the Western Love comic. Was that real?", "options": {"A": "Yes, dinosaurs were in Western Love", "B": "No, dinosaurs were only in Alley Oop", "C": "Dinosaurs appeared in all four comics", "D": "I never read Western Love"}, "answer": "B", "session_id": ["D3", "WL1"], "clue": ["D3:1", "WL1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Did all 4 comics feature characters in plaid shirts?", "options": {"A": "Yes, all 4 had plaid shirts", "B": "No, the plaid shirt is mainly Paul Bunyan in Treasure Comics", "C": "Only Western Love had plaid shirts", "D": "None of the comics had plaid shirts"}, "answer": "B", "session_id": ["TC3"], "clue": ["TC3:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Was there a treasure chest in Champ Comics or Treasure Comics?", "options": {"A": "Champ Comics", "B": "Treasure Comics", "C": "Both", "D": "Neither"}, "answer": "B", "session_id": ["TC10"], "clue": ["TC10:1"]},
    # =================================================================
    # Temporal ordering across all 4 comics
    # =================================================================
    {"point": [["X2"], ["Y3"]], "question": "Which comic series did you start reading first?", "options": {"A": "Alley Oop", "B": "Treasure Comics", "C": "Champ Comics", "D": "Western Love"}, "answer": "A", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Which came first in your reading: the Champ Comics cover or the Western Love opener?", "options": {"A": "Champ Comics first", "B": "Western Love first", "C": "Same day", "D": "I never read either"}, "answer": "B", "session_id": ["CH1", "WL1"], "clue": ["CH1:1", "WL1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Did you finish the Alley Oop palace arc before or after starting Champ Comics?", "options": {"A": "Before starting Champ", "B": "After starting Champ", "C": "Never started Champ", "D": "Never finished the palace arc"}, "answer": "B", "session_id": ["CH1", "D33"], "clue": ["CH1:1", "D33:1"]},
    # =================================================================
    # Within-series recall after cross-series interruption
    # =================================================================
    {"point": [["X4"], ["Y2"]], "question": "Long after switching between several comics, can you recall: in the very first Alley Oop scene, was someone drowning?", "options": {"A": "Yes, a crowned figure was drowning in water", "B": "No, everyone was on dry land", "C": "That scene was in Champ Comics", "D": "There was no drowning anywhere"}, "answer": "A", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X3"], ["Y3"]], "question": "Among all the comics, where did the dealership scene with labeled dinosaurs appear — and were the animals indoors or outdoors?", "options": {"A": "Treasure Comics, indoors", "B": "Alley Oop, outdoors in the open", "C": "Champ Comics, in cages", "D": "Western Love, in a barn"}, "answer": "B", "session_id": ["D3"], "clue": ["D3:1"]},
    {"point": [["X4"], ["Y2"]], "question": "Recalling the Treasure Comics very first page, was Paul Bunyan wearing his plaid shirt or shirtless?", "options": {"A": "Wearing the plaid shirt", "B": "Shirtless, with bare arms", "C": "Wearing a suit", "D": "Wearing a tunic"}, "answer": "B", "session_id": ["TC1"], "clue": ["TC1:1"]},
    {"point": [["X3"], ["Y2"]], "question": "On the Champ Comics cover, where is The Human Meteor positioned relative to the divers?", "options": {"A": "Above them, flying or swimming horizontally", "B": "Below them on the seafloor", "C": "Behind them out of view", "D": "There were no divers"}, "answer": "A", "session_id": ["CH1"], "clue": ["CH1:1"]},
    # =================================================================
    # Caption-proof visual micro-attributes (X4)
    # =================================================================
    {"point": [["X4"], ["Y1"]], "question": "What color is The Human Meteor's costume on the Champ Comics cover?", "options": {"A": "Red and blue", "B": "Green and yellow", "C": "Black with white cape", "D": "Purple"}, "answer": "B", "session_id": ["CH1"], "clue": ["CH1:1"]},
    {"point": [["X4"], ["Y1"]], "question": "What background color dominates the Champ Comics cover?", "options": {"A": "Underwater blue", "B": "Sky pink", "C": "Forest green", "D": "Desert tan"}, "answer": "A", "session_id": ["CH1"], "clue": ["CH1:1"]},
    {"point": [["X4"], ["Y1"]], "question": "How are the divers on the Champ Comics cover dressed?", "options": {"A": "Modern scuba gear", "B": "Old-style diving suits with metal helmets", "C": "Just swimming trunks", "D": "Knight armor"}, "answer": "B", "session_id": ["CH1"], "clue": ["CH1:1"]},
    {"point": [["X4"], ["Y1"]], "question": "In the Western Love opening page, what is the building behind the cowboys?", "options": {"A": "A church", "B": "A saloon", "C": "A train station", "D": "A barn"}, "answer": "B", "session_id": ["WL2"], "clue": ["WL2:1"]},
    {"point": [["X4"], ["Y1"]], "question": "What kind of hats are the Western Love cowboys wearing?", "options": {"A": "Top hats", "B": "Cowboy hats with wide brims", "C": "Baseball caps", "D": "No hats"}, "answer": "B", "session_id": ["WL2"], "clue": ["WL2:1"]},
    {"point": [["X4"], ["Y1"]], "question": "In Champ Comics, what color is the gangsters' car?", "options": {"A": "Red", "B": "Yellow", "C": "Black/dark", "D": "White"}, "answer": "C", "session_id": ["CH2"], "clue": ["CH2:1"]},
    {"point": [["X4"], ["Y1"]], "question": "In Treasure Comics, was Paul Bunyan ever shown holding a barbell or weights?", "options": {"A": "Yes, lifting heavy weights overhead", "B": "No, he never lifted anything", "C": "Only a sword", "D": "Only a hammer"}, "answer": "A", "session_id": ["TC1"], "clue": ["TC1:1"]},
    # =================================================================
    # Caption-proof spatial reasoning (X3)
    # =================================================================
    {"point": [["X3"], ["Y2"]], "question": "In the Western Love saloon scene, are the cowboys outside the saloon or inside it?", "options": {"A": "Outside, on horses near the building", "B": "Inside drinking", "C": "On the roof", "D": "There is no saloon"}, "answer": "A", "session_id": ["WL2"], "clue": ["WL2:1"]},
    {"point": [["X3"], ["Y2"]], "question": "On the Treasure Comics first page, is Paul Bunyan shown above or below the price tags hanging in the panel?", "options": {"A": "Above the price tags", "B": "Below the price tags", "C": "Beside them at the same level", "D": "There are no price tags"}, "answer": "B", "session_id": ["TC1"], "clue": ["TC1:1"]},
    {"point": [["X3"], ["Y2"]], "question": "In the Champ Comics chase scene, are the gangsters in front of or behind the hero?", "options": {"A": "In front, with the hero chasing them", "B": "Behind the hero", "C": "Beside the hero", "D": "There is no chase"}, "answer": "A", "session_id": ["CH2"], "clue": ["CH2:1"]},
    # =================================================================
    # Cross-series visual identity (caption never captures this)
    # =================================================================
    {"point": [["X2", "X4"], ["Y3"]], "question": "Which character wore a red costume with green elements: Alley Oop, Paul Bunyan, the Champ hero, or a Western Love cowboy?", "options": {"A": "Alley Oop", "B": "Paul Bunyan", "C": "The Champ hero (green/yellow costume)", "D": "A Western Love cowboy"}, "answer": "C", "session_id": ["CH1"], "clue": ["CH1:1"]},
    {"point": [["X2", "X4"], ["Y3"]], "question": "Which comic featured a character wielding a sword in a fight scene?", "options": {"A": "Alley Oop", "B": "Treasure Comics", "C": "Champ Comics", "D": "Western Love"}, "answer": "B", "session_id": ["TC4"], "clue": ["TC4:1"]},
    {"point": [["X2"], ["Y2"]], "question": "Which comic was set in the modern era with cars and city streets?", "options": {"A": "Alley Oop (prehistoric)", "B": "Treasure Comics (early circus era)", "C": "Champ Comics (modern with cars)", "D": "Western Love (frontier era)"}, "answer": "C", "session_id": ["CH2"], "clue": ["CH2:1"]},
    {"point": [["X2"], ["Y2"]], "question": "Which two of the four comics had clear historical or period settings (not modern)?", "options": {"A": "Alley Oop and Champ Comics", "B": "Alley Oop and Western Love", "C": "Treasure Comics and Champ Comics", "D": "All four were modern"}, "answer": "B", "session_id": ["D1", "WL1"], "clue": ["D1:1", "WL1:1"]},
    # =================================================================
    # Long-tail recall after heavy interruption
    # =================================================================
    {"point": [["X4"], ["Y2"]], "question": "Recall the very first thing you saw in Alley Oop — what color was the drowning figure's crown?", "options": {"A": "Gold/yellow", "B": "Silver", "C": "Black", "D": "There was no crown visible"}, "answer": "A", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X4"], ["Y2"]], "question": "Of the dinosaur dealership, the circus tent, the saloon, and the underwater scene — which two were in COLOR comics?", "options": {"A": "Dinosaur dealership and saloon", "B": "Circus tent and underwater", "C": "All four were in color", "D": "Only the saloon was in color"}, "answer": "C", "session_id": ["D3", "TC1", "WL2", "CH1"], "clue": ["D3:1", "TC1:1", "WL2:1", "CH1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "How many distinct comic series did you read in total?", "options": {"A": "2", "B": "3", "C": "4", "D": "5"}, "answer": "C", "session_id": ["D1", "TC1", "CH1", "WL1"], "clue": ["D1:1", "TC1:1", "CH1:1", "WL1:1"]},
    {"point": [["X2"], ["Y3"]], "question": "Among the 4 comics, which had the LEAST modern setting (oldest historical era)?", "options": {"A": "Alley Oop (prehistoric/caveman)", "B": "Treasure Comics", "C": "Champ Comics", "D": "Western Love"}, "answer": "A", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X2"], ["Y2"]], "question": "Recall: did the Treasure Comics character ever fight a wild animal in the early pages?", "options": {"A": "Yes, he wrestled circus animals", "B": "No, only humans", "C": "Only dinosaurs", "D": "He never fought"}, "answer": "A", "session_id": ["TC2", "TC5"], "clue": ["TC2:1", "TC5:1"]},
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


def _build_tc_session(session_id: str) -> Dict[str, Any]:
    """Build a Treasure Comics session (has pre-built dialogue dicts with images)."""
    spec = TC_SESSIONS[session_id]
    return {
        'session_id': session_id,
        'date': spec['date'],
        'dialogues': copy.deepcopy(spec['dialogues']),
    }


def _build_ch_session(session_id: str) -> Dict[str, Any]:
    spec = CH_SESSIONS[session_id]
    return {
        'session_id': session_id,
        'date': spec['date'],
        'dialogues': copy.deepcopy(spec['dialogues']),
    }


def _build_wl_session(session_id: str) -> Dict[str, Any]:
    spec = WL_SESSIONS[session_id]
    return {
        'session_id': session_id,
        'date': spec['date'],
        'dialogues': copy.deepcopy(spec['dialogues']),
    }


def _build_cx_session(session_id: str) -> Dict[str, Any]:
    """Build a cross-series distractor session (tuple format)."""
    spec = CROSS_SERIES_SESSIONS[session_id]
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
        elif session_id in TC_SESSIONS:
            ordered_sessions.append(_build_tc_session(session_id))
        elif session_id in CH_SESSIONS:
            ordered_sessions.append(_build_ch_session(session_id))
        elif session_id in WL_SESSIONS:
            ordered_sessions.append(_build_wl_session(session_id))
        elif session_id in CROSS_SERIES_SESSIONS:
            ordered_sessions.append(_build_cx_session(session_id))
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
