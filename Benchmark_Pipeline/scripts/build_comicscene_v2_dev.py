#!/usr/bin/env python3
"""
Build a smaller ComicScene dev set that is designed to separate:
  - target_session_context: target sessions only
  - full_context: all sessions mixed together

The dev set keeps a handful of anchor image pages, adds later visual decoys
from the same story arc, and injects text-only confusion sessions that reuse
the same characters and motifs without explicitly restating target answers.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIALOG = REPO_ROOT / "Benchmark_Pipeline" / "data" / "dialog" / "ComicScene_Alley_Oop_Draft.json"
OUTPUT_DIALOG = REPO_ROOT / "Benchmark_Pipeline" / "data" / "dialog" / "ComicScene_Alley_Oop_V2_Dev.json"

ANCHOR_SESSION_IDS = ["D1", "D2", "D3", "D4", "D27", "D28", "D29"]

NEW_IMAGE_SESSIONS = [
    {
        "session_id": "D30",
        "date": "1933-02-10",
        "page": 30,
        "user": "Here's another later page from the palace subplot of the same Alley Oop story.",
        "assistant": "I see the later palace page. I'll keep it separate from the earlier king-for-a-day scenes.",
    },
    {
        "session_id": "D31",
        "date": "1933-02-11",
        "page": 31,
        "user": "I kept reading the same palace storyline. Here's the next page.",
        "assistant": "Got it. I'll track this page as a later palace scene with the same cast.",
    },
    {
        "session_id": "D32",
        "date": "1933-02-12",
        "page": 32,
        "user": "Here's another page from the same palace arc. Please keep following the details carefully.",
        "assistant": "I see it. I'll keep this page distinct from the earlier throne, crowd, and chore scenes.",
    },
    {
        "session_id": "D33",
        "date": "1933-02-13",
        "page": 33,
        "user": "I've got one more later page from the same Alley Oop palace sequence.",
        "assistant": "Understood. I'll keep this later page in mind without mixing it up with the earlier crown scenes.",
    },
]

IMAGE_RECAPS: Dict[str, Tuple[str, str]] = {
    "D1": (
        "Please give me a short recap of page 1 so I do not mix it up with the later crown scenes.",
        "Page 1 starts with a crowned figure crying for help from the water while a woman points. Alley pulls him out, and the still-wet ruler later points accusingly back at Alley.",
    ),
    "D2": (
        "Summarize page 2 for me before I blend it into the dealership material.",
        "Page 2 shows Alley already riding Dinny, then Dinny crashes into a tree. The page is about the tree collision and Dinny being sore, not about water or bargaining.",
    ),
    "D3": (
        "Recap the dealership page in one compact memory note.",
        "The dinosaur lot page lines up creatures labeled cheap, bargain, and today's special, plus a little baby dinosaur. Alley ends up test-driving the little baby rather than the larger sale animals.",
    ),
    "D4": (
        "Give me a quick reminder of what happens on the test-drive page.",
        "The test-drive page keeps Alley on the small compact dinosaur, then the little dinosaur kicks him off and the final close-up shows the animal drooling.",
    ),
    "D27": (
        "Recap the page where the throne argument starts.",
        "On page 27 the crowned ruler tells Oop he ought to take the throne for a while, and by the end of the page Oop strikes the crowned figure instead of calmly accepting the idea.",
    ),
    "D28": (
        "Summarize the crowd-and-cave page so I keep its sequence straight.",
        "On page 28 the temporary ruler is pelted by the crowd outside, then later the same page shows him working inside a cave. It is a punishment-and-labor page, not a lounging page.",
    ),
    "D29": (
        "Give me a recap of the chores page with the basin and fake bon-bons.",
        "The chores page begins with a reclining ruler beside round treats that are said not to be bon-bons. Later the same page shows the former ruler running with a basin on his head and then rushing outside.",
    ),
    "D30": (
        "Recap the clams-dispute page in one memory note.",
        "Page 30 centers on a clams dispute: two subjects face the ruler with their hands raised, the ruler stays seated while hearing the complaint, and the page keeps returning to the stolen-clams argument.",
    ),
    "D31": (
        "Summarize the ulcer dinner page so I do not confuse it with the other palace pages.",
        "Page 31 begins with fighting outside, then shifts to a dinner table where the temporary ruler complains about food and ulcers before getting hit at the table.",
    ),
    "D32": (
        "Give me a short recap of the peek-a-boo palace page.",
        "Page 32 continues the ulcer-and-dinner sequence, then Oop sneaks up behind the necklace-wearing man with a peek-a-boo and hits him during the meal scene.",
    ),
    "D33": (
        "Recap the apology-and-crown-return page in one note.",
        "Page 33 includes an apology, the crown being handed back, Oop becoming an ordinary citizen again, and a later panel where he is carrying a wooden board near the strange-duck remark.",
    ),
}

TEXT_CONFUSION_SESSIONS: List[Dict[str, Any]] = [
    {
        "session_id": "C1",
        "date": "1933-01-01",
        "dialogues": [
            (
                "We have already had one page where a crowned man ends up soaked and another much later page where a substitute ruler gets pelted by a crowd. Those public-humiliation scenes are easy to blend together.",
                "Exactly. The story keeps reusing crowns, public embarrassment, and rescue beats, so later palace scenes can overwrite the earlier water rescue page if you are not tracking the page boundaries carefully.",
            ),
            (
                "So if I ask later whether someone was still in trouble by the final panel, I should be careful not to mix the rescue page with the palace pages.",
                "Right. The shared crown imagery is memorable, but the actual actions at the ends of those pages are different.",
            ),
        ],
    },
    {
        "session_id": "C2",
        "date": "1933-01-02",
        "dialogues": [
            (
                "The dinosaur material is also easy to confuse. One page is about Dinny crashing, another is about trading him in, and another is about that smaller trial dinosaur.",
                "Yes. The comic reuses similar dinosaurs across nearby pages, so a model that just scans for dinosaur words can easily mix the accident page, the dealership page, and the later test-drive page.",
            ),
            (
                "Especially because the big one and the little one are being judged against each other in conversation.",
                "Exactly. Size comparisons, trade-in talk, and test-drive talk recur across those pages, but they do not point to the same image evidence.",
            ),
        ],
    },
    {
        "session_id": "C3",
        "date": "1933-01-03",
        "dialogues": [
            (
                "The dealership page is funny because all the labels make it feel easy, but then the next page reuses the same small dinosaur in motion.",
                "Right. Static lineup pages and action pages can look related even when the evidence you need is different. One page is about labels and placement, while another is about what the little dinosaur actually does.",
            ),
            (
                "So remembering the order of events matters more than just remembering that there was a small dinosaur around.",
                "Yes. If you collapse those pages into one generic memory, you lose whether the evidence is a label, a ride, a kick, or drool in a later panel.",
            ),
        ],
    },
    {
        "session_id": "C4",
        "date": "1933-02-07",
        "dialogues": [
            (
                "Once the palace subplot starts, the comic keeps circling around the crown, work, punishment, and replacement-ruler jokes.",
                "Exactly. The same two or three visual motifs repeat: the crown changes heads, the crowd reacts, and someone ends up doing work or getting struck.",
            ),
            (
                "That means a question about the crowned figure can be ambiguous if you only remember the general theme instead of the specific page.",
                "Yes. The correct answer often depends on whether the page is before the crowd scene, after it, or already in the domestic chore sequence.",
            ),
        ],
    },
    {
        "session_id": "C5",
        "date": "1933-02-09",
        "dialogues": [
            (
                "The later palace pages also stack new arguments on top of the king-for-a-day joke: clams, taxes, meals, and complaints about ulcers.",
                "Right. Those later cave-and-palace pages create strong lexical overlap with the earlier ruler pages, but the physical actions are very different from the crowd, cave-work, and basin scenes.",
            ),
            (
                "So if a model just grabs the latest crown-and-palace references, it could miss whether a question was really about food, chores, or getting hit.",
                "Exactly. The overlap is semantic enough to distract full-context baselines even though the target evidence still lives in one or two specific pages.",
            ),
        ],
    },
    {
        "session_id": "C6",
        "date": "1933-02-10",
        "dialogues": [
            (
                "I can already tell I might mix up the scene with the thrown objects, the scene with the cave labor, and the scene with the lounging ruler eating treats.",
                "That is the intended trap. Those pages share the same ruler and setting, but the key detail can be whether he is being pelted, working, reclining, or running with some object on him.",
            ),
            (
                "And some of the palace pages look like simple relaxation scenes until something else happens later on the same page.",
                "Yes. Several pages pivot from comfort to punishment or from authority to embarrassment, so late-panel evidence matters a lot.",
            ),
        ],
    },
    {
        "session_id": "C7",
        "date": "1933-02-11",
        "dialogues": [
            (
                "The chores sequence is especially confusable because it starts with one image of comfort and then turns into work, injury, and rushing around.",
                "Exactly. That sequence punishes shallow retrieval because the earliest panel and the later panels imply different states for the same character.",
            ),
            (
                "So if a question mentions a relaxed ruler, I should not assume the whole page stayed relaxed.",
                "Right. The answer can hinge on the transition within that page rather than the opening panel alone.",
            ),
        ],
    },
    {
        "session_id": "C8",
        "date": "1933-02-12",
        "dialogues": [
            (
                "By the time the palace pages keep escalating, I can imagine confusing the temporary ruler, the crowned ruler, and the later former ruler with each other.",
                "Yes. Role changes happen quickly in this arc, so models that rely on global gist instead of page-local state will often bind the wrong action to the wrong person.",
            ),
            (
                "That seems like the main thing we want to test: whether the model can isolate the right page instead of relying on the general palace storyline.",
                "Exactly. The target evidence is still local, but the full conversation now contains many tempting near-matches.",
            ),
        ],
    },
]

QA_ITEMS: List[Dict[str, Any]] = [
    {
        "point": [["X2"], ["Y2"]],
        "question": "On the page where the crowned figure cries for help from the water, is he still in the water in the final panel? Reply with only yes or no in lowercase, with no punctuation.",
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
        "point": [["X4"], ["Y1"]],
        "question": "On the tree-crash page, is Dinny shown splashing through water after the crash? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
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
        "point": [["X3"], ["Y2"]],
        "question": "On the test-drive page, does the small dinosaur kick Alley off before the final drooling close-up? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D4"],
        "clue": ["D4:1"],
    },
    {
        "point": [["X3"], ["Y1"]],
        "question": "Who suggests that Oop ought to take the ruler's place on that page: the crowned man, Oop, or a guard? Reply with only one of: crowned man, Oop, guard.",
        "answer": "crowned man",
        "session_id": ["D27"],
        "clue": ["D27:1"],
    },
    {
        "point": [["X2", "X3"], ["Y2"]],
        "question": "On the page where the ruler tells Oop he should take the throne, is the crowned ruler the one who gets hit by the end of the page? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D27"],
        "clue": ["D27:1"],
    },
    {
        "point": [["X3"], ["Y2"]],
        "question": "Later on that same crowd-and-cave page, where is the temporary ruler shown working: in a cave, by the water, or inside a wagon? Reply with only one of: in a cave, by the water, inside a wagon.",
        "answer": "in a cave",
        "session_id": ["D28"],
        "clue": ["D28:1"],
    },
    {
        "point": [["X2", "X3"], ["Y2"]],
        "question": "On the crowd-and-cave page, after the temporary ruler is pelted outside, is he later shown working inside a cave on that same page? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
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
        "point": [["X2", "X3"], ["Y3"]],
        "question": "Across the crowd-throwing page and the next chores page, is the temporary ruler shown doing cave work before the crowd throws things at him? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "no",
        "session_id": ["D28", "D29"],
        "clue": ["D28:1", "D29:1"],
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
        "point": [["X2", "X3"], ["Y2"]],
        "question": "On the ulcer dinner page, after the fight outside, does the temporary ruler later get hit at the dinner table? Reply with only yes or no in lowercase, with no punctuation.",
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
        "question": "On the apology-and-crown-return page, is the crown handed back before Oop is shown as an ordinary citizen again? Reply with only yes or no in lowercase, with no punctuation.",
        "answer": "yes",
        "session_id": ["D33"],
        "clue": ["D33:1"],
    },
]

SESSION_ORDER: List[str] = [
    "D1",
    "C1",
    "D2",
    "C2",
    "D3",
    "C3",
    "D4",
    "C4",
    "D27",
    "C5",
    "D28",
    "D30",
    "C6",
    "D29",
    "D31",
    "C7",
    "D32",
    "C8",
    "D33",
]


def load_source() -> Dict[str, Any]:
    with SOURCE_DIALOG.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_text_session(session_id: str, date: str, dialogue_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    dialogues: List[Dict[str, Any]] = []
    for idx, (user, assistant) in enumerate(dialogue_pairs, start=1):
        dialogues.append(
            {
                "round": f"{session_id}:{idx}",
                "user": user,
                "assistant": assistant,
            }
        )
    return {"session_id": session_id, "date": date, "dialogues": dialogues}


def build_image_session(session_id: str, date: str, page: int, user: str, assistant: str) -> Dict[str, Any]:
    image_name = f"Alley_Oop_Page_{page}.jpg"
    return {
        "session_id": session_id,
        "date": date,
        "dialogues": [
            {
                "round": f"{session_id}:1",
                "user": user,
                "assistant": assistant,
                "image_id": [f"{session_id}:IMG_001"],
                "input_image": [f"ComicScene_Alley_Oop_Draft/{image_name}"],
                "image_caption": [f"Alley Oop comic page {page}."],
            }
        ],
    }


def add_recap_round(session: Dict[str, Any], recap_user: str, recap_assistant: str) -> Dict[str, Any]:
    updated = copy.deepcopy(session)
    next_idx = len(updated["dialogues"]) + 1
    updated["dialogues"].append(
        {
            "round": f"{updated['session_id']}:{next_idx}",
            "user": recap_user,
            "assistant": recap_assistant,
        }
    )
    return updated


def build_dataset() -> Dict[str, Any]:
    source = load_source()
    source_sessions = {
        sess["session_id"]: copy.deepcopy(sess)
        for sess in source.get("multi_session_dialogues", [])
        if sess.get("session_id") in ANCHOR_SESSION_IDS
    }
    sessions: Dict[str, Dict[str, Any]] = {}
    sessions.update(source_sessions)
    for sid, session in list(sessions.items()):
        if sid in IMAGE_RECAPS:
            recap_user, recap_assistant = IMAGE_RECAPS[sid]
            sessions[sid] = add_recap_round(session, recap_user, recap_assistant)
    for item in NEW_IMAGE_SESSIONS:
        session = build_image_session(
            session_id=item["session_id"],
            date=item["date"],
            page=item["page"],
            user=item["user"],
            assistant=item["assistant"],
        )
        recap_user, recap_assistant = IMAGE_RECAPS[item["session_id"]]
        sessions[item["session_id"]] = add_recap_round(session, recap_user, recap_assistant)
    for item in TEXT_CONFUSION_SESSIONS:
        sessions[item["session_id"]] = build_text_session(
            session_id=item["session_id"],
            date=item["date"],
            dialogue_pairs=item["dialogues"],
        )

    ordered_sessions = [sessions[sid] for sid in SESSION_ORDER]
    result = {
        "character_profile": copy.deepcopy(source.get("character_profile", {})),
        "multi_session_dialogues": ordered_sessions,
        "human-annotated QAs": copy.deepcopy(QA_ITEMS),
    }
    return result


def main() -> None:
    data = build_dataset()
    OUTPUT_DIALOG.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DIALOG.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_sessions = len(data["multi_session_dialogues"])
    total_rounds = sum(len(sess["dialogues"]) for sess in data["multi_session_dialogues"])
    total_qas = len(data["human-annotated QAs"])
    print(f"[INFO] Wrote {OUTPUT_DIALOG}")
    print(f"[INFO] sessions={total_sessions} rounds={total_rounds} qas={total_qas}")


if __name__ == "__main__":
    main()
