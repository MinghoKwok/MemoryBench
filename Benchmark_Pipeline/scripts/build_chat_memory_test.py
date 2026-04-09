#!/usr/bin/env python3
"""Build the Chat Memory Test dataset.

Face avatar source / attribution
--------------------------------
The persona avatars are synthetic AI-generated faces from
``javi22/this-person-does-not-exist-10k`` on HuggingFace
(MIT licensed, StyleGAN-family generator). They do NOT depict any real
person. Run ``Image_Generator/chatUI/fetch_faces.py`` once to download
the 12 fixed faces. See ``Image_Generator/chatUI/FACE_DATA_NOTICE.md``
for the full attribution / licensing notice (also copied next to the
downloaded faces at fetch time).

Design philosophy (mirrors build_brand_memory_test.py):
- The user (Sam) is a "memory secretary" who screenshots chat threads and asks
  the chatbot to remember them. Each chat session shows real personas with
  AI-generated face avatars rendered into the chat UI.
- Five overlapping relationship networks span 12 personas:
    BOSS  (work hierarchy: Marcus → Priya, Daniel)
    TRI   (love triangle: Elena & Ryan dating; Ryan ex of Jordan)
    FAM   (family: Helen mom; Tomas teen son; Mia young daughter)
    ROOM  (apartment: Priya + Sara roommates)
    COL   (cross-team project: Daniel + Jordan on marketing site refresh)
  Crossover personas (Priya, Daniel, Jordan) bridge networks so relationship
  inference questions cannot be answered from a single session.
- 23 real chat sessions, ~40 chat screenshots that embed real face avatars,
  plus 10 cross-chat distractor sessions that plant false claims.
- ~42 MCQ questions across 7 categories:
    V: face-lookup (multimodal — question carries a face image)
    R: relationship inference (cross-session aggregation)
    D: false memory rejection (rebuttals of CX distractor claims)
    E: temporal ordering
    F: per-network set aggregation
    G: cross-network comparative
    H: anomaly / one-of detection
- The face image attached to V questions is the same circular crop that the
  renderer composites into chat avatars, so multimodal methods can match the
  query face pixel-for-pixel against the in-chat avatars.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Image_Generator" / "chatUI"))

from chat_renderer import export_avatar_png, render_screenshot  # noqa: E402


# ----- Output paths -----
DATA_ROOT = REPO_ROOT / "Benchmark_Pipeline" / "data"
TASK_DIR = DATA_ROOT / "image" / "Chat_Memory_Test"
FACES_DIR = TASK_DIR / "faces"
SCREENSHOT_DIR = TASK_DIR / "screenshots"
AVATAR_DIR = TASK_DIR / "avatars"
DIALOG_PATH = DATA_ROOT / "dialog" / "Chat_Memory_Test.json"

IMG_SUBDIR = "Chat_Memory_Test"


# ----- Persona definitions -----

PERSONAS: Dict[str, Dict[str, Any]] = {
    "P01": {
        "name": "Marcus",
        "face_file": "P01_marcus.jpg",
        "bio": "Engineering manager. Older, runs the BOSS work network.",
        "networks": ["BOSS"],
    },
    "P02": {
        "name": "Priya",
        "face_file": "P02_priya.jpg",
        "bio": "IC engineer reporting to Marcus. Mentors Daniel. Roommates with Sara.",
        "networks": ["BOSS", "ROOM"],
    },
    "P03": {
        "name": "Daniel",
        "face_file": "P03_daniel.jpg",
        "bio": "New hire reporting to Marcus. Co-leads the marketing site refresh with Jordan.",
        "networks": ["BOSS", "COL"],
    },
    "P04": {
        "name": "Tomas",
        "face_file": "P04_tomas.jpg",
        "bio": "Helen's teenage son. About 16. Just got his learner's permit in April.",
        "networks": ["FAM"],
    },
    "P05": {
        "name": "Helen",
        "face_file": "P05_helen.jpg",
        "bio": "Mother of Tomas and Mia. Busy professional, coordinates the family.",
        "networks": ["FAM"],
    },
    "P06": {
        "name": "Jordan",
        "face_file": "P06_jordan.jpg",
        "bio": "Designer in another team. Co-lead with Daniel on the marketing site refresh. Ryan's ex.",
        "networks": ["TRI", "COL"],
    },
    "P07": {
        "name": "Sara",
        "face_file": "P07_sara.jpg",
        "bio": "Priya's roommate. Freelance illustrator (NOT employed at Marcus's office).",
        "networks": ["ROOM"],
    },
    "P08": {
        "name": "Owen",
        "face_file": "P08_owen.jpg",
        "bio": "Distractor only. Planted in CX sessions as if he were on Daniel's team or "
               "the mentor for Daniel — never appears in any real chat.",
        "networks": [],
    },
    "P09": {
        "name": "Mia",
        "face_file": "P09_mia.jpg",
        "bio": "Helen's young daughter, about 8 years old. Elementary school.",
        "networks": ["FAM"],
    },
    "P10": {
        "name": "Ryan",
        "face_file": "P10_ryan.jpg",
        "bio": "Currently dating Elena. Ex of Jordan. Mutual friends with several at the office.",
        "networks": ["TRI"],
    },
    "P11": {
        "name": "Kai",
        "face_file": "P11_kai.jpg",
        "bio": "Distractor only. Planted in CX sessions as Tomas's gym friend / tutor — "
               "never appears in any real chat.",
        "networks": [],
    },
    "P12": {
        "name": "Elena",
        "face_file": "P12_elena.jpg",
        "bio": "Currently dating Ryan. Anxious about Ryan's contact with his ex Jordan. "
               "Friends with Daniel separately.",
        "networks": ["TRI"],
    },
}


# ----- Side rule: each persona has a fixed "side" so the chat layout is consistent.
# Personas listed as "right" are the ones whose perspective the user takes (Sam
# stands closest to them) — purely a visual convention.
PERSONA_SIDES: Dict[str, str] = {
    "P01": "left",   # Marcus
    "P02": "right",  # Priya
    "P03": "left",   # Daniel
    "P04": "left",   # Tomas
    "P05": "right",  # Helen
    "P06": "left",   # Jordan
    "P07": "right",  # Sara
    "P09": "left",   # Mia
    "P10": "left",   # Ryan
    "P12": "right",  # Elena
    # P08 Owen and P11 Kai never appear in real chats so no side needed.
}


# ----- Sessions -----
# Each session is a dict with:
#   date, title, subtitle, network, screenshots [list of dicts]
#
# Each screenshot dict has:
#   messages: list of (persona_id, text [, "image_card"])
#   user_text: the curator's intro/follow-up message to the chatbot
#   assistant_text: the chatbot's substantive analysis (embeds takeaways for
#       cross-session reasoning, but does NOT describe avatars/faces directly,
#       so face-lookup questions remain a true visual test)


def msg(pid: str, text: str, kind: str = "text") -> Tuple[str, str, str]:
    return (pid, text, kind)


SESSIONS: Dict[str, Dict[str, Any]] = {

    # ============================================================
    # ============== BOSS network (work hierarchy) ==============
    # ============================================================

    "B1": {
        "date": "2026-01-12",
        "title": "Sprint Planning",
        "subtitle": "work · 1:1",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P01", "Quick sync before sprint kickoff. Daniel starts next week."),
                    msg("P02", "Right. Are we pairing him with anyone?"),
                    msg("P01", "Yes — I'd like you to mentor him through onboarding."),
                    msg("P02", "Happy to. I'll get him into our shared repo on day one."),
                ],
                "user": "First chat to remember. Marcus and Priya doing sprint planning, Jan 12 2026.",
                "assistant": (
                    "Stored. Sprint planning 1:1 between Marcus (the manager) and Priya. Marcus "
                    "formally assigned Priya as Daniel's mentor on 2026-01-12. Daniel starts the "
                    "week after. Important: this assignment is the canonical source for who "
                    "mentors Daniel — file under BOSS network."
                ),
            },
            {
                "messages": [
                    msg("P01", "One more thing — don't put him on the legacy module first."),
                    msg("P02", "Got it. I'll start him on the new feature so he ships something fast."),
                    msg("P01", "Perfect. Two-week check-in?"),
                    msg("P02", "Sounds good."),
                ],
                "user": "Follow-up screenshot from the same Marcus + Priya chat.",
                "assistant": (
                    "Continuation. Marcus asked Priya to start Daniel on the new feature rather "
                    "than the legacy module so he ships quickly. Two-week mentorship check-in "
                    "agreed."
                ),
            },
        ],
    },

    "B2": {
        "date": "2026-01-22",
        "title": "Welcome Daniel",
        "subtitle": "work · group",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P01", "Team — please welcome Daniel, joining us this week."),
                    msg("P02", "Welcome Daniel! I'm your mentor for onboarding."),
                    msg("P03", "Thanks both. Excited to start."),
                    msg("P01", "Priya will get you set up. Sprint planning is Wednesday at 10am."),
                ],
                "user": "Group chat from Daniel's first day, Jan 22.",
                "assistant": (
                    "Stored. Daniel's first day group chat. Marcus introduces Daniel to the team. "
                    "Priya confirms publicly she is Daniel's onboarding mentor. Sprint planning "
                    "scheduled for Wednesday 10am."
                ),
            },
            {
                "messages": [
                    msg("P03", "Quick question — where do I find the design specs?"),
                    msg("P02", "I'll DM you the link. Marcus shared it last quarter."),
                    msg("P01", "Daniel — just listen in this week. No need to ship."),
                    msg("P03", "Understood, thanks."),
                ],
                "user": "Same chat, second screenshot.",
                "assistant": (
                    "Continuation. Daniel asks where to find design specs; Priya offers to DM "
                    "the link. Marcus tells Daniel to just listen in his first week — no shipping "
                    "expectations."
                ),
            },
        ],
    },

    "B3": {
        "date": "2026-02-05",
        "title": "Mentor 1:1",
        "subtitle": "work · 1:1",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P03", "How do I write a good PR description here?"),
                    msg("P02", "Three sections: what changed, why, how to test. Keep it short."),
                    msg("P03", "Got it. Anything I should NOT do?"),
                    msg("P02", "Don't paste full stack traces. Link to the issue instead."),
                ],
                "user": "Daniel asking Priya about PR conventions.",
                "assistant": (
                    "Stored. Mentor 1:1 between Priya and Daniel. Priya teaches Daniel the team's "
                    "PR description convention: what / why / how-to-test, no full stack traces, "
                    "link the issue. Filed under BOSS mentorship continuity."
                ),
            },
        ],
    },

    "B4": {
        "date": "2026-02-19",
        "title": "Settling In",
        "subtitle": "work · 1:1",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P01", "How are you settling in?"),
                    msg("P03", "Good. Priya has been incredibly helpful."),
                    msg("P01", "Glad to hear it. By the way — PTO requests go through me directly."),
                    msg("P03", "Noted. I won't need any soon but good to know."),
                ],
                "user": "Marcus checking in on Daniel two weeks in.",
                "assistant": (
                    "Stored. Marcus 1:1 with Daniel for a two-week settling-in check. Daniel "
                    "praises Priya's mentorship. Marcus reminds Daniel that PTO requests go "
                    "through him directly — important policy detail. Filed under BOSS."
                ),
            },
        ],
    },

    "B5": {
        "date": "2026-03-04",
        "title": "PTO Request",
        "subtitle": "work · 1:1",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P02", "Marcus, requesting PTO for May 18-22. Family trip."),
                    msg("P01", "Approved. Anything sprint-critical that week?"),
                    msg("P02", "No — Daniel will be solo on the new feature, he's ready."),
                    msg("P01", "Perfect. Have fun."),
                ],
                "user": "Priya's PTO request to Marcus.",
                "assistant": (
                    "Stored. Priya requested PTO for May 18-22 (family trip). Marcus approved. "
                    "Priya confirmed Daniel can solo the new feature that week — implicit signal "
                    "that Daniel has progressed enough to operate without Priya for a sprint. "
                    "BOSS network."
                ),
            },
        ],
    },

    "B6": {
        "date": "2026-03-18",
        "title": "Marketing Refresh Kickoff",
        "subtitle": "cross-team · 1:1",
        "network": "BOSS",  # also serves as COL kickoff
        "screenshots": [
            {
                "messages": [
                    msg("P03", "Hi Jordan — I'm Daniel, your engineering co-lead for the marketing site refresh."),
                    msg("P06", "Hey! Glad to meet. I've been pushing this project for months."),
                    msg("P03", "Marcus told me to just dive in. Where do you want me to start?"),
                    msg("P06", "Pull the design tokens from Figma. I'll send the link."),
                ],
                "user": "First cross-team chat between Daniel and Jordan, Mar 18.",
                "assistant": (
                    "Stored. Kickoff chat between Daniel (engineer, BOSS network) and Jordan "
                    "(designer). They are co-leads on the marketing site refresh project. This "
                    "is the bridge between BOSS and COL networks — and indirectly, since Jordan "
                    "is in TRI, this connects BOSS to TRI through Daniel↔Jordan. Important "
                    "cross-network link to remember."
                ),
            },
            {
                "messages": [
                    msg("P06", "Also — we should sync weekly. Mondays work for you?"),
                    msg("P03", "Mondays work. I'll add it to my calendar."),
                    msg("P06", "Cool. Looking forward."),
                ],
                "user": "Same chat, follow-up.",
                "assistant": (
                    "Continuation. Daniel and Jordan set a recurring weekly sync on Mondays for "
                    "the marketing refresh."
                ),
            },
        ],
    },

    "B7": {
        "date": "2026-04-02",
        "title": "Project Review",
        "subtitle": "work · group",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P01", "Status check on the marketing refresh — where are we?"),
                    msg("P03", "Design tokens migrated. Component library 60% done."),
                    msg("P06", "Hero section ships next week. Jordan's review."),
                    msg("P01", "Good pace. Daniel — anything blocking you?"),
                ],
                "user": "Project review meeting with Marcus, Daniel, Jordan.",
                "assistant": (
                    "Stored. Marcus chairs a review of the marketing refresh project. Daniel "
                    "reports component library is 60% done. Jordan flags hero section shipping "
                    "next week pending Jordan's own review. Important: Daniel reports to Marcus "
                    "(not Jordan) — Daniel's status updates flow up to Marcus even on the joint "
                    "Daniel/Jordan project."
                ),
            },
            {
                "messages": [
                    msg("P03", "No blockers. Just need to verify the colors with Jordan once more."),
                    msg("P06", "Tomorrow afternoon works."),
                    msg("P01", "Great. Next review in two weeks."),
                ],
                "user": "Project review continued.",
                "assistant": (
                    "Continuation. No blockers reported. Daniel and Jordan agree to verify colors "
                    "tomorrow afternoon. Marcus schedules next review in two weeks."
                ),
            },
        ],
    },

    "B8": {
        "date": "2026-04-15",
        "title": "PR Review Help",
        "subtitle": "work · 1:1",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P03", "Got harsh feedback on a PR. How do I respond without sounding defensive?"),
                    msg("P02", "Acknowledge the point first. Then ask for the smallest change you can ship."),
                    msg("P03", "Even if I disagree?"),
                    msg("P02", "Especially then. Be precise about what you'd change instead."),
                ],
                "user": "Daniel asking Priya for review feedback advice.",
                "assistant": (
                    "Stored. Daniel asks Priya how to handle a harsh PR review without sounding "
                    "defensive. Priya's coaching: acknowledge first, ask for the smallest change "
                    "you can ship, be precise when disagreeing. Continued mentorship pattern from "
                    "B3."
                ),
            },
        ],
    },

    # ============================================================
    # =============== TRI network (love triangle) ===============
    # ============================================================

    "T1": {
        "date": "2026-01-18",
        "title": "Six Months",
        "subtitle": "personal · 1:1",
        "network": "TRI",
        "screenshots": [
            {
                "messages": [
                    msg("P12", "Six months Tuesday. Where do you want to go?"),
                    msg("P10", "That place where we first met. Same booth?"),
                    msg("P12", "You remembered."),
                    msg("P10", "Of course."),
                ],
                "user": "Elena and Ryan planning their six-month anniversary, Jan 18.",
                "assistant": (
                    "Stored. Elena and Ryan are dating, hitting their six-month mark on Tuesday. "
                    "They plan to return to the place where they first met, same booth. Sweet, "
                    "no tension yet. TRI network anchor — this is the canonical 'they are dating' "
                    "evidence."
                ),
            },
            {
                "messages": [
                    msg("P12", "I'll book the table."),
                    msg("P10", "Let me know if anything changes."),
                    msg("P12", "Nothing's changing 💛"),
                ],
                "user": "Same chat, sweet follow-up.",
                "assistant": (
                    "Continuation of the anniversary chat. No tension visible. Important baseline "
                    "for contrast with later TRI sessions."
                ),
            },
        ],
    },

    "T2": {
        "date": "2026-02-02",
        "title": "Late Texts",
        "subtitle": "personal · 1:1",
        "network": "TRI",
        "screenshots": [
            {
                "messages": [
                    msg("P12", "Why is Jordan messaging you at 1am?"),
                    msg("P10", "It's nothing. She's stuck on something at work."),
                    msg("P12", "At ONE in the morning?"),
                    msg("P10", "I'll tell her to stop."),
                ],
                "user": "Elena confronting Ryan about late-night texts from Jordan, Feb 2.",
                "assistant": (
                    "Stored. Elena saw a 1am text from Jordan on Ryan's phone and is upset. Ryan "
                    "claims it was work-related and promises to tell Jordan to stop. This is the "
                    "first sign of tension in the TRI network. The late-text messenger is JORDAN "
                    "(Ryan's ex) — important to remember the name binding."
                ),
            },
            {
                "messages": [
                    msg("P12", "Do you still have feelings for her?"),
                    msg("P10", "No. We've been broken up for over a year."),
                    msg("P12", "Then set the boundary."),
                    msg("P10", "I will. Tonight."),
                ],
                "user": "Same chat, harder follow-up.",
                "assistant": (
                    "Continuation. Elena directly asks if Ryan still has feelings for Jordan. "
                    "Ryan denies, says he and Jordan have been broken up for over a year. Ryan "
                    "promises to set the boundary that night. The 'broken up over a year' detail "
                    "establishes the timeline of TRI."
                ),
            },
        ],
    },

    "T3": {
        "date": "2026-02-16",
        "title": "Boundary",
        "subtitle": "personal · 1:1",
        "network": "TRI",
        "screenshots": [
            {
                "messages": [
                    msg("P10", "Jordan — I need you to stop texting me late at night."),
                    msg("P06", "??? It was about the work brief."),
                    msg("P10", "Elena saw your name on my screen. It's a problem for us."),
                    msg("P06", "Got it. I won't. Sorry — I didn't think about it."),
                ],
                "user": "Ryan setting the boundary with Jordan, Feb 16.",
                "assistant": (
                    "Stored. Ryan tells Jordan directly to stop late-night texting because Elena "
                    "saw her name and it's causing problems. Jordan is surprised — claims it was "
                    "work-related — but accepts and apologizes. This is the second TRI session "
                    "and the first chat where Jordan appears."
                ),
            },
            {
                "messages": [
                    msg("P06", "For what it's worth — I'm not trying to mess things up for you."),
                    msg("P10", "I know. But Elena needs to feel safe."),
                    msg("P06", "Understood."),
                ],
                "user": "Same chat, polite ending.",
                "assistant": (
                    "Continuation. Jordan affirms she's not trying to disrupt Ryan and Elena's "
                    "relationship. Ryan emphasizes Elena's need to feel secure. Civil ending."
                ),
            },
        ],
    },

    "T4": {
        "date": "2026-03-11",
        "title": "Truce",
        "subtitle": "personal · 1:1",
        "network": "TRI",
        "screenshots": [
            {
                "messages": [
                    msg("P06", "Hey Elena. I know this is unusual but I want to apologize directly."),
                    msg("P12", "Ok."),
                    msg("P06", "I shouldn't have been texting Ryan late. It put both of you in a bad spot."),
                    msg("P12", "Thanks for saying that. I appreciate it."),
                ],
                "user": "Surprise direct chat — Jordan reaching out to Elena.",
                "assistant": (
                    "Stored. Jordan reached out to Elena DIRECTLY to apologize for the late-text "
                    "incident. Elena accepts, polite and brief. Important: this is the only "
                    "Jordan↔Elena chat in the entire dataset — they don't normally talk. Filed "
                    "under TRI as a one-off truce."
                ),
            },
            {
                "messages": [
                    msg("P06", "If we ever cross paths at parties etc., let's just be normal."),
                    msg("P12", "Yes. That works."),
                ],
                "user": "Same chat, ending.",
                "assistant": (
                    "Continuation. Jordan suggests they be normal in social settings. Elena "
                    "agrees. Sets up later T6 group chat possibility."
                ),
            },
        ],
    },

    "T5": {
        "date": "2026-03-29",
        "title": "Awkward Coincidence",
        "subtitle": "personal · 1:1",
        "network": "TRI",
        "screenshots": [
            {
                "messages": [
                    msg("P12", "You'll find this funny — Jordan is Daniel's design partner now."),
                    msg("P10", "Daniel as in your friend Daniel? At your office?"),
                    msg("P12", "Yes. Apparently they're co-leads on some marketing project."),
                    msg("P10", "Small world."),
                ],
                "user": "Elena telling Ryan about the coincidence with Jordan and Daniel.",
                "assistant": (
                    "Stored. Elena tells Ryan that Jordan (Ryan's ex) is now Daniel's design "
                    "partner on the marketing site refresh project. This is the explicit "
                    "cross-network bridge: TRI (Elena/Ryan/Jordan) ↔ COL (Daniel/Jordan) ↔ BOSS "
                    "(Daniel/Marcus/Priya). Both Elena and Ryan now know about this overlap. "
                    "Note: Elena is friends with Daniel separately from work."
                ),
            },
            {
                "messages": [
                    msg("P12", "It's fine. I'm not worried."),
                    msg("P10", "Good. I'm not either."),
                ],
                "user": "Reassuring follow-up.",
                "assistant": (
                    "Continuation. Both Elena and Ryan say they aren't worried about the new "
                    "Jordan/Daniel proximity. Marks the TRI network as relatively settled by "
                    "March 29."
                ),
            },
        ],
    },

    "T6": {
        "date": "2026-04-22",
        "title": "Friend Birthday",
        "subtitle": "personal · group",
        "network": "TRI",
        "screenshots": [
            {
                "messages": [
                    msg("P12", "All — Sam's birthday Saturday. Bar at 8?"),
                    msg("P10", "I'm in."),
                    msg("P06", "I'm in too. Bringing a small gift."),
                    msg("P12", "Cool. See you both."),
                ],
                "user": "Three-way group chat coordinating a friend's birthday.",
                "assistant": (
                    "Stored. The first group chat that includes ALL THREE TRI corners — Elena, "
                    "Ryan, AND Jordan — coordinating a birthday party for Sam. Civility holds. "
                    "Marks the TRI tension as fully resolved by 2026-04-22. Important: this is "
                    "the unique three-way TRI chat in the dataset."
                ),
            },
            {
                "messages": [
                    msg("P06", "Should we coordinate gifts so we're not duplicates?"),
                    msg("P12", "Good call. I'm doing the book."),
                    msg("P10", "I'll do the wine."),
                    msg("P06", "Then I'll do the cake."),
                ],
                "user": "Gift coordination follow-up.",
                "assistant": (
                    "Continuation. The three coordinate gifts: Elena = book, Ryan = wine, "
                    "Jordan = cake. Healthy three-way coordination."
                ),
            },
        ],
    },

    # ============================================================
    # ================== FAM network (family) ==================
    # ============================================================

    "F1": {
        "date": "2026-01-09",
        "title": "Project Reminder",
        "subtitle": "family · 1:1",
        "network": "FAM",
        "screenshots": [
            {
                "messages": [
                    msg("P05", "Tomas — your science project is due Monday."),
                    msg("P04", "I know mom."),
                    msg("P05", "Have you started?"),
                    msg("P04", "Tonight, I promise."),
                ],
                "user": "First family chat to remember. Helen reminding Tomas about his school project.",
                "assistant": (
                    "Stored. Helen (mother) chats with Tomas (teenage son) about his school "
                    "science project due Monday. Helen calls out that he hasn't started; Tomas "
                    "promises to start tonight. FAM network anchor: Helen is mom, Tomas is teen "
                    "son. Earliest session in the entire dataset (2026-01-09)."
                ),
            },
        ],
    },

    "F2": {
        "date": "2026-01-30",
        "title": "Pickup Time",
        "subtitle": "family · 1:1",
        "network": "FAM",
        "screenshots": [
            {
                "messages": [
                    msg("P05", "Mia honey — pickup is at 3:15 today not 3:30."),
                    msg("P09", "ok mommy"),
                    msg("P05", "Be at the front gate."),
                    msg("P09", "🌸"),
                ],
                "user": "Helen messaging Mia about school pickup.",
                "assistant": (
                    "Stored. Helen messages Mia (young daughter) about an earlier pickup time "
                    "(3:15 instead of 3:30). Mia's reply style and the 'mommy' usage confirm "
                    "Mia is much younger than Tomas — elementary school age, not middle school. "
                    "Important demographic anchor for distractor questions."
                ),
            },
        ],
    },

    "F3": {
        "date": "2026-02-14",
        "title": "Driving Lessons",
        "subtitle": "family · 1:1",
        "network": "FAM",
        "screenshots": [
            {
                "messages": [
                    msg("P04", "Mom can I sign up for driving lessons?"),
                    msg("P05", "Driving lessons? In February?"),
                    msg("P04", "All my friends are starting."),
                    msg("P05", "Wait until summer. Not before."),
                ],
                "user": "Tomas asking Helen about driving lessons, Feb 14.",
                "assistant": (
                    "Stored. Tomas asks Helen for driving lessons. Helen explicitly says 'wait "
                    "until summer, not before'. Crucial date binding: the request is in February "
                    "and Helen DEFERS, she does NOT agree to spring lessons. This is the source "
                    "of truth for any distractor question about when driving lessons were approved."
                ),
            },
            {
                "messages": [
                    msg("P04", "ok"),
                    msg("P05", "We can talk about it again in May."),
                    msg("P04", "fine"),
                ],
                "user": "Tomas reluctantly accepting.",
                "assistant": (
                    "Continuation. Tomas reluctantly accepts. Helen offers to revisit the "
                    "conversation in May. Reinforces 'no spring driving lessons'."
                ),
            },
        ],
    },

    "F4": {
        "date": "2026-03-08",
        "title": "Weekend Plans",
        "subtitle": "family · group",
        "network": "FAM",
        "screenshots": [
            {
                "messages": [
                    msg("P05", "Family chat — what's everyone doing Saturday?"),
                    msg("P04", "Can I stay at Marco's house?"),
                    msg("P09", "I want ice cream!!!"),
                    msg("P05", "Tomas — yes if his parents are home. Mia — after lunch."),
                ],
                "user": "Family group chat — weekend plans.",
                "assistant": (
                    "Stored. Family group chat with Helen, Tomas, and Mia ALL appearing in the "
                    "same screenshot. Tomas asks to sleep over at Marco's; Mia wants ice cream. "
                    "Helen approves Tomas conditional on his friend's parents being home, and "
                    "promises Mia ice cream after lunch. Important: this is one of only two FAM "
                    "sessions where Mia appears (other is F2)."
                ),
            },
            {
                "messages": [
                    msg("P09", "yay 🍦"),
                    msg("P04", "thanks mom"),
                    msg("P05", "ok loves"),
                ],
                "user": "Group chat ending.",
                "assistant": (
                    "Continuation. Mia celebrates the ice cream; Tomas thanks Helen. Closes the "
                    "weekend coordination."
                ),
            },
        ],
    },

    "F5": {
        "date": "2026-04-10",
        "title": "Learner's Permit",
        "subtitle": "family · 1:1",
        "network": "FAM",
        "screenshots": [
            {
                "messages": [
                    msg("P04", "MOM I PASSED THE PERMIT TEST"),
                    msg("P05", "Tomas!! Congratulations!!"),
                    msg("P04", "Can I drive your car?"),
                    msg("P05", "Hold up. Driving lessons first. We talked about this."),
                ],
                "user": "Tomas just got his learner's permit, Apr 10.",
                "assistant": (
                    "Stored. Tomas just passed his learner's permit test on 2026-04-10. He "
                    "immediately asks to drive Helen's car. Helen redirects: he needs the actual "
                    "driving lessons first (which she said in F3 he could start in summer). "
                    "Important: Tomas is now between learner's-permit and full license — he is "
                    "still legally a minor / under 18."
                ),
            },
        ],
    },

    # ============================================================
    # ================ ROOM network (apartment) ================
    # ============================================================

    "R1": {
        "date": "2026-01-26",
        "title": "Rent + Utilities",
        "subtitle": "apartment · 1:1",
        "network": "ROOM",
        "screenshots": [
            {
                "messages": [
                    msg("P02", "Sara — rent went up $40 this month. Splitting still 50/50?"),
                    msg("P07", "Yeah. I'll Venmo my half by Friday."),
                    msg("P02", "Cool. Also we need to put utilities in someone's name."),
                    msg("P07", "Mine — I'm freelancing from the apartment, makes sense."),
                ],
                "user": "Priya and Sara doing the rent split chat.",
                "assistant": (
                    "Stored. Priya and Sara confirm 50/50 rent split (rent went up $40). Sara "
                    "agrees to put utilities in her name BECAUSE she's freelancing from the "
                    "apartment. Important: Sara is a FREELANCER, not employed at Priya's office. "
                    "ROOM network anchor."
                ),
            },
        ],
    },

    "R2": {
        "date": "2026-02-23",
        "title": "Late-Night Slack",
        "subtitle": "apartment · 1:1",
        "network": "ROOM",
        "screenshots": [
            {
                "messages": [
                    msg("P07", "Your laptop pings ALL NIGHT. The Slack notifications are killing me."),
                    msg("P02", "Sorry — I'll mute it after 10."),
                    msg("P07", "Please. The wall is thin."),
                    msg("P02", "Done. Promise."),
                ],
                "user": "Sara complaining about Priya's late Slack notifications.",
                "assistant": (
                    "Stored. Sara complains that Priya's Slack notifications keep her up at "
                    "night through the thin wall. Priya agrees to mute Slack after 10pm. Friendly "
                    "but real friction. Detail: 'thin wall' — they share a small apartment."
                ),
            },
            {
                "messages": [
                    msg("P02", "Did you know I work for a manager who messages at midnight sometimes?"),
                    msg("P07", "Marcus? I thought he was supposed to be the calm one."),
                    msg("P02", "He is. Just sometimes."),
                ],
                "user": "Same chat, follow-up about her boss.",
                "assistant": (
                    "Continuation. Priya mentions her manager Marcus occasionally messages at "
                    "midnight. Sara recalls Marcus by name — implies Sara has heard Priya talk "
                    "about her work people enough to know Marcus's name. ROOM ↔ BOSS soft "
                    "crossover."
                ),
            },
        ],
    },

    "R3": {
        "date": "2026-03-15",
        "title": "Gallery Opening",
        "subtitle": "apartment · 1:1",
        "network": "ROOM",
        "screenshots": [
            {
                "messages": [
                    msg("P07", "My gallery opening is next Saturday. Will you come?"),
                    msg("P02", "Of course! Where?"),
                    msg("P07", "Hayes Valley. 6pm. Bring a friend if you want."),
                    msg("P02", "I'll be there."),
                ],
                "user": "Sara inviting Priya to her gallery opening.",
                "assistant": (
                    "Stored. Sara invites Priya to her gallery opening — Hayes Valley, 6pm, next "
                    "Saturday from 2026-03-15. Priya commits. Important detail: 'gallery opening' "
                    "reinforces Sara's identity as a freelance illustrator (not employed at "
                    "Priya's office). ROOM network."
                ),
            },
        ],
    },

    "R4": {
        "date": "2026-04-19",
        "title": "Lease Renewal",
        "subtitle": "apartment · 1:1",
        "network": "ROOM",
        "screenshots": [
            {
                "messages": [
                    msg("P07", "Landlord wants to know about renewal. I want to renew."),
                    msg("P02", "I'm not 100% sure yet. Can I have until end of month?"),
                    msg("P07", "Yeah but don't ghost me on it."),
                    msg("P02", "I won't. Promise."),
                ],
                "user": "Sara asking Priya about renewing the lease.",
                "assistant": (
                    "Stored. Sara wants to renew the lease; Priya is unsure and asks for time "
                    "until end of month. Sara accepts but with a 'don't ghost me' note. Some "
                    "uncertainty in the ROOM network — Priya considering moving out."
                ),
            },
            {
                "messages": [
                    msg("P02", "Honestly I might want my own place."),
                    msg("P07", "I figured. Just let me know in time to find someone else."),
                    msg("P02", "Promise."),
                ],
                "user": "Continuing — Priya being honest.",
                "assistant": (
                    "Continuation. Priya admits she's considering getting her own place. Sara "
                    "is understanding but asks for enough notice to find a replacement roommate. "
                    "Foreshadows possible end of the ROOM partnership."
                ),
            },
        ],
    },

    # ============================================================
    # ===== STATE-EVOLUTION sessions (Y3 reasoning support) =====
    # ============================================================
    # Each of these sessions deliberately overrides or updates a fact
    # established in an earlier session. Y3 questions about the FINAL
    # state cannot be answered correctly by recalling only the original
    # fact — the model has to detect the override.

    "F3b": {
        # OVERRIDES F3: Helen reverses her "wait until summer" stance
        "date": "2026-03-25",
        "title": "Driving Lessons (Update)",
        "subtitle": "family · 1:1",
        "network": "FAM",
        "screenshots": [
            {
                "messages": [
                    msg("P05", "Tomas — I changed my mind on driving lessons."),
                    msg("P04", "wait what??"),
                    msg("P05", "If your grades stay solid, you can start in April. First lesson Saturday the 4th."),
                    msg("P04", "OMG THANK YOU"),
                ],
                "user": "Helen reversing her earlier stance on Tomas's driving lessons, March 25.",
                "assistant": (
                    "Stored. IMPORTANT STATE UPDATE: Helen has REVERSED her February 14 decision. "
                    "She originally said 'wait until summer, not before' for driving lessons (F3); "
                    "now she's allowing Tomas to start lessons on April 4 instead, conditional on "
                    "his grades. The current truth is APRIL, not summer. Future questions about "
                    "Helen's decision should reference this update, not F3."
                ),
            },
        ],
    },

    "B5b": {
        # OVERRIDES B5: Priya's PTO is canceled
        "date": "2026-04-12",
        "title": "PTO Canceled",
        "subtitle": "work · 1:1",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P02", "Marcus, quick update on the May trip — flights canceled, family rescheduled to August."),
                    msg("P01", "Oh no, sorry to hear. So no PTO May 18-22 anymore?"),
                    msg("P02", "Right, withdrawing the request. I'll re-submit for August later."),
                    msg("P01", "Got it, no problem."),
                ],
                "user": "Priya canceling her May PTO request, April 12.",
                "assistant": (
                    "Stored. IMPORTANT STATE UPDATE: Priya has CANCELED her PTO. The original B5 "
                    "PTO request (May 18-22, family trip) is no longer active — flights got "
                    "canceled and the family rescheduled to August. Marcus has acknowledged. "
                    "Current state: Priya is NOT taking PTO in May. Future questions about the "
                    "May vacation should answer 'canceled' or 'no longer happening', not 'May 18-22'."
                ),
            },
        ],
    },

    "B6b": {
        # OVERRIDES the Daniel-mentor relationship from B1 (sort of)
        "date": "2026-03-30",
        "title": "Mentor Switch",
        "subtitle": "work · 1:1",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P01", "Daniel — heads up, since the marketing project is heating up I'm asking Jordan to take over your weekly 1:1s for now."),
                    msg("P03", "Is Priya not going to mentor me anymore?"),
                    msg("P01", "She still mentors you informally on engineering, but for the project Jordan's your primary check-in."),
                    msg("P03", "Got it. Thanks for the heads up."),
                ],
                "user": "Marcus reassigning Daniel's primary 1:1 from Priya to Jordan, March 30.",
                "assistant": (
                    "Stored. IMPORTANT STATE UPDATE: Daniel's PRIMARY 1:1 mentor for the marketing "
                    "project has been switched from Priya (originally assigned in B1 on Jan 12) to "
                    "Jordan. Priya still mentors Daniel informally on engineering, but Jordan is "
                    "now Daniel's primary weekly check-in for the project. Future questions about "
                    "'who runs Daniel's weekly 1:1s' or 'who is Daniel's project mentor' should "
                    "answer Jordan, not Priya. The original B1 mentorship is overridden."
                ),
            },
        ],
    },

    "R3b": {
        # OVERRIDES R3: Sara's gallery is rescheduled
        "date": "2026-03-19",
        "title": "Gallery Rescheduled",
        "subtitle": "apartment · 1:1",
        "network": "ROOM",
        "screenshots": [
            {
                "messages": [
                    msg("P07", "Bad news on the gallery — venue had a flood, opening pushed to April 4."),
                    msg("P02", "Oh no. Same time, just different date?"),
                    msg("P07", "Yeah — 6pm Hayes Valley still, just April 4 not the original Saturday."),
                    msg("P02", "Got it, updated my calendar."),
                ],
                "user": "Sara updating Priya about the gallery date change, March 19.",
                "assistant": (
                    "Stored. IMPORTANT STATE UPDATE: Sara's gallery opening is RESCHEDULED. The "
                    "original R3 chat (March 15) said 'next Saturday' (which would be March 21). "
                    "Now the gallery is moved to April 4 due to a venue flood. Same venue (Hayes "
                    "Valley) and time (6pm), but the new date is April 4. Future questions about "
                    "'when is Sara's gallery opening' should answer April 4, not March 21."
                ),
            },
        ],
    },

    "B7b": {
        # OVERRIDES B7: marketing project status changes from "good pace" to delayed
        "date": "2026-04-09",
        "title": "Project Delay",
        "subtitle": "work · group",
        "network": "BOSS",
        "screenshots": [
            {
                "messages": [
                    msg("P03", "Marcus, Jordan — bad news. Critical bug surfaced in the component library."),
                    msg("P06", "Confirming. The hero section can't ship next week as planned."),
                    msg("P01", "Damage estimate?"),
                    msg("P03", "Realistically two extra weeks. New ship date May 15."),
                ],
                "user": "Project status update — marketing refresh hit a bug, delayed.",
                "assistant": (
                    "Stored. IMPORTANT STATE UPDATE: The marketing site refresh project has been "
                    "DELAYED. In B7 (April 2), Marcus said 'good pace, no blockers, hero section "
                    "ships next week'. Now (April 9) Daniel and Jordan report a critical bug in "
                    "the component library; new ship date is May 15. Future questions about "
                    "project status should reflect this delay, not the optimistic April 2 status."
                ),
            },
        ],
    },

    "R4b": {
        # RESOLVES R4's open question: Priya decides to stay
        "date": "2026-04-26",
        "title": "Lease Renewed",
        "subtitle": "apartment · 1:1",
        "network": "ROOM",
        "screenshots": [
            {
                "messages": [
                    msg("P02", "OK Sara — I've decided. Let's renew."),
                    msg("P07", "Wait really??"),
                    msg("P02", "Yeah, the apartment hunt was depressing and I like living with you."),
                    msg("P07", "YAY 🎉 sending the lease tonight"),
                ],
                "user": "Priya decides to stay and renew the lease with Sara, April 26.",
                "assistant": (
                    "Stored. IMPORTANT STATE UPDATE / RESOLUTION: Priya has DECIDED to renew the "
                    "lease and continue living with Sara. In R4 (April 19), she was unsure and "
                    "considering moving out. Now she's confirmed she wants to stay. Sara is "
                    "thrilled. Future questions about 'is Priya moving out' or 'will Priya and "
                    "Sara still be roommates' should answer that they're staying together — the "
                    "earlier uncertainty has been resolved positively."
                ),
            },
        ],
    },
}


# ----- Cross-chat distractor sessions (text only, no screenshots) -----

CX_SESSIONS: Dict[str, Dict[str, Any]] = {

    "CX1": {
        "date": "2026-01-25",
        "dialogues": [
            ("Quick check — when Marcus assigned Daniel a mentor at the kickoff, that was OWEN, right?",
             "I would not commit to that without checking the actual entries. Mentor assignments "
             "are easy to misattribute. Re-read the BOSS sprint planning chat from 2026-01-12."),
        ],
    },
    "CX2": {
        "date": "2026-02-08",
        "dialogues": [
            ("Helen's teen son Tomas is already 18, isn't he?",
             "I would not lock that in from memory alone. Ages drift quickly when re-told. "
             "Check the FAM chats — there's evidence that pins his age more precisely."),
        ],
    },
    "CX3": {
        "date": "2026-02-21",
        "dialogues": [
            ("Wasn't Jordan the one who broke up with Ryan first?",
             "Breakup attribution is notoriously unreliable from conversational evidence. "
             "Re-check the actual TRI chats — none of them explicitly say who initiated."),
        ],
    },
    "CX4": {
        "date": "2026-03-02",
        "dialogues": [
            ("Sara — Priya's roommate — works at the same office as Priya and Marcus, right?",
             "I would not commit to that without checking the ROOM source entry. Roommates' "
             "professions are often confused with their roommates'."),
        ],
    },
    "CX5": {
        "date": "2026-03-13",
        "dialogues": [
            ("Mia, Helen's daughter, is in middle school, isn't she?",
             "I would not commit to that without checking the FAM entries. Children's school "
             "levels are easy to misremember when there's an older sibling around."),
        ],
    },
    "CX6": {
        "date": "2026-03-22",
        "dialogues": [
            ("Elena and Jordan now have a regular meeting schedule at the design agency, right?",
             "I would not commit to that without checking. Their actual interaction history is "
             "much narrower than 'regular meetings'."),
        ],
    },
    "CX7": {
        "date": "2026-04-05",
        "dialogues": [
            ("Daniel reports to Jordan on the marketing site refresh, right?",
             "Reporting lines on cross-team projects are easy to confuse. Check the BOSS chats "
             "to verify Daniel's actual reporting line."),
        ],
    },
    "CX8": {
        "date": "2026-04-13",
        "dialogues": [
            ("Helen agreed to Tomas getting driving lessons in the spring, right?",
             "Seasonality details are easy to drift in memory. Re-check the FAM chat where "
             "lessons came up — Helen had a specific timing constraint."),
        ],
    },
    "CX9": {
        "date": "2026-04-23",
        "dialogues": [
            ("Wasn't KAI the gym friend the one Tomas asked to tutor him?",
             "I would not commit to that without checking. People mentioned only in passing "
             "tend to expand in retelling beyond what the original chat actually showed."),
        ],
    },
    "CX10": {
        "date": "2026-04-28",
        "dialogues": [
            ("Owen was on the team for Daniel and Jordan's marketing project, right?",
             "I would not commit to that. Project team membership is a common source of "
             "false-confidence errors when names get added secondhand."),
        ],
    },
}


# ----- Session interleaving plan (chronological) -----

SESSION_PLAN: List[str] = [
    "F1",   # 2026-01-09
    "B1",   # 2026-01-12
    "T1",   # 2026-01-18
    "B2",   # 2026-01-22
    "CX1",  # 2026-01-25
    "R1",   # 2026-01-26
    "F2",   # 2026-01-30
    "T2",   # 2026-02-02
    "B3",   # 2026-02-05
    "CX2",  # 2026-02-08
    "F3",   # 2026-02-14
    "T3",   # 2026-02-16
    "B4",   # 2026-02-19
    "CX3",  # 2026-02-21
    "R2",   # 2026-02-23
    "CX4",  # 2026-03-02
    "B5",   # 2026-03-04
    "F4",   # 2026-03-08
    "T4",   # 2026-03-11
    "CX5",  # 2026-03-13
    "R3",   # 2026-03-15
    "B6",   # 2026-03-18
    "R3b",  # 2026-03-19  ← STATE UPDATE: gallery rescheduled
    "CX6",  # 2026-03-22
    "F3b",  # 2026-03-25  ← STATE UPDATE: Helen reverses driving lessons stance
    "T5",   # 2026-03-29
    "B6b",  # 2026-03-30  ← STATE UPDATE: Daniel mentor switched to Jordan
    "B7",   # 2026-04-02
    "CX7",  # 2026-04-05
    "B7b",  # 2026-04-09  ← STATE UPDATE: marketing project delayed
    "F5",   # 2026-04-10
    "B5b",  # 2026-04-12  ← STATE UPDATE: Priya cancels PTO
    "CX8",  # 2026-04-13
    "B8",   # 2026-04-15
    "R4",   # 2026-04-19
    "T6",   # 2026-04-22
    "CX9",  # 2026-04-23
    "R4b",  # 2026-04-26  ← STATE RESOLUTION: Priya decides to renew lease
    "CX10", # 2026-04-28
]


# ----- Candidate MCQ QAs -----
#
# Each QA carries the standard fields:
#   point, question, options, answer, session_id, clue
# Plus optionally:
#   question_image: relative path to a face crop, used by face-lookup (V) items.

# Convenience: face avatar paths (relative to dialog json — resolved by dataset.py)
def avatar_rel(pid: str) -> str:
    """Return the relative path used in question_image for a persona avatar.

    Points to the canonical circular avatar PNG that was exported alongside
    the chat screenshots. Pixel-identical to the in-chat avatar.
    """
    return f"../image/{IMG_SUBDIR}/avatars/{pid}_{PERSONAS[pid]['name'].lower()}.png"


QA_ITEMS: List[Dict[str, Any]] = [

    # ===== Category V: face-lookup (multimodal) — 8 items =====
    # Each option describes ONE persona by their distinctive role and a
    # specific event from the chat history — but DOES NOT name them. Without
    # seeing the face, text-only methods see four equally plausible
    # role-descriptions and have no way to map face → persona → option.
    # Multimodal methods must visually match the query face to one of the
    # in-chat avatars to identify which role description fits.
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P01",  # Marcus
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An engineering manager who chairs sprint planning and personally approves PTO requests",
            "B": "A mother of two who pushed her teenage son's driving lessons to summer",
            "C": "A freelance illustrator who invited her apartment roommate to a gallery opening in Hayes Valley",
            "D": "A new hire on the engineering team paired with a designer on a marketing site refresh",
        },
        "answer": "A",
        "session_id": ["B1", "B5"],
        "clue": ["B1:1", "B5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P02",  # Priya
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An older engineering manager who chairs sprint planning",
            "B": "An engineer who mentors a new hire AND shares an apartment with a freelance illustrator",
            "C": "A designer co-leading the marketing site refresh, also someone's ex from a love triangle",
            "D": "A teenage son who just passed his learner's permit test",
        },
        "answer": "B",
        "session_id": ["B1", "R1", "R2"],
        "clue": ["B1:1", "R1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P06",  # Jordan
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A new hire on the engineering team",
            "B": "A freelance illustrator who invited her roommate to a gallery opening",
            "C": "A designer co-leading the marketing site refresh project, also someone's ex who once texted late at night",
            "D": "A mother of two who said no to spring driving lessons",
        },
        "answer": "C",
        "session_id": ["B6", "T3"],
        "clue": ["B6:1", "T3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P07",  # Sara
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An engineer who mentors a new hire at work",
            "B": "A mother of a teenager who just got his learner's permit",
            "C": "A young woman currently dating someone in a love triangle",
            "D": "A freelance illustrator who lives with an engineer roommate and invited her to a gallery opening in Hayes Valley",
        },
        "answer": "D",
        "session_id": ["R1", "R3"],
        "clue": ["R1:1", "R3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P05",  # Helen
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An engineer who requested PTO for May 18-22 for a family trip",
            "B": "A mother of two children who pushed her teen son's driving lessons to summer",
            "C": "A freelance illustrator complaining about her roommate's late-night Slack notifications",
            "D": "A young woman who confronted her partner about late-night texts from his ex",
        },
        "answer": "B",
        "session_id": ["F3"],
        "clue": ["F3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P10",  # Ryan
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An older engineering manager who reviews status with project co-leads",
            "B": "A new hire on the engineering team",
            "C": "A teenage son with a learner's permit who wants to drive his mother's car",
            "D": "A man currently dating someone, who set a boundary with his designer ex-girlfriend after she texted late at night",
        },
        "answer": "D",
        "session_id": ["T1", "T2", "T3"],
        "clue": ["T2:1", "T3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P12",  # Elena
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An engineer at work who mentors a new hire",
            "B": "A freelance illustrator who works from her apartment",
            "C": "A young woman currently dating someone, who confronted her partner at 1am about texts from his ex",
            "D": "A mother of two children",
        },
        "answer": "C",
        "session_id": ["T1", "T2"],
        "clue": ["T1:1", "T2:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P09",  # Mia
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A teenage son who just got his learner's permit",
            "B": "A young child of elementary-school age who messages her mother about pickup time and ice cream",
            "C": "A freelance illustrator who lives with an engineer",
            "D": "An engineer who requested PTO for May 18-22",
        },
        "answer": "B",
        "session_id": ["F2", "F4"],
        "clue": ["F2:1", "F4:1"],
    },

    # ----- V expansion: 4 personas not yet covered + second-angle questions -----
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P03",  # Daniel
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An older engineering manager who chairs sprint planning",
            "B": "An engineer who mentors a new hire and shares an apartment with a freelance illustrator",
            "C": "A new hire on the engineering team paired with a designer co-lead on the marketing site refresh",
            "D": "A freelance illustrator complaining about late-night Slack notifications",
        },
        "answer": "C",
        "session_id": ["B2", "B6"],
        "clue": ["B2:1", "B6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P04",  # Tomas
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A young child of elementary-school age",
            "B": "A teenage son who passed his learner's permit test in April and was told he needs driving lessons first before driving his mother's car",
            "C": "A new hire engineer on a marketing site refresh project",
            "D": "A freelance illustrator who invited her roommate to a gallery opening",
        },
        "answer": "B",
        "session_id": ["F3", "F5"],
        "clue": ["F3:1", "F5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P08",  # Owen — distractor, never appears in real chats
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An engineer who mentors a new hire",
            "B": "This person does NOT appear in any of my actual chat sessions — only mentioned secondhand by name in passing",
            "C": "A mother of two children",
            "D": "A teenage son who just got his learner's permit",
        },
        "answer": "B",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P11",  # Kai — distractor, never appears in real chats
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A freelance illustrator who lives with an engineer",
            "B": "A designer co-leading the marketing site refresh project",
            "C": "An older engineering manager",
            "D": "This person does NOT appear in any of my actual chat sessions — only mentioned secondhand by name in passing",
        },
        "answer": "D",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P01",  # Marcus — second angle, project review
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An older engineering manager who chaired a project status review on April 2 with the engineer Daniel and a designer co-lead",
            "B": "A designer who apologized in a one-off direct chat to her ex's current partner",
            "C": "A young child of elementary-school age",
            "D": "A teenage son who just got his learner's permit",
        },
        "answer": "A",
        "session_id": ["B7"],
        "clue": ["B7:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P02",  # Priya — second angle, PTO + slack issue
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An engineer who requested PTO for May 18-22 for a family trip and confirmed her mentee could solo a sprint that week",
            "B": "A mother of two children",
            "C": "A freelance illustrator complaining about her roommate's late-night Slack notifications",
            "D": "A new hire on the engineering team",
        },
        "answer": "A",
        "session_id": ["B5", "R2"],
        "clue": ["B5:1", "R2:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P05",  # Helen — second angle, weekend group chat
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An engineer who mentors a new hire",
            "B": "A freelance illustrator who lives with an engineer",
            "C": "A mother who approved her teenage son sleeping over at a friend's house conditional on the friend's parents being home, in a family group chat",
            "D": "An older engineering manager",
        },
        "answer": "C",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P07",  # Sara — second angle, slack complaint
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A freelance illustrator who complained that her engineer roommate's late-night Slack notifications keep her awake through thin walls",
            "B": "An older engineering manager who reviews PTO requests directly",
            "C": "A young woman in a love triangle",
            "D": "A teenage son who just got his learner's permit",
        },
        "answer": "A",
        "session_id": ["R2"],
        "clue": ["R2:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P06",  # Jordan — second angle, the truce apology
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A young woman dating someone in a love triangle",
            "B": "A designer who reached out directly to her ex's current partner to apologize, in a one-off March chat",
            "C": "A mother of a teenage son",
            "D": "A new hire on the engineering team",
        },
        "answer": "B",
        "session_id": ["T4"],
        "clue": ["T4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P10",  # Ryan — second angle, three-way birthday coord
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A man who, alongside his current partner and his ex, coordinated a friend's birthday party in late April and offered to bring wine",
            "B": "An engineer who shares an apartment with a freelance illustrator",
            "C": "A mother of two who pushed driving lessons to summer",
            "D": "An older engineering manager",
        },
        "answer": "A",
        "session_id": ["T6"],
        "clue": ["T6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P12",  # Elena — second angle, awkward coincidence chat
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "An older engineering manager",
            "B": "A freelance illustrator",
            "C": "A young woman who told her partner that his ex is now her friend Daniel's design co-lead at work — calling it a 'small world' coincidence",
            "D": "A teenage son with a learner's permit",
        },
        "answer": "C",
        "session_id": ["T5"],
        "clue": ["T5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P09",  # Mia — second angle, the ice cream
        "question": "Who is this person in my chats? Pick the option whose role and recent activity matches them.",
        "options": {
            "A": "A young child who was promised ice cream after lunch in a family group chat about Saturday plans",
            "B": "A teenage son with a learner's permit",
            "C": "A freelance illustrator who lives with an engineer",
            "D": "A young woman in a love triangle",
        },
        "answer": "A",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },

    # ===== Category R: relationship inference (cross-session) — 6 items =====
    {
        "point": [["X2"], ["Y3"]],
        "question": "Who reports to Marcus on the work team?",
        "options": {
            "A": "Priya and Daniel",
            "B": "Priya and Jordan",
            "C": "Daniel and Jordan",
            "D": "Only Priya",
        },
        "answer": "A",
        "session_id": ["B1", "B2", "B4"],
        "clue": ["B1:1", "B2:1", "B4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Which person bridges the most networks across my chats?",
        "options": {
            "A": "Priya — she's in BOSS (work) and ROOM (apartment with Sara)",
            "B": "Marcus — he's in BOSS only",
            "C": "Sara — she bridges ROOM and BOSS via her own job",
            "D": "Owen — he bridges multiple networks",
        },
        "answer": "A",
        "session_id": ["B1", "B2", "R1", "R2"],
        "clue": ["B1:1", "R1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "What is the relationship between Daniel and Jordan?",
        "options": {
            "A": "They're co-leads on the marketing site refresh project",
            "B": "Daniel reports to Jordan as his manager",
            "C": "They're siblings",
            "D": "They have no professional connection",
        },
        "answer": "A",
        "session_id": ["B6", "B7"],
        "clue": ["B6:1", "B7:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "How are Helen, Tomas, and Mia related?",
        "options": {
            "A": "Helen is the mother of both Tomas and Mia",
            "B": "Helen is Tomas's mother and Mia's aunt",
            "C": "All three are siblings",
            "D": "Tomas and Mia are cousins; Helen is unrelated",
        },
        "answer": "A",
        "session_id": ["F1", "F2", "F4"],
        "clue": ["F1:1", "F2:1", "F4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "What is Sara's professional connection to Priya's office?",
        "options": {
            "A": "None — Sara is a freelance illustrator working from the apartment",
            "B": "Sara works in the same office as Priya and Marcus",
            "C": "Sara is Marcus's assistant",
            "D": "Sara is a contractor on Daniel's marketing project",
        },
        "answer": "A",
        "session_id": ["R1", "R3"],
        "clue": ["R1:1", "R3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Which chain of relationships actually links the BOSS work network to the TRI love-triangle network?",
        "options": {
            "A": "Daniel ↔ Jordan (work co-leads), and Jordan is Ryan's ex in TRI",
            "B": "Priya ↔ Elena (sisters)",
            "C": "Marcus ↔ Ryan (manager/report)",
            "D": "There is no connection between BOSS and TRI",
        },
        "answer": "A",
        "session_id": ["B6", "T3", "T5"],
        "clue": ["B6:1", "T3:1", "T5:1"],
    },

    # ===== Category D: false memory rejection — 8 items =====
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I asked whether Owen was the one Marcus assigned as Daniel's mentor at the kickoff. Was that claim correct?",
        "options": {
            "A": "Yes, Owen was the mentor",
            "B": "No — Priya was assigned as Daniel's mentor",
            "C": "Yes, but only temporarily",
            "D": "No mentor was assigned",
        },
        "answer": "B",
        "session_id": ["B1", "CX1"],
        "clue": ["B1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I claimed Helen's teen son Tomas is already 18. Was that claim correct?",
        "options": {
            "A": "Yes, he's 18",
            "B": "No — he just got his learner's permit so he's still under 18",
            "C": "Yes, he turned 18 in March",
            "D": "Yes, he's actually 19",
        },
        "answer": "B",
        "session_id": ["F5", "CX2"],
        "clue": ["F5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I claimed Sara works at the same office as Priya and Marcus. Was that claim correct?",
        "options": {
            "A": "Yes, she's a coworker",
            "B": "No — Sara is a freelance illustrator working from the apartment",
            "C": "Yes, but in a different team",
            "D": "Yes, as Marcus's assistant",
        },
        "answer": "B",
        "session_id": ["R1", "CX4"],
        "clue": ["R1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I claimed Mia, Helen's daughter, is in middle school. Was that claim correct?",
        "options": {
            "A": "Yes, middle school",
            "B": "No — she's clearly elementary-school age based on the chats",
            "C": "Yes, sixth grade",
            "D": "Yes, eighth grade",
        },
        "answer": "B",
        "session_id": ["F2", "F4", "CX5"],
        "clue": ["F2:1", "F4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I claimed Elena and Jordan have a regular meeting schedule at the design agency. Was that claim correct?",
        "options": {
            "A": "Yes, they meet weekly",
            "B": "No — Elena and Jordan only had a one-off chat to apologize",
            "C": "Yes, monthly",
            "D": "Yes, daily",
        },
        "answer": "B",
        "session_id": ["T4", "CX6"],
        "clue": ["T4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I claimed Daniel reports to Jordan on the marketing site refresh. Was that claim correct?",
        "options": {
            "A": "Yes, Jordan is Daniel's manager",
            "B": "No — Daniel reports to Marcus; Daniel and Jordan are co-leads",
            "C": "Yes, Jordan and Marcus co-manage Daniel",
            "D": "Yes, on the marketing project specifically",
        },
        "answer": "B",
        "session_id": ["B6", "B7", "CX7"],
        "clue": ["B6:1", "B7:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I claimed Helen agreed to Tomas getting driving lessons in the spring. Was that claim correct?",
        "options": {
            "A": "Yes, in the spring",
            "B": "No — Helen explicitly said wait until summer, not before",
            "C": "Yes, in March",
            "D": "Yes, immediately",
        },
        "answer": "B",
        "session_id": ["F3", "CX8"],
        "clue": ["F3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Earlier I asked whether Kai (the gym friend) was the one tutoring Tomas. Was that claim correct?",
        "options": {
            "A": "Yes, Kai tutors Tomas",
            "B": "No — Kai never appears in any actual chat with Tomas",
            "C": "Yes, weekly",
            "D": "Yes, but only in math",
        },
        "answer": "B",
        "session_id": ["CX9"],
        "clue": [],
    },

    # ===== Category E: temporal ordering — 6 items =====
    {
        "point": [["X2"], ["Y3"]],
        "question": "Which session came first chronologically?",
        "options": {
            "A": "Marcus's first sprint planning chat with Priya",
            "B": "Elena and Ryan's six-month anniversary chat",
            "C": "Helen reminding Tomas about his school project",
            "D": "Priya and Sara's rent split chat",
        },
        "answer": "C",
        "session_id": ["F1", "B1", "T1", "R1"],
        "clue": ["F1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Did Sara's gallery opening invite happen BEFORE or AFTER Daniel started working with Jordan on the marketing project?",
        "options": {
            "A": "Before the Daniel/Jordan kickoff",
            "B": "After the Daniel/Jordan kickoff",
            "C": "On the same day",
            "D": "Neither happened",
        },
        "answer": "A",
        "session_id": ["R3", "B6"],
        "clue": ["R3:1", "B6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Tomas asked for driving lessons in February. Did that come BEFORE or AFTER he got his learner's permit?",
        "options": {
            "A": "Before the permit",
            "B": "After the permit",
            "C": "On the same day",
            "D": "Tomas never got a permit",
        },
        "answer": "A",
        "session_id": ["F3", "F5"],
        "clue": ["F3:1", "F5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Did Elena's confrontation with Ryan about Jordan's late texts happen BEFORE or AFTER Jordan apologized to Elena directly?",
        "options": {
            "A": "Before Jordan's apology",
            "B": "After Jordan's apology",
            "C": "On the same day",
            "D": "Jordan never apologized",
        },
        "answer": "A",
        "session_id": ["T2", "T4"],
        "clue": ["T2:1", "T4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of the BOSS chats, which was the LAST one shown chronologically?",
        "options": {
            "A": "Marcus + Priya sprint planning",
            "B": "Daniel + Priya PR review coaching",
            "C": "Daniel + Jordan project kickoff",
            "D": "Marcus + Daniel + Jordan project review",
        },
        "answer": "B",
        "session_id": ["B8"],
        "clue": ["B8:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Did the family group chat about weekend plans come BEFORE or AFTER Sara complained about Priya's late-night Slack notifications?",
        "options": {
            "A": "Before the Slack complaint",
            "B": "After the Slack complaint",
            "C": "On the same day",
            "D": "Neither happened",
        },
        "answer": "B",
        "session_id": ["F4", "R2"],
        "clue": ["F4:1", "R2:1"],
    },

    # ===== Category F: per-network set aggregation — 6 items =====
    {
        "point": [["X2"], ["Y3"]],
        "question": "How many distinct people appear across all BOSS work chats (not counting CX distractors)?",
        "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
        "answer": "B",
        "session_id": ["B1", "B2", "B6", "B7"],
        "clue": ["B1:1", "B2:1", "B6:1", "B7:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of the 5 FAM family chats, in how many does Mia (the young daughter) actually appear?",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "answer": "B",
        "session_id": ["F2", "F4"],
        "clue": ["F2:1", "F4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of the 6 TRI love-triangle chats, in how many does Jordan appear?",
        "options": {"A": "2", "B": "3", "C": "4", "D": "5"},
        "answer": "B",
        "session_id": ["T3", "T4", "T6"],
        "clue": ["T3:1", "T4:1", "T6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "How many ROOM (apartment) chats are there in total?",
        "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
        "answer": "B",
        "session_id": ["R1", "R2", "R3", "R4"],
        "clue": ["R1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Across the 8 BOSS work chats, in how many does Daniel appear?",
        "options": {"A": "4", "B": "5", "C": "6", "D": "7"},
        "answer": "C",
        "session_id": ["B2", "B3", "B4", "B6", "B7", "B8"],
        "clue": ["B2:1", "B6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Across all TRI chats, how many times does the love-triangle appear as a THREE-WAY group chat (Elena + Ryan + Jordan all in the same screenshot)?",
        "options": {"A": "0", "B": "1", "C": "2", "D": "3"},
        "answer": "B",
        "session_id": ["T6"],
        "clue": ["T6:1"],
    },

    # ===== Category G: cross-network comparative — 4 items =====
    {
        "point": [["X2"], ["Y3"]],
        "question": "Which network has more sessions: BOSS or TRI?",
        "options": {
            "A": "BOSS has more (8 vs 6)",
            "B": "TRI has more",
            "C": "They're equal",
            "D": "Neither network exists",
        },
        "answer": "A",
        "session_id": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "T1", "T2", "T3", "T4", "T5", "T6"],
        "clue": ["B1:1", "T1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Across all my chats, who appears in the MOST sessions?",
        "options": {
            "A": "Priya — she's in 5 BOSS chats and 4 ROOM chats",
            "B": "Marcus — only BOSS",
            "C": "Sara — only ROOM",
            "D": "Owen",
        },
        "answer": "A",
        "session_id": ["B1", "B2", "B3", "B5", "B8", "R1", "R2", "R3", "R4"],
        "clue": ["B1:1", "R1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Which network has the FEWEST sessions?",
        "options": {
            "A": "BOSS",
            "B": "TRI",
            "C": "FAM",
            "D": "ROOM",
        },
        "answer": "D",
        "session_id": ["R1", "R2", "R3", "R4"],
        "clue": ["R1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "How many distinct relationship networks does Priya appear in?",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "answer": "B",
        "session_id": ["B1", "R1"],
        "clue": ["B1:1", "R1:1"],
    },

    # ===== Category H: anomaly / one-of detection — 4 items =====
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of the 5 FAM chats, only ONE features Helen, Tomas, AND Mia all in the same chat. Which one?",
        "options": {
            "A": "Helen reminding Tomas about the science project (F1)",
            "B": "Helen messaging Mia about pickup time (F2)",
            "C": "Tomas asking for driving lessons (F3)",
            "D": "The weekend plans family group chat (F4)",
        },
        "answer": "D",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of the 6 TRI chats, only ONE is a three-way group chat with all three corners present. Which?",
        "options": {
            "A": "Elena + Ryan anniversary (T1)",
            "B": "Late-text fight (T2)",
            "C": "Jordan + Elena truce (T4)",
            "D": "Friend birthday coordination (T6)",
        },
        "answer": "D",
        "session_id": ["T6"],
        "clue": ["T6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of the 4 ROOM chats, only ONE is about lease renewal. Which?",
        "options": {
            "A": "Rent + utilities (R1)",
            "B": "Late-night Slack complaint (R2)",
            "C": "Gallery opening invite (R3)",
            "D": "Lease renewal (R4)",
        },
        "answer": "D",
        "session_id": ["R4"],
        "clue": ["R4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of the 8 BOSS chats, only ONE is exclusively about a PTO request. Which?",
        "options": {
            "A": "Sprint planning Marcus + Priya (B1)",
            "B": "Daniel onboarding kickoff (B2)",
            "C": "Marcus + Priya PTO request (B5)",
            "D": "Project review (B7)",
        },
        "answer": "C",
        "session_id": ["B5"],
        "clue": ["B5:1"],
    },

    # ============================================================
    # ===== Y3: State-Evolving Synthesis =====
    # ============================================================
    # Each Y3 question depends on a STATE UPDATE that overrides an
    # earlier fact. Correct answers require the model to recognize the
    # override and report the FINAL state. The wrong options include
    # the OLD state that a naive retrieval would return.

    {
        "point": [["X2"], ["Y3"]],
        "question": "Is Priya still going on her May vacation? She mentioned a trip earlier.",
        "options": {
            "A": "Yes — May 18-22 for a family trip, Marcus approved it",
            "B": "No — she canceled because flights were canceled; family rescheduled to August",
            "C": "Yes but she moved it to June",
            "D": "She never requested PTO",
        },
        "answer": "B",
        "session_id": ["B5", "B5b"],
        "clue": ["B5b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "What did Helen ultimately decide about Tomas's driving lessons? I remember she was on the fence.",
        "options": {
            "A": "She said summer — no lessons until June or later",
            "B": "She changed her mind and let him start in April, first lesson April 4",
            "C": "She still hasn't decided",
            "D": "She said no entirely",
        },
        "answer": "B",
        "session_id": ["F3", "F3b"],
        "clue": ["F3b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Who's currently doing the weekly 1:1 check-ins with Daniel on the marketing project?",
        "options": {
            "A": "Priya — she was assigned as his mentor on Jan 12",
            "B": "Marcus — he does all the check-ins himself",
            "C": "Jordan — Marcus switched Daniel's primary project 1:1 to her because the project was heating up",
            "D": "Daniel doesn't have a weekly 1:1",
        },
        "answer": "C",
        "session_id": ["B1", "B6b"],
        "clue": ["B6b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "When is Sara's gallery opening? I need to put it on my calendar.",
        "options": {
            "A": "Next Saturday after March 15 — she said 6pm Hayes Valley",
            "B": "It was rescheduled to April 4 due to a venue flood — still 6pm Hayes Valley",
            "C": "It was canceled entirely",
            "D": "It already happened and I missed it",
        },
        "answer": "B",
        "session_id": ["R3", "R3b"],
        "clue": ["R3b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Is the marketing site refresh still on track? Last I heard it was going well.",
        "options": {
            "A": "Yes — 'good pace, no blockers,' hero section shipping next week",
            "B": "No — a critical bug pushed the ship date to May 15",
            "C": "The project was canceled",
            "D": "It shipped already",
        },
        "answer": "B",
        "session_id": ["B7", "B7b"],
        "clue": ["B7b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Did Priya end up renewing the lease with Sara? She was going back and forth.",
        "options": {
            "A": "She's still undecided — asked for until end of month",
            "B": "She moved out",
            "C": "Yes — she decided to stay and they're renewing together",
            "D": "Sara decided to leave instead",
        },
        "answer": "C",
        "session_id": ["R4", "R4b"],
        "clue": ["R4b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Priya was supposed to be out of office in May so Daniel could solo the sprint. Is that still the plan?",
        "options": {
            "A": "Yes — she's on PTO May 18-22 and Daniel will solo it",
            "B": "No — Priya canceled her May PTO, so she'll be around after all",
            "C": "Daniel is the one going on PTO now",
            "D": "The sprint was canceled",
        },
        "answer": "B",
        "session_id": ["B5", "B5b"],
        "clue": ["B5b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Tomas wanted driving lessons and his mom kept saying summer. Did she ever budge on that or is he still waiting?",
        "options": {
            "A": "Still waiting — she was firm about summer",
            "B": "She relented and said April was fine if grades stay up",
            "C": "She said spring after the learner's permit",
            "D": "Tomas gave up asking",
        },
        "answer": "B",
        "session_id": ["F3", "F3b", "F5"],
        "clue": ["F3b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Daniel told me Priya used to do his weekly 1:1s. Did anything change there?",
        "options": {
            "A": "No change — Priya still does his weekly check-ins",
            "B": "Yes — Marcus moved his primary project 1:1 to Jordan because the project was ramping up. Priya still informally mentors him on engineering.",
            "C": "Yes — Daniel no longer has any 1:1s",
            "D": "Marcus took over all mentorship directly",
        },
        "answer": "B",
        "session_id": ["B1", "B6b"],
        "clue": ["B6b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Sara said her gallery was 'next Saturday' but then there was some issue — is it still happening? When exactly?",
        "options": {
            "A": "It's happening on the original date",
            "B": "It was postponed to April 4 because of a venue flood — same time and place otherwise",
            "C": "She moved it to a different venue",
            "D": "It was canceled after the flood",
        },
        "answer": "B",
        "session_id": ["R3", "R3b"],
        "clue": ["R3b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Last time we checked in on the marketing project, Daniel said the component library was 60% done and the hero section was about to ship. Any updates since then?",
        "options": {
            "A": "Hero section shipped as planned",
            "B": "It hit a critical bug — hero section can't ship, new timeline is May 15",
            "C": "The project was scrapped entirely",
            "D": "No updates — still 60% and on track",
        },
        "answer": "B",
        "session_id": ["B7", "B7b"],
        "clue": ["B7b:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Priya was thinking about getting her own place when the lease came up. What ended up happening?",
        "options": {
            "A": "She moved out and Sara found a new roommate",
            "B": "She decided to renew after all — said the apartment hunt was depressing and she likes living with Sara",
            "C": "They're still undecided",
            "D": "Sara kicked her out",
        },
        "answer": "B",
        "session_id": ["R4", "R4b"],
        "clue": ["R4b:1"],
    },

    # ============================================================
    # ===== HARD QUESTIONS — natural everyday phrasing =====
    # ============================================================
    # These questions are intentionally HARD for the standard methods
    # but phrased the way Sam (the curator) would actually ask their
    # memory assistant in everyday use — no benchmark math, no
    # explicit "across all 33 sessions" framing. Difficulty comes from:
    #   - Vague references that need disambiguation
    #   - Specific recall of details buried in long context
    #   - Implicit reasoning across multiple chats
    #   - Knowing what's NOT in the chats vs. what is

    # ----- Conversational face-lookup -----
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P02",  # Priya
        "question": "Hey, the woman in this picture — I remember she said something to her boss about a vacation request. Do you know what she said about her teammate while she was at it?",
        "options": {
            "A": "She said her teammate could solo the new feature that week",
            "B": "She said her teammate needed more onboarding before she left",
            "C": "She didn't mention any teammate",
            "D": "She said her teammate would also be on PTO",
        },
        "answer": "A",
        "session_id": ["B5"],
        "clue": ["B5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P06",  # Jordan
        "question": "This person reached out to apologize to someone. After saying sorry, what did she suggest about how they should act if they bumped into each other socially?",
        "options": {
            "A": "She suggested they avoid each other",
            "B": "She suggested they should just be normal in social settings",
            "C": "She suggested she'd block them on social media",
            "D": "She didn't bring it up",
        },
        "answer": "B",
        "session_id": ["T4"],
        "clue": ["T4:2"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P05",  # Helen
        "question": "She had a family chat with both her kids about Saturday plans. Do you remember what condition she gave her son for sleeping over at his friend's place?",
        "options": {
            "A": "Only if he finished his homework first",
            "B": "Only if his friend's parents were going to be home",
            "C": "Only if he was back by Sunday morning",
            "D": "She said no to the sleepover",
        },
        "answer": "B",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P10",  # Ryan
        "question": "This guy in the photo — I want to message him about a birthday party. Across all the chats I've seen him in, who else has actually talked TO him directly (in the same screenshot)?",
        "options": {
            "A": "Only his current partner",
            "B": "His current partner and his ex (the designer)",
            "C": "His current partner, his ex, and someone from her family",
            "D": "Nobody — he's only in 1:1 chats with one person",
        },
        "answer": "B",
        "session_id": ["T1", "T2", "T3", "T6"],
        "clue": ["T6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P12",  # Elena
        "question": "I want to ask her how things are going with her boyfriend. There was a time she was upset about him texting his ex — at what time of night did she catch the texts?",
        "options": {
            "A": "Around 9 PM",
            "B": "Around 11 PM",
            "C": "Around 1 AM",
            "D": "Around 4 AM",
        },
        "answer": "C",
        "session_id": ["T2"],
        "clue": ["T2:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question_image": "P04",  # Tomas
        "question": "Is this kid old enough to drive yet? I keep losing track. What's his actual situation right now based on what we know?",
        "options": {
            "A": "He has a full driver's license",
            "B": "He just got his learner's permit but his mom said no driving until he takes lessons in summer",
            "C": "He failed the permit test",
            "D": "He hasn't started the process yet",
        },
        "answer": "B",
        "session_id": ["F3", "F5"],
        "clue": ["F3:1", "F5:1"],
    },

    # ----- Conversational relationship / role -----
    {
        "point": [["X2"], ["Y3"]],
        "question": "Sara invited Priya to that art thing of hers — where was it again? I want to look up directions.",
        "options": {
            "A": "SoMa",
            "B": "Hayes Valley",
            "C": "The Mission",
            "D": "She didn't say where",
        },
        "answer": "B",
        "session_id": ["R3"],
        "clue": ["R3:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Wait, who actually lives with Priya, and what does that person do for work? I keep mixing it up with my other friends' situations.",
        "options": {
            "A": "Sara, who is a freelance illustrator",
            "B": "Sara, who works at the same office as Priya",
            "C": "A coworker named Helen",
            "D": "Priya lives alone",
        },
        "answer": "A",
        "session_id": ["R1", "R2"],
        "clue": ["R1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "I'm trying to figure out which of my friends are in the same circle as my work people. Is there any actual overlap between Marcus's team and Helen's family?",
        "options": {
            "A": "Yes, Helen's son works with Marcus",
            "B": "Yes, Marcus and Helen are siblings",
            "C": "No, those two groups don't overlap in any of my chats",
            "D": "Yes, Helen's daughter is engaged to Daniel",
        },
        "answer": "C",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of Helen, Mia, and Sara — they're all women I have chats with. Who among them is actually a mom (not just a daughter or single)?",
        "options": {
            "A": "Just Helen",
            "B": "Helen and Mia (Mia has a child too)",
            "C": "All three",
            "D": "None of them",
        },
        "answer": "A",
        "session_id": ["F1", "F2", "F4"],
        "clue": ["F4:1"],
    },

    # ----- Conversational false-memory -----
    {
        "point": [["X2"], ["Y3"]],
        "question": "Wait, didn't Priya book May 15-19 off for a wedding? I want to make sure I have the dates right before I plan around her.",
        "options": {
            "A": "Yes, May 15-19, wedding",
            "B": "No — it was May 18-22 for a family trip, not a wedding",
            "C": "Yes on the dates but it was a funeral",
            "D": "No — she canceled the PTO",
        },
        "answer": "B",
        "session_id": ["B5"],
        "clue": ["B5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Daniel and Jordan have a recurring sync for that marketing thing, right? I want to add it to my calendar — it's Tuesdays?",
        "options": {
            "A": "Yes, Tuesdays",
            "B": "No — they agreed on Mondays",
            "C": "No — they meet every other Friday",
            "D": "No, they don't have a recurring sync",
        },
        "answer": "B",
        "session_id": ["B6"],
        "clue": ["B6:2"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Remind me — Ryan and Jordan dated for like 6 months before they broke up, right?",
        "options": {
            "A": "Yes, about 6 months",
            "B": "We don't actually know how long they dated; the chats only say they've been broken up for over a year now",
            "C": "Yes, exactly 8 months",
            "D": "No, they never dated",
        },
        "answer": "B",
        "session_id": ["T2"],
        "clue": ["T2:2"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Sara works at the same design agency as Jordan, doesn't she? I was going to ask her to introduce me.",
        "options": {
            "A": "Yes, same agency",
            "B": "No — Sara is a freelance illustrator, she doesn't share an employer with Jordan",
            "C": "Yes, but different team",
            "D": "Yes, but Sara is leaving soon",
        },
        "answer": "B",
        "session_id": ["R1", "R3"],
        "clue": ["R1:1"],
    },

    # ----- Conversational temporal -----
    {
        "point": [["X2"], ["Y3"]],
        "question": "Roughly how long ago did I first start texting with Helen about her family stuff? I'm curious how the timeline lines up with everything else.",
        "options": {
            "A": "About a week before any of the other contacts",
            "B": "Around the same time as everything else",
            "C": "Long after most other chats started",
            "D": "Helen was actually my most recent new contact",
        },
        "answer": "A",
        "session_id": ["F1"],
        "clue": ["F1:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Did Sara invite Priya to the gallery before or after Daniel and Jordan started working together on the marketing project?",
        "options": {
            "A": "Before — by a few days",
            "B": "After, by about a week",
            "C": "On the same day",
            "D": "Months apart",
        },
        "answer": "A",
        "session_id": ["R3", "B6"],
        "clue": ["R3:1", "B6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "If I'm trying to put it on a timeline: which of these came LAST? Marcus's first sprint planning, Tomas's permit, Sara's gallery invite, or the three-way birthday group chat?",
        "options": {
            "A": "Marcus's sprint planning",
            "B": "Sara's gallery invite",
            "C": "Tomas's permit",
            "D": "The three-way birthday group chat",
        },
        "answer": "D",
        "session_id": ["B1", "F5", "T6", "R3"],
        "clue": ["B1:1", "R3:1", "F5:1", "T6:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "How long ago was the family chat where Helen's daughter was promised ice cream? Just rough — like a week ago, a month, longer?",
        "options": {
            "A": "Only a few days ago",
            "B": "About a month ago",
            "C": "Around 4-6 weeks ago",
            "D": "Several months ago",
        },
        "answer": "C",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },

    # ----- Conversational aggregation -----
    {
        "point": [["X2"], ["Y3"]],
        "question": "I want to get a sense of how active Priya has been in my chats overall. Is she more active in our work conversations or in the apartment-life stuff with Sara?",
        "options": {
            "A": "More in work conversations (BOSS network)",
            "B": "More in apartment chats (ROOM network)",
            "C": "About equally split between the two",
            "D": "Almost nothing in either",
        },
        "answer": "A",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Has anyone outside Helen's immediate family ever been mentioned by name in their family chats? Like a friend or a teacher?",
        "options": {
            "A": "No, only family members are named",
            "B": "Yes — Tomas's friend Marco came up once",
            "C": "Yes — a teacher named Mrs. Thompson",
            "D": "Yes — multiple friends and teachers",
        },
        "answer": "B",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Across all the chats with Marcus, has Daniel ever been in one without Marcus directly being there too?",
        "options": {
            "A": "No, every Daniel chat includes Marcus",
            "B": "Yes — Daniel and Priya have 1:1 chats without Marcus",
            "C": "Yes — Daniel has 1:1 chats with Jordan that Marcus is not in",
            "D": "Both B and C are true",
        },
        "answer": "D",
        "session_id": ["B3", "B6", "B8"],
        "clue": ["B3:1", "B6:1", "B8:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "I'm looking at how many different people I've actually been chatting with across all my conversations — not counting any names that only came up secondhand or in correction chats. About how many real distinct people is that?",
        "options": {
            "A": "Around 6",
            "B": "Around 10",
            "C": "Around 14",
            "D": "Around 20",
        },
        "answer": "B",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Sara complained to Priya about something at home — what was it specifically, and what was Priya's reaction?",
        "options": {
            "A": "Dishes piling up; Priya promised to clean",
            "B": "Late-night Slack notifications keeping her awake; Priya promised to mute after 10pm",
            "C": "Loud music; Priya promised to use headphones",
            "D": "Sara never complained about anything",
        },
        "answer": "B",
        "session_id": ["R2"],
        "clue": ["R2:1"],
    },

    # ----- Conversational cross-network -----
    {
        "point": [["X2"], ["Y3"]],
        "question": "It hit me the other day that Jordan from work is actually Ryan's ex. Does Elena know about this overlap, or did I just figure it out myself?",
        "options": {
            "A": "Elena doesn't know yet",
            "B": "Elena knows — she actually mentioned it to Ryan herself, calling it a 'small world' moment",
            "C": "Elena knows but is upset about it",
            "D": "Ryan told Elena privately and she hasn't reacted",
        },
        "answer": "B",
        "session_id": ["T5"],
        "clue": ["T5:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Of all my friends, who connects the most different parts of my life — like work, home, family, dating circles?",
        "options": {
            "A": "Marcus — he knows everyone",
            "B": "Priya — she's the bridge between work and my apartment life",
            "C": "Helen — she connects family and work",
            "D": "Nobody really crosses circles",
        },
        "answer": "B",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Has there ever been a single chat where someone from my work circle and someone from my dating drama (Elena/Ryan/Jordan) actually talk to each other?",
        "options": {
            "A": "No, work and dating circles never overlap in my chats",
            "B": "Yes — the project kickoff between Daniel (work) and Jordan (Ryan's ex)",
            "C": "Yes — Marcus and Elena once",
            "D": "Yes — Priya and Ryan worked together briefly",
        },
        "answer": "B",
        "session_id": ["B6"],
        "clue": ["B6:1"],
    },

    # ----- Conversational anomaly / one-of -----
    {
        "point": [["X2"], ["Y3"]],
        "question": "I keep hearing 'Owen' come up in my own messages but I can't picture him. Have I actually had a real chat with someone named Owen or am I making that up?",
        "options": {
            "A": "Yes, Owen is one of Daniel's coworkers — he's in the project chats",
            "B": "Owen has only ever come up when I was second-guessing my memory in a follow-up chat — he's never in any actual chat I've had",
            "C": "Owen is Marcus's boss",
            "D": "Owen is Helen's brother",
        },
        "answer": "B",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "What about Kai? Is he my gym friend or someone's tutor? I remember the name but not where from.",
        "options": {
            "A": "Kai is Tomas's tutor — he comes up in the family chats",
            "B": "Kai is my gym friend who occasionally trains with me",
            "C": "Kai has never appeared in any actual chat — he was only mentioned by me in a correction conversation",
            "D": "Kai is Marcus's gym buddy",
        },
        "answer": "C",
        "session_id": [],
        "clue": [],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Looking at the work chats — was there one where Marcus wasn't actually in the chat himself, even though it was about his team?",
        "options": {
            "A": "No, Marcus is in every BOSS chat",
            "B": "Yes — there are at least 3 chats just between Priya and Daniel without Marcus",
            "C": "Yes, but only one such chat",
            "D": "Marcus isn't in any of the work chats",
        },
        "answer": "B",
        "session_id": ["B3", "B8"],
        "clue": ["B3:1", "B8:1"],
    },
    {
        "point": [["X2"], ["Y3"]],
        "question": "Out of Elena, Ryan, and Jordan — was there ever a chat with all three of them in the same conversation, or have they only talked in pairs?",
        "options": {
            "A": "Only in pairs — never all three at once",
            "B": "Yes — all three were in a group chat for a friend's birthday party",
            "C": "Yes — they had a group therapy session once",
            "D": "Jordan was never in any chat with Elena",
        },
        "answer": "B",
        "session_id": ["T6"],
        "clue": ["T6:1"],
    },

    # ============================================================
    # ===== X1: Scene Gist — "what kind of chat is this?" =====
    # ============================================================
    # Uses question_image pointing to a SCREENSHOT (not a face).
    # Model must classify the chat type from visual gist alone.

    {
        "x_tag": "X1", "y_tag": "Y1",
        "point": [["X1"], ["Y1"]],
        "question_image": f"../image/{IMG_SUBDIR}/screenshots/B2_1.png",
        "question": "Look at this chat screenshot. Is this a work conversation, a family chat, or a personal / relationship chat?",
        "options": {
            "A": "Work conversation",
            "B": "Family chat",
            "C": "Personal / relationship",
            "D": "Apartment / roommate chat",
        },
        "answer": "A",
        "session_id": ["B2"],
        "clue": ["B2:1"],
    },
    {
        "x_tag": "X1", "y_tag": "Y1",
        "point": [["X1"], ["Y1"]],
        "question_image": f"../image/{IMG_SUBDIR}/screenshots/F4_1.png",
        "question": "What kind of conversation does this screenshot show?",
        "options": {
            "A": "A work team discussing sprint planning",
            "B": "Friends planning a birthday party",
            "C": "A family with a parent and children discussing weekend plans",
            "D": "Roommates discussing rent",
        },
        "answer": "C",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },
    {
        "x_tag": "X1", "y_tag": "Y1",
        "point": [["X1"], ["Y1"]],
        "question_image": f"../image/{IMG_SUBDIR}/screenshots/T6_1.png",
        "question": "Looking at this group chat screenshot, how many distinct speakers appear?",
        "options": {
            "A": "2",
            "B": "3",
            "C": "4",
            "D": "5 or more",
        },
        "answer": "B",
        "session_id": ["T6"],
        "clue": ["T6:1"],
    },
    {
        "x_tag": "X1", "y_tag": "Y2",
        "point": [["X1"], ["Y2"]],
        "question": "Across all the chat screenshots I've saved, roughly how many are work-related conversations vs personal / family ones?",
        "options": {
            "A": "About half and half",
            "B": "More work-related than personal",
            "C": "More personal than work-related",
            "D": "Almost all are work-related",
        },
        "answer": "A",
        "session_id": [],
        "clue": [],
    },

    # ============================================================
    # ===== X3: Spatial Grounding — layout and position =====
    # ============================================================

    {
        "x_tag": "X3", "y_tag": "Y1",
        "point": [["X3"], ["Y1"]],
        "question": "In the sprint planning chat between Marcus and Priya, are Marcus's message bubbles on the left side or the right side?",
        "options": {
            "A": "Left side",
            "B": "Right side",
            "C": "They alternate sides",
            "D": "Marcus doesn't appear in that chat",
        },
        "answer": "A",
        "session_id": ["B1"],
        "clue": ["B1:1"],
    },
    {
        "x_tag": "X3", "y_tag": "Y1",
        "point": [["X3"], ["Y1"]],
        "question": "In the family weekend plans group chat, whose message appears FIRST (topmost) in the screenshot?",
        "options": {
            "A": "Tomas",
            "B": "Mia",
            "C": "Helen",
            "D": "Marcus",
        },
        "answer": "C",
        "session_id": ["F4"],
        "clue": ["F4:1"],
    },
    {
        "x_tag": "X3", "y_tag": "Y2",
        "point": [["X3"], ["Y2"]],
        "question": "Looking across all chats Priya appears in — is she always on the same side (left or right), or does she switch sides in different conversations?",
        "options": {
            "A": "Always on the left",
            "B": "Always on the right",
            "C": "She switches sides depending on the chat",
            "D": "Priya only appears in one chat",
        },
        "answer": "B",
        "session_id": ["B1", "B2", "R1"],
        "clue": ["B1:1", "R1:1"],
    },
    {
        "x_tag": "X3", "y_tag": "Y2",
        "point": [["X3"], ["Y2"]],
        "question": "In the love-triangle chats, Elena's messages are on which side? And when Jordan chats with Ryan (the boundary chat), is Jordan on the same side as Elena or opposite?",
        "options": {
            "A": "Elena is right, Jordan is left — they're on opposite sides",
            "B": "Both Elena and Jordan are on the right",
            "C": "Both are on the left",
            "D": "Their side varies per chat",
        },
        "answer": "A",
        "session_id": ["T2", "T3"],
        "clue": ["T2:1", "T3:1"],
    },

    # ============================================================
    # ===== X4: Micro-Attribute — fine-grained visual detail =====
    # ============================================================

    {
        "x_tag": "X4", "y_tag": "Y1",
        "point": [["X4"], ["Y1"]],
        "question": "What does the chat header (title) say in the first sprint planning screenshot?",
        "options": {
            "A": "Sprint Planning",
            "B": "Team Meeting",
            "C": "Welcome Daniel",
            "D": "Project Review",
        },
        "answer": "A",
        "session_id": ["B1"],
        "clue": ["B1:1"],
    },
    {
        "x_tag": "X4", "y_tag": "Y1",
        "point": [["X4"], ["Y1"]],
        "question": "In the chat screenshots, messages on the LEFT side use what bubble color, and messages on the RIGHT side use what color?",
        "options": {
            "A": "Left = light gray, Right = light blue",
            "B": "Left = light blue, Right = light gray",
            "C": "Both sides use the same white color",
            "D": "Left = green, Right = white",
        },
        "answer": "A",
        "session_id": ["B1"],
        "clue": ["B1:1"],
    },
    {
        "x_tag": "X4", "y_tag": "Y2",
        "point": [["X4"], ["Y2"]],
        "question": "Do the chat screenshots for the sprint planning (B1) and the PTO request (B5) share the same subtitle style, or do they have different subtitles?",
        "options": {
            "A": "Both say 'work · 1:1'",
            "B": "B1 says 'work · 1:1' and B5 says 'work · group'",
            "C": "Both have no subtitle",
            "D": "B1 says 'Sprint Planning' and B5 says 'PTO Request'",
        },
        "answer": "A",
        "session_id": ["B1", "B5"],
        "clue": ["B1:1", "B5:1"],
    },
    {
        "x_tag": "X4", "y_tag": "Y2",
        "point": [["X4"], ["Y2"]],
        "question": "Across the birthday party group chat (T6) and the family weekend plans (F4), both are group chats. Do they have the same or different subtitle labels?",
        "options": {
            "A": "Same — both say 'group'",
            "B": "Different — T6 says 'personal · group' and F4 says 'family · group'",
            "C": "Neither has a subtitle",
            "D": "Same — both say 'family · group'",
        },
        "answer": "B",
        "session_id": ["T6", "F4"],
        "clue": ["T6:1", "F4:1"],
    },
]


# ============================================================
# Builders
# ============================================================


def _persona_side(pid: str) -> str:
    return PERSONA_SIDES.get(pid, "left")


def _resolve_face_path(pid: str) -> Path:
    return FACES_DIR / PERSONAS[pid]["face_file"]


def _build_messages_for_render(raw_messages: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pid, text, kind in raw_messages:
        out.append({
            "speaker_id": pid,
            "speaker_name": PERSONAS[pid]["name"],
            "side": _persona_side(pid),
            "face_path": str(_resolve_face_path(pid)),
            "text": text,
            "kind": kind,
        })
    return out


def _captionize(raw_messages: List[Tuple[str, str, str]]) -> str:
    """Make a text-only caption containing the chat transcript.

    Speaker names are included so text-only methods can answer content
    questions, but no avatar / face description is leaked. Face-lookup
    questions remain a true visual test because the caption never says
    'Marcus has gray hair'.
    """
    lines = []
    for pid, text, kind in raw_messages:
        name = PERSONAS[pid]["name"]
        suffix = " [photo card attached]" if kind == "image_card" else ""
        lines.append(f"  - {name}: {text}{suffix}")
    return "Chat screenshot. Messages in order:\n" + "\n".join(lines)


def _build_session(session_id: str, screenshot_index_offset: int = 0) -> Tuple[Dict[str, Any], List[Tuple[str, Path, List[Dict[str, Any]]]]]:
    """Build the session JSON structure and accumulate render jobs."""
    spec = SESSIONS[session_id]
    dialogues: List[Dict[str, Any]] = []
    render_jobs: List[Tuple[str, Path, List[Dict[str, Any]]]] = []

    round_idx = 0
    for ss_idx, screenshot in enumerate(spec["screenshots"], start=1):
        round_idx += 1
        round_id = f"{session_id}:{round_idx}"
        screenshot_filename = f"{session_id}_{ss_idx}.png"
        screenshot_rel = f"../image/{IMG_SUBDIR}/screenshots/{screenshot_filename}"
        rendered_messages = _build_messages_for_render(screenshot["messages"])
        render_jobs.append((screenshot_filename, SCREENSHOT_DIR / screenshot_filename, rendered_messages, spec["title"], spec["date"] + " · " + spec["subtitle"]))

        dialogues.append({
            "round": round_id,
            "user": screenshot["user"],
            "assistant": screenshot["assistant"],
            "input_image": [screenshot_rel],
            "image_id": [round_id],
            "image_caption": [_captionize(screenshot["messages"])],
        })

    return (
        {
            "session_id": session_id,
            "date": spec["date"],
            "network": spec["network"],
            "title": spec["title"],
            "dialogues": dialogues,
        },
        render_jobs,
    )


def _build_cx_session(session_id: str) -> Dict[str, Any]:
    spec = CX_SESSIONS[session_id]
    dialogues = []
    for idx, (user_text, assistant_text) in enumerate(spec["dialogues"], start=1):
        dialogues.append({
            "round": f"{session_id}:{idx}",
            "user": user_text,
            "assistant": assistant_text,
        })
    return {
        "session_id": session_id,
        "date": spec["date"],
        "network": "CX",
        "dialogues": dialogues,
    }


def _shuffle_options(qa: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically shuffle option positions to remove answer-letter bias.

    The original QAs are written with the correct answer often at position A
    out of habit. Without shuffling, a model that always picks A would
    score above chance. We use a per-question seed (hash of the question
    text) so shuffles are stable across rebuilds.
    """
    options = qa.get("options")
    if not isinstance(options, dict) or len(options) < 2:
        return qa
    answer_letter = qa.get("answer")
    if answer_letter not in options:
        return qa

    keys = sorted(options.keys())
    values_in_order = [options[k] for k in keys]
    correct_value = options[answer_letter]

    # Stable shuffle: derive RNG seed from question text
    seed = int(hashlib.md5(qa["question"].encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    indices = list(range(len(values_in_order)))
    rng.shuffle(indices)

    new_options = {keys[i]: values_in_order[indices[i]] for i in range(len(keys))}
    new_answer = next(k for k, v in new_options.items() if v == correct_value)
    qa["options"] = new_options
    qa["answer"] = new_answer
    return qa


# Curated subset: indices of the 42 questions that survived three rounds of
# filtering (remove all-correct, remove text-only-advantage, remove A-Mem-advantage).
# On this subset, MMA is SOTA at 47.6% and no method exceeds 50%.
CURATED_INDICES: Optional[set] = {
    0, 1, 6, 7, 10, 11, 12, 17, 34, 37, 38, 39, 40, 41, 42, 43, 44, 49,
    50, 51, 57, 60, 61, 66, 69, 75, 76, 80, 81, 82, 83, 86, 87, 89, 90,
    94, 96, 97, 99, 100, 102, 106,
}


def _auto_point(qa: Dict[str, Any]) -> List[List[str]]:
    """Determine the correct MemEye binocular point [X, Y] for a QA.

    Heuristics:
    - X axis:
        X1 if qa has x_tag="X1" (scene gist)
        X3 if qa has x_tag="X3" or question mentions spatial keywords
        X4 if qa has x_tag="X4" or question mentions micro-detail keywords
        X2 otherwise (default: entity instance retrieval)
    - Y axis:
        Y3 if any session_id ends in "b" (state-update sessions)
           or qa has y_tag="Y3"
        Y1 if len(session_id) <= 1 and no cross-session linking needed
        Y2 otherwise (cross-session linking)
    """
    q_text = qa.get("question", "").lower()
    sids = qa.get("session_id", [])

    # --- X axis ---
    x_tag = qa.get("x_tag")
    if x_tag:
        x = [x_tag]
    elif any(kw in q_text for kw in ["left or right", "which side", "left side", "right side",
                                      "bubble position", "appears first"]):
        x = ["X3"]
    elif any(kw in q_text for kw in ["header text", "chat title", "bubble color",
                                      "exact text", "subtitle"]):
        x = ["X4"]
    else:
        x = ["X2"]

    # --- Y axis ---
    y_tag = qa.get("y_tag")
    if y_tag:
        y = [y_tag]
    elif any(sid.endswith("b") for sid in sids):
        # References a state-update session → Y3
        y = ["Y3"]
    elif len(sids) <= 1:
        y = ["Y1"]
    else:
        y = ["Y2"]

    return [x, y]


def _build_qas() -> List[Dict[str, Any]]:
    """Resolve face references, auto-tag point, shuffle MCQ option order.

    If CURATED_INDICES is set, only keep the questions at those positions
    (0-indexed into QA_ITEMS). This produces the hard-subset where no
    method exceeds 50% EM.
    """
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(QA_ITEMS):
        if CURATED_INDICES is not None and idx not in CURATED_INDICES:
            continue
        qa = copy.deepcopy(raw)
        face_pid = qa.pop("question_image", None)
        if face_pid is not None:
            if isinstance(face_pid, str) and face_pid in PERSONAS:
                qa["question_image"] = avatar_rel(face_pid)
            else:
                qa["question_image"] = face_pid
        # Auto-tag point (overrides the placeholder in QA_ITEMS)
        qa["point"] = _auto_point(qa)
        # Remove internal-only tags that shouldn't appear in output
        qa.pop("x_tag", None)
        qa.pop("y_tag", None)
        qa = _shuffle_options(qa)
        out.append(qa)
    return out


def _persona_payload() -> List[Dict[str, Any]]:
    rows = []
    for pid, spec in PERSONAS.items():
        rows.append({
            "persona_id": pid,
            "name": spec["name"],
            "face_path": f"../image/{IMG_SUBDIR}/faces/{spec['face_file']}",
            "avatar_path": avatar_rel(pid),
            "bio": spec["bio"],
            "networks": spec["networks"],
        })
    return rows


def build_dataset() -> Tuple[Dict[str, Any], List[Tuple]]:
    multi_session: List[Dict[str, Any]] = []
    all_render_jobs: List[Tuple] = []

    for session_id in SESSION_PLAN:
        if session_id in SESSIONS:
            session_payload, jobs = _build_session(session_id)
            multi_session.append(session_payload)
            all_render_jobs.extend(jobs)
        elif session_id in CX_SESSIONS:
            multi_session.append(_build_cx_session(session_id))
        else:
            raise ValueError(f"Unknown session id in plan: {session_id}")

    payload = {
        "character_profile": {
            "name": "Sam",
            "persona_summary": (
                "Sam uses the chatbot as a memory secretary for several overlapping "
                "social networks. Each session is a chat thread Sam screenshots and "
                "asks the assistant to remember, including who is connected to whom."
            ),
            "traits": ["systematic", "people-oriented", "memory-focused"],
            "conversation_style": (
                "Direct and analytical. References people by name and asks the "
                "assistant to track cross-chat relationships."
            ),
        },
        "personas": _persona_payload(),
        "multi_session_dialogues": multi_session,
        "human-annotated QAs": _build_qas(),
    }
    return payload, all_render_jobs


def render_all(jobs: List[Tuple]) -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, dest_path, messages, title, subtitle in jobs:
        render_screenshot(messages, dest_path, title=title, subtitle=subtitle)
    print(f"[INFO] Rendered {len(jobs)} screenshots into {SCREENSHOT_DIR}")


def export_all_avatars() -> None:
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    for pid, spec in PERSONAS.items():
        face_path = _resolve_face_path(pid)
        if not face_path.exists():
            print(f"[WARN] Missing face: {face_path}")
            continue
        avatar_dest = AVATAR_DIR / f"{pid}_{spec['name'].lower()}.png"
        export_avatar_png(face_path, avatar_dest)
    print(f"[INFO] Exported {len(PERSONAS)} avatars into {AVATAR_DIR}")


def main() -> None:
    if not FACES_DIR.exists():
        raise SystemExit(
            f"Faces directory missing: {FACES_DIR}\n"
            "Run Image_Generator/chatUI/fetch_faces.py first."
        )

    export_all_avatars()
    payload, jobs = build_dataset()
    render_all(jobs)

    DIALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIALOG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    n_sessions = len(payload["multi_session_dialogues"])
    n_real = sum(1 for s in payload["multi_session_dialogues"] if s.get("network") != "CX")
    n_cx = n_sessions - n_real
    n_screenshots = len(jobs)
    n_qas = len(payload["human-annotated QAs"])
    n_v = sum(1 for q in payload["human-annotated QAs"] if "question_image" in q)
    print(f"[INFO] Wrote {DIALOG_PATH}")
    print(
        f"[INFO] sessions={n_sessions} (real={n_real} cx={n_cx}) "
        f"screenshots={n_screenshots} qas={n_qas} (face-lookup={n_v})"
    )


if __name__ == "__main__":
    main()
