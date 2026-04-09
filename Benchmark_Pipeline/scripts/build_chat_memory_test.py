#!/usr/bin/env python3
"""Build the Chat Memory Test dataset.

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
    "CX6",  # 2026-03-22
    "T5",   # 2026-03-29
    "B7",   # 2026-04-02
    "CX7",  # 2026-04-05
    "F5",   # 2026-04-10
    "CX8",  # 2026-04-13
    "B8",   # 2026-04-15
    "R4",   # 2026-04-19
    "T6",   # 2026-04-22
    "CX9",  # 2026-04-23
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


def _build_qas() -> List[Dict[str, Any]]:
    """Resolve any V-category face references and shuffle MCQ option order."""
    out: List[Dict[str, Any]] = []
    for raw in QA_ITEMS:
        qa = copy.deepcopy(raw)
        face_pid = qa.pop("question_image", None)
        if face_pid is not None:
            # Could be a persona id ("P01") or already a path string
            if isinstance(face_pid, str) and face_pid in PERSONAS:
                qa["question_image"] = avatar_rel(face_pid)
            else:
                qa["question_image"] = face_pid
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
