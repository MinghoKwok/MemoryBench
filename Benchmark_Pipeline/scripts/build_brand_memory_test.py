#!/usr/bin/env python3
"""
Build Brand Memory Test v2 (designer brainstorm + multi-image fusion).

Design philosophy:
- Each brand session is a 3-turn design discussion (not "upload + stored")
- Design takeaways are embedded in assistant text so they can be recalled
- 6 brands x 5 ads each = 30 image sessions
- 12 cross-brand distractor sessions (CX1-CX12) plant false claims
- ~70 candidate MCQ QAs across 9 categories: A/B/C (per-image), D (false memory),
  E (temporal), F (set-level aggregation within brand), G (cross-brand comparative),
  H (anomaly detection), I (design conversation continuity)

Config F curation: the canonical dataset uses a 32-QA subset selected from
the full 67 pool (see SELECTED_QA_INDICES). The full pool is retained in
QA_ITEMS so the selection can be revised without losing source material.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIALOG = REPO_ROOT / "Benchmark_Pipeline" / "data" / "dialog" / "Brand_Memory_Test.json"
IMAGE_DIR = "Brand_Memory_Test"

# Config F curation: 1-based indices into QA_ITEMS (full 67-item pool).
# 20 differentiating (agentic SOTA) + 7 all-wrong (hard ceiling) + 5 agent-fail (multimodal-favored).
# Set to None to emit the full 67-QA pool.
SELECTED_QA_INDICES: List[int] = [
    1, 2, 13, 15, 16, 17, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 37, 38, 39, 40, 42, 43, 45, 54, 55, 57, 59, 60, 61, 62, 66,
]


CHARACTER_PROFILE = {
    "name": "Alex Mercer",
    "persona_summary": (
        "Alex is a brand strategist working on a cross-category visual reference deck. "
        "He brings real ad work into the conversation as a brainstorm partner, asking the "
        "assistant to weigh in on layout choices, color palettes, product framing, and "
        "campaign positioning. He treats each session as a design crit, not a memorization drill."
    ),
    "traits": ["observant", "analytical", "marketing-focused", "opinionated"],
    "conversation_style": (
        "Direct and analytical. Frames each ad as a design problem, asks specific layout / "
        "color / format questions, and often draws explicit comparisons to ads discussed earlier."
    ),
}


# ----- Brand sessions -----
# Each spec has:
#   date, image_file
#   intro_user / intro_assistant — round 1 (with image)
#   extra_turns — list of (user, assistant) tuples for follow-up rounds (text only)
#
# Design takeaways are deliberately embedded in assistant text so Category I
# (design conversation continuity) questions are answerable from memory.

BRAND_SESSIONS: Dict[str, Dict[str, Any]] = {

    # ========== Coca-Cola ==========
    "CC1": {
        "date": "2024-01-05",
        "image_file": "CocaCola_1.png",
        "intro_user": (
            "Adding a Coca-Cola co-branded sponsorship piece to my reference deck. "
            "How does sponsorship visual language differ from a standalone brand ad?"
        ),
        "intro_assistant": (
            "Sponsorship ads share frame real estate with the partner brand. It's a brand "
            "awareness play that trades visual dominance for distribution reach across "
            "adjacent contexts. The cola brand becomes one signal among several rather than the lead."
        ),
        "extra_turns": [
            ("Right — sponsorships extend brand reach into contexts pure product ads cannot.",
             "Yes. The trade is reach versus ownership."),
        ],
    },

    "CC2": {
        "date": "2024-02-02",
        "image_file": "CocaCola_2.png",
        "intro_user": (
            "Continuing the cola thread — this is the Share a Coke personalization campaign. "
            "What do you make of name-personalization as a marketing mechanic?"
        ),
        "intro_assistant": (
            "Name personalization converts a mass product into a 1:1 gift. One of the cleanest "
            "examples of mass customization in CPG. The brand stays consistent but the SKU "
            "becomes intimate — that's why it generated unprecedented social shareability."
        ),
        "extra_turns": [
            ("So it's a personalization play that depends on retail variety.",
             "Yes — different in different stores, which is the whole engagement loop."),
        ],
    },

    "CC3": {
        "date": "2024-03-01",
        "image_file": "CocaCola_3.png",
        "intro_user": (
            "Lifestyle Coke ad. How does this compare to the Share-a-Coke approach we just discussed?"
        ),
        "intro_assistant": (
            "It's the inverse mode. Share-a-Coke was personalization-driven; this one is "
            "emotion-driven. Pure affect marketing — older Coca-Cola playbook. They alternate "
            "between the two depending on era."
        ),
        "extra_turns": [
            ("So Coca-Cola has at least two distinct lifestyle modes — personalization and pure emotion.",
             "Yes. They alternate between them depending on market and era."),
        ],
    },

    "CC4": {
        "date": "2024-04-01",
        "image_file": "CocaCola_4.png",
        "intro_user": (
            "Another Coke piece. The format choice here signals a different consumption "
            "occasion versus the previous ones."
        ),
        "intro_assistant": (
            "Different package formats tend to live in different contexts — convenience / "
            "on-the-go versus nostalgia / premium. Format itself becomes part of the messaging."
        ),
        "extra_turns": [
            ("So Coke uses format diversity to signal occasions.",
             "Right — format itself is part of the messaging."),
        ],
    },

    "CC5": {
        "date": "2024-05-02",
        "image_file": "CocaCola_5.png",
        "intro_user": (
            "Last Coke piece — the variant lineup ad. Does this approach actually help variant "
            "recall, or does it dilute the master brand?"
        ),
        "intro_assistant": (
            "Variant lineups lean into family resemblance. The master brand stays anchored "
            "while smaller variant cues do the disambiguation work. Classic CPG move when "
            "introducing multiple SKUs without confusing the consumer."
        ),
        "extra_turns": [
            ("So the trick is balancing consistency with differentiation.",
             "Yes — too consistent and variants vanish; too different and the brand fragments."),
        ],
    },

    # ========== Pepsi ==========
    "P1": {
        "date": "2024-01-20",
        "image_file": "Pepsi_1.png",
        "intro_user": (
            "Switching to Pepsi. Canonical product hero shot. How does Pepsi's visual language "
            "differ from the Coke ads we just discussed?"
        ),
        "intro_assistant": (
            "Pepsi tends to be product-first where Coke tends to be people-first. Less narrative, "
            "more product. Pepsi commits hard to its signature color across the entire campaign system."
        ),
        "extra_turns": [
            ("So Pepsi is product-first, Coke is people-first.",
             "Roughly yes — useful contrast for the deck."),
        ],
    },

    "P2": {
        "date": "2024-02-17",
        "image_file": "Pepsi_2.png",
        "intro_user": (
            "Another Pepsi piece — same campaign cycle but a more minimal execution. "
            "Note the typography emphasis."
        ),
        "intro_assistant": (
            "When you reduce an ad to one product plus one slogan, the typography has to do "
            "all the work. Poster treatment — readable from across a room. Trusts that brand "
            "recognition does the rest."
        ),
        "extra_turns": [
            ("Same campaign cycle, different execution mode.",
             "Yes — campaign systems usually have several executions like this."),
        ],
    },

    "P3": {
        "date": "2024-03-15",
        "image_file": "Pepsi_3.png",
        "intro_user": (
            "Special-edition Pepsi piece — limited-edition can wrap. What's the read on LE creative?"
        ),
        "intro_assistant": (
            "LE wraps are collector bait. They reward the shelf hunter who notices something "
            "different from the standard product. The base brand still anchors recognition, "
            "but the variation rewards engagement."
        ),
        "extra_turns": [
            ("So LE is a brand-engagement device, not a sales driver per se.",
             "Right — sales come from regular SKUs; LE drives PR and trial."),
        ],
    },

    "P4": {
        "date": "2024-04-15",
        "image_file": "Pepsi_4.png",
        "intro_user": (
            "Different mode for Pepsi — celebrity-style endorsement creative. "
            "What do you make of the cohesion play?"
        ),
        "intro_assistant": (
            "Visual cohesion turns the spokesperson into an extension of the product. It's one "
            "of the core tools of celebrity endorsement: subsume the celebrity into the brand "
            "system so the two become inseparable."
        ),
        "extra_turns": [
            ("So the celebrity becomes part of the brand visual identity.",
             "Yes — that's the whole game."),
        ],
    },

    "P5": {
        "date": "2024-05-18",
        "image_file": "Pepsi_5.png",
        "intro_user": (
            "Last Pepsi piece — and it breaks pattern with the format choice. "
            "What do you make of the channel shift?"
        ),
        "intro_assistant": (
            "Format choice signals consumption context. Different formats live in different "
            "retail channels — fountain belongs to QSR / foodservice, packaged versions live "
            "at retail. Format diversification IS channel diversification."
        ),
        "extra_turns": [
            ("So format diversity is channel diversity.",
             "Yes — both Coke and Pepsi juggle this carefully."),
        ],
    },

    # ========== Dunkin' ==========
    "D1": {
        "date": "2024-01-12",
        "image_file": "Dunkin_1.png",
        "intro_user": (
            "Switching to coffee. This is a Dunkin' Donuts marketing case study from Korea — "
            "experiential / sensory marketing piece, not really a poster ad."
        ),
        "intro_assistant": (
            "Right — experiential marketing case studies justify themselves with results data "
            "instead of with a hero image. Different class of ad — strategy showcase rather "
            "than product push."
        ),
        "extra_turns": [
            ("So our Dunkin' deck will include one case study and four product ads.",
             "Correct."),
        ],
    },

    "D2": {
        "date": "2024-02-05",
        "image_file": "Dunkin_2.png",
        "intro_user": (
            "Weird Dunkin' piece — minimalist food photography. Menu-expansion announcement."
        ),
        "intro_assistant": (
            "It's a menu-expansion announcement using high-end food photography. The minimalist "
            "treatment borrows from luxury food magazines. Saying 'we sell more than the core "
            "category now' while signaling premium aesthetic credentials."
        ),
        "extra_turns": [
            ("So menu expansion plus premium positioning, in one move.",
             "Yes — both at once."),
        ],
    },

    "D3": {
        "date": "2024-03-09",
        "image_file": "Dunkin_3.png",
        "intro_user": (
            "Dunkin' creative experiment — high-risk concept work, definitely Halloween-coded."
        ),
        "intro_assistant": (
            "Pure shock creative. The marketing intent is to lodge the brand in a memorable "
            "frame. Halloween imagery for food brands is high-risk because it can backfire — "
            "but when it works, it lives on in case studies."
        ),
        "extra_turns": [
            ("So this is Halloween creative for a coffee chain. Unusual move.",
             "Yes — and one that lives on in case studies because of how memorable it is."),
        ],
    },

    "D4": {
        "date": "2024-04-04",
        "image_file": "Dunkin_4.png",
        "intro_user": (
            "Another minimalist Dunkin' breakfast ad — same campaign series as the menu-expansion piece."
        ),
        "intro_assistant": (
            "Same campaign series, same visual rule. Series reward repetition — each ad "
            "introduces a different food category but the visual rule stays fixed."
        ),
        "extra_turns": [
            ("So Dunkin' has a deliberate minimalist breakfast campaign series.",
             "Yes — consistency across executions is the whole point."),
        ],
    },

    "D5": {
        "date": "2024-05-10",
        "image_file": "Dunkin_5.png",
        "intro_user": (
            "Final Dunkin' piece — recession-era promo creative."
        ),
        "intro_assistant": (
            "Promo / discount creative usually leans into either savings or joy. This one "
            "frames the discount as a happiness gesture rather than a desperation discount. "
            "Pricing psychology dressed up as warmth."
        ),
        "extra_turns": [
            ("Smart — pricing psychology dressed up as happiness marketing.",
             "Yes — softer take on a hard discount."),
        ],
    },

    # ========== Starbucks ==========
    "S1": {
        "date": "2024-01-22",
        "image_file": "Starbucks_1.png",
        "intro_user": (
            "Switching to Starbucks. This is the Ethos Water cause-marketing piece. Note this "
            "is the only Starbucks piece in our deck off the core product."
        ),
        "intro_assistant": (
            "Right — CSR / cause-marketing, deliberately positioned away from the core product. "
            "The brand becomes the underwriter, not the hero. Worth flagging that this Ethos "
            "Water piece is the visual outlier in our 5-Starbucks set."
        ),
        "extra_turns": [
            ("Outlier noted.",
             "Correct."),
        ],
    },

    "S2": {
        "date": "2024-02-20",
        "image_file": "Starbucks_2.png",
        "intro_user": (
            "Standard Starbucks lifestyle ad — comforting mode."
        ),
        "intro_assistant": (
            "Yes — Starbucks comforting / lifestyle mode. Sells atmosphere, not coffee. "
            "Reliable, almost over-familiar."
        ),
        "extra_turns": [
            ("So this is the comforting mode of Starbucks creative.",
             "Right."),
        ],
    },

    "S3": {
        "date": "2024-03-22",
        "image_file": "Starbucks_3.png",
        "intro_user": (
            "Very different Starbucks ad — dramatic mode. How does this compare to the comforting "
            "mode we just discussed?"
        ),
        "intro_assistant": (
            "It's a deliberate departure — call it the dramatic / cinematic mode. The standard "
            "Starbucks lifestyle aesthetic sells comfort. This one sells craft and intensity. "
            "Starbucks has TWO distinct visual modes: comforting and dramatic."
        ),
        "extra_turns": [
            ("Two modes confirmed.",
             "Right. Catalog them as 'Starbucks lifestyle' and 'Starbucks craft'."),
        ],
    },

    "S4": {
        "date": "2024-04-19",
        "image_file": "Starbucks_4.png",
        "intro_user": (
            "Starbucks Brazil 'Coffee Braille' accessibility creative."
        ),
        "intro_assistant": (
            "Accessibility creative — CSR move in the same family as the Ethos Water piece, "
            "but executed as a tangible service innovation. Different angle on cause-marketing."
        ),
        "extra_turns": [
            ("So Starbucks has done CSR creative twice in our deck.",
             "Yes — two CSR plays, two different angles."),
        ],
    },

    "S5": {
        "date": "2024-05-22",
        "image_file": "Starbucks_5.png",
        "intro_user": (
            "Final Starbucks — seasonal blend lineup ad. Structurally similar to the Coca-Cola "
            "variant lineup we discussed earlier."
        ),
        "intro_assistant": (
            "Same idea — variant family laid out as a grid. Both pick the same readable count "
            "for the same reason: it's the variant-lineup sweet spot."
        ),
        "extra_turns": [
            ("So both Coke and Starbucks use lineup compositions for variants.",
             "Yes — useful pattern."),
        ],
    },

    # ========== McDonald's ==========
    "M1": {
        "date": "2024-01-14",
        "image_file": "McDonalds_1.png",
        "intro_user": (
            "Switching to fast food. McDonald's healthy-positioning campaign. "
            "What's McDonald's trying to do here?"
        ),
        "intro_assistant": (
            "Healthy positioning. McDonald's promoting healthier items is an uphill messaging "
            "fight, so they pair the product with active-lifestyle imagery to associate it with "
            "aspiration rather than fast-food guilt. Counterprogramming against the burger association."
        ),
        "extra_turns": [
            ("So this is McDonald's healthy-positioning play.",
             "Right — counterprogramming against the burger association."),
        ],
    },

    "M2": {
        "date": "2024-02-08",
        "image_file": "McDonalds_2.png",
        "intro_user": (
            "McDonald's heritage piece."
        ),
        "intro_assistant": (
            "Heritage / nostalgia play. Frames McDonald's as a brand with menu depth and "
            "history. Useful when the brand wants to remind consumers it has been around forever — "
            "competing on permanence rather than novelty."
        ),
        "extra_turns": [
            ("Heritage / nostalgia move.",
             "Yes."),
        ],
    },

    "M3": {
        "date": "2024-03-05",
        "image_file": "McDonalds_3.png",
        "intro_user": (
            "Surreal McDonald's piece — product reinterpretation."
        ),
        "intro_assistant": (
            "The design choice is a literal interpretation of the product name. Single-image "
            "visual pun that lodges the product name in memory. McDonald's does this kind of "
            "creative product reinterpretation well."
        ),
        "extra_turns": [
            ("So McDonald's does product reinterpretations in their creative.",
             "Yes — there are more in the queue."),
        ],
    },

    "M4": {
        "date": "2024-04-08",
        "image_file": "McDonalds_4.png",
        "intro_user": (
            "McDonald's Cannes ad — self-congratulatory creative-award acknowledgment."
        ),
        "intro_assistant": (
            "It's an industry self-pat-on-the-back ad, dressed up as creative bravado. Awards "
            "ads are unusual — they're for the industry, not the consumer."
        ),
        "extra_turns": [
            ("So this is McDonald's celebrating its own creative recognition.",
             "Yes."),
        ],
    },

    "M5": {
        "date": "2024-05-06",
        "image_file": "McDonalds_5.png",
        "intro_user": (
            "Wi-Fi-themed McDonald's ad — service-as-feature creative."
        ),
        "intro_assistant": (
            "Free wifi is a service, the brand is the icon, combining them turns infrastructure "
            "into a brand attribute. Service-oriented ads are unusual for QSR — most ads are "
            "product-first."
        ),
        "extra_turns": [
            ("So McDonald's expanding into service messaging.",
             "Right — useful angle to catalog."),
        ],
    },

    # ========== Burger King ==========
    "BK1": {
        "date": "2024-01-08",
        "image_file": "BurgerKing_1.png",
        "intro_user": (
            "Switching to Burger King. Fashion crossover ad."
        ),
        "intro_assistant": (
            "Fashion-crossover play — Burger King's brand iconography pulled into a fashion "
            "context. Fashion crossovers are unusual for QSR; they aim for cultural relevance "
            "rather than direct sales."
        ),
        "extra_turns": [
            ("So brand iconography pulled into fashion territory.",
             "Yes — most unusual BK creative in our deck."),
        ],
    },

    "BK2": {
        "date": "2024-02-14",
        "image_file": "BurgerKing_2.png",
        "intro_user": (
            "BK breakfast launch ad."
        ),
        "intro_assistant": (
            "Standard menu-launch ad. New products, headline that calls lapsed customers back. "
            "Breakfast is BK's perennial growth lever — they keep launching breakfast SKUs."
        ),
        "extra_turns": [
            ("So this is BK's breakfast strategy.",
             "Yes — recurring theme."),
        ],
    },

    "BK3": {
        "date": "2024-03-18",
        "image_file": "BurgerKing_3.png",
        "intro_user": (
            "Brazilian BK ad — phonetic typography play for Carioca dialect."
        ),
        "intro_assistant": (
            "The headline does the entire creative work — phonetically renders the regional "
            "accent applied to the brand name. Pure regional / market-specific creative that "
            "works in Portuguese only."
        ),
        "extra_turns": [
            ("So this is BK doing localized phonetic creative.",
             "Yes — regional ads have very different tones."),
        ],
    },

    "BK4": {
        "date": "2024-04-23",
        "image_file": "BurgerKing_4.png",
        "intro_user": (
            "BK Halloween limited edition."
        ),
        "intro_assistant": (
            "Seasonal LE play. Limited editions are a classic seasonal move because the visual "
            "departure from the default is shocking enough to drive trial. The campaign rises "
            "and falls on a single bold execution choice."
        ),
        "extra_turns": [
            ("So this is BK seasonal LE.",
             "Right."),
        ],
    },

    "BK5": {
        "date": "2024-05-26",
        "image_file": "BurgerKing_5.png",
        "intro_user": (
            "Final BK piece — surreal creative."
        ),
        "intro_assistant": (
            "Surreal composition — visual whimsy. Almost the inverse of McDonald's surreal "
            "product ads, where the product was substituted for something else; here the "
            "source ingredient becomes the creative. Different surreal mode."
        ),
        "extra_turns": [
            ("So our BK deck has unusual creative experiments.",
             "Right — surreal and fashion both make appearances."),
        ],
    },
}


# ----- Cross-brand distractor sessions (text-only false claims) -----
CROSS_BRAND_SESSIONS: Dict[str, Dict[str, Any]] = {
    "CX1": {
        "date": "2024-01-16",
        "dialogues": [
            ("I'm reconciling the cola sponsorship examples in my reference deck. The soccer goalkeeper piece feels like it could belong to either Coca-Cola or Pepsi. Which side should I file it under?",
             "That kind of sports-sponsorship creative is easy to mix up between the two cola brands. We should resolve it against the original reference, not by gut memory."),
        ],
    },
    "CX2": {
        "date": "2024-01-24",
        "dialogues": [
            ("I'm grouping the red-background brand ads together and hesitated on the Starbucks Ethos Water heart-shaped pour piece. Does that one really belong in the red-background bucket?",
             "That is exactly the kind of visual trait that gets blurred when several brands use strong color blocking. Better to check the original image than trust the bucket label."),
        ],
    },
    "CX3": {
        "date": "2024-02-10",
        "dialogues": [
            ("I'm reviewing the healthy-positioning fast-food examples. The salad ad with the running woman could fit more than one chain in memory. Which brand should that one stay under?",
             "Active-lifestyle salad creative is easy to overgeneralize across chains. The safest move is to anchor it to the original campaign image, not the vibe alone."),
        ],
    },
    "CX4": {
        "date": "2024-02-22",
        "dialogues": [
            ("I'm sorting the Pepsi references by background treatment. The minimalist single-can 'REFRESH YOUR WORLD' piece feels brand-blue to me, but I want to verify it before I tag it. Is that instinct right?",
             "Pepsi often leans blue, but background recall is still easy to overconfidently compress. We should verify that specific creative against the source image."),
        ],
    },
    "CX5": {
        "date": "2024-03-11",
        "dialogues": [
            ("I'm comparing surreal fast-food visualizations. The goldfish-burger image could plausibly sit in either a Burger King or McDonald's folder at first glance. Which brand should it actually live under?",
             "That kind of surreal food visual is exactly where cross-brand confusion happens. We need the original reference rather than a loose category memory."),
        ],
    },
    "CX6": {
        "date": "2024-03-24",
        "dialogues": [
            ("I'm separating the dramatic coffee ads from the playful ones. The espresso glass with sparks has a moody style that could fit Starbucks or Dunkin'. Which deck should it stay in?",
             "Dark, stylized coffee imagery can collapse together in memory. Better to ground it in the actual saved creative than the overall aesthetic."),
        ],
    },
    "CX7": {
        "date": "2024-04-10",
        "dialogues": [
            ("I'm tagging the Cannes-style McDonald's references by background color. The fries-as-fireworks piece is the one I'm least sure about. Should I file it with the red-background work or not?",
             "Award-style food ads can be especially easy to flatten into the same color memory. We should verify the actual frame instead of relying on recall."),
        ],
    },
    "CX8": {
        "date": "2024-04-26",
        "dialogues": [
            ("I'm reviewing the unusual seasonal burger concepts. The black-bun Halloween piece feels like something either Burger King or McDonald's could have run. Which brand should I keep it under?",
             "Both chains have enough Halloween or stunt-food history to make that an easy mix-up. We should anchor it to the original brand example."),
        ],
    },
    "CX9": {
        "date": "2024-05-13",
        "dialogues": [
            ("I'm organizing the Halloween snack references. The spider-shaped piece looks like something that could belong to multiple chains at a glance. Which brand was it actually tied to?",
             "Seasonal novelty food creative is another place where category memory can overpower source memory. We should resolve it from the saved ad itself."),
        ],
    },
    "CX10": {
        "date": "2024-05-30",
        "dialogues": [
            ("I'm cleaning up the personalized-bottle examples in the Coke folder. I want to make sure I have the right pair of names on that 'Share a Coke with a friend' image. Can we verify the names before I lock the note?",
             "Name-personalization campaigns vary across executions and regions, so that is not something I would trust from memory alone."),
        ],
    },
    "CX11": {
        "date": "2024-05-04",
        "dialogues": [
            ("I'm annotating the Coca-Cola variant-lineup piece and want to record the bottle count correctly. Before I finalize the note, can we verify how many bottles were actually shown in the row?",
             "Lineup counts are easy to overstate when several bottles are arranged symmetrically. We should read it off the actual creative rather than memory."),
        ],
    },
    "CX12": {
        "date": "2024-03-25",
        "dialogues": [
            ("I'm reviewing the Starbucks executions with cups and glassware. For the 'Buy Me a Coffee' piece, I want to make sure I'm labeling the container correctly before I summarize it. Can we verify that detail?",
             "Starbucks uses paper cups, mugs, and glassware across very different campaign styles, so the container type is worth checking against the original image."),
        ],
    },
}


# ----- Session interleaving plan -----
SESSION_PLAN: List[str] = [
    "CC1",  # 01-05
    "BK1",  # 01-08
    "D1",   # 01-12
    "M1",   # 01-14
    "CX1",  # 01-16
    "P1",   # 01-20
    "S1",   # 01-22
    "CX2",  # 01-24
    "CC2",  # 02-02
    "D2",   # 02-05
    "M2",   # 02-08
    "CX3",  # 02-10
    "BK2",  # 02-14
    "P2",   # 02-17
    "S2",   # 02-20
    "CX4",  # 02-22
    "CC3",  # 03-01
    "M3",   # 03-05
    "D3",   # 03-09
    "CX5",  # 03-11
    "P3",   # 03-15
    "BK3",  # 03-18
    "S3",   # 03-22
    "CX6",  # 03-24
    "CX12", # 03-25
    "CC4",  # 04-01
    "D4",   # 04-04
    "M4",   # 04-08
    "CX7",  # 04-10
    "P4",   # 04-15
    "S4",   # 04-19
    "BK4",  # 04-23
    "CX8",  # 04-26
    "CC5",  # 05-02
    "CX11", # 05-04
    "M5",   # 05-06
    "D5",   # 05-10
    "CX9",  # 05-13
    "P5",   # 05-18
    "S5",   # 05-22
    "BK5",  # 05-26
    "CX10", # 05-30
]


# ----- Candidate MCQ QAs (v3 — harder distribution) -----
#
# Removed (too easy for FC):
#   A: cross-brand visual disambiguation (FC has all images, easy lookup)
#   B: X4 micro-attributes (FC sees the image directly)
#   C: spatial grounding (same)
#   I: design conversation continuity (assistant text is in FC's context)
#
# Kept and expanded:
#   D (12): false memory rejection
#   E (8):  temporal ordering
#   F (15): set-level aggregation within a brand (12 + 3 conjunctive)
#   G (12): cross-brand comparative (8 + 4 intersection)
#   H (8):  anomaly / one-of detection
# New:
#   J (6):  negative counting ("how many did NOT show X")
#   K (6):  multi-step reasoning across categories

QA_ITEMS: List[Dict[str, Any]] = [
    # ===== Category D: false memory (12) =====
    {"point": [["X2"], ["Y3"]],
     "question": "When I revisited the cola sponsorship examples, which brand did the soccer goalkeeper image actually belong to?",
     "options": {"A": "Pepsi", "B": "Coca-Cola", "C": "There was no goalkeeper ad", "D": "It was Sprite"},
     "answer": "B", "session_id": ["CC1", "CX1"], "clue": ["CC1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "For the Starbucks Ethos Water 'heart-shaped pour' ad, which background description matches the original image?",
     "options": {"A": "There was no Ethos Water ad", "B": "Red background", "C": "White background", "D": "Black background"},
     "answer": "C", "session_id": ["S1", "CX2"], "clue": ["S1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "In the healthy-positioning fast-food set, which brand did the salad-with-running-woman ad actually belong to?",
     "options": {"A": "McDonald's", "B": "Subway", "C": "Burger King", "D": "Wendy's"},
     "answer": "A", "session_id": ["M1", "CX3"], "clue": ["M1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "For the minimalist single-can Pepsi 'REFRESH YOUR WORLD' ad, which background description is correct?",
     "options": {"A": "Red", "B": "Black", "C": "Blue", "D": "Yellow"},
     "answer": "C", "session_id": ["P2", "CX4"], "clue": ["P2:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "The surreal goldfish-burger visual belonged to which brand's campaign?",
     "options": {"A": "Burger King", "B": "Long John Silver's", "C": "McDonald's (Filet-O-Fish)", "D": "Captain D's"},
     "answer": "C", "session_id": ["M3", "CX5"], "clue": ["M3:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "The espresso-glass-with-sparks ad belonged to which coffee brand?",
     "options": {"A": "Starbucks", "B": "Dunkin'", "C": "Nespresso", "D": "Lavazza"},
     "answer": "A", "session_id": ["S3", "CX6"], "clue": ["S3:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "For the McDonald's fries-as-fireworks Cannes ad, which background description is correct?",
     "options": {"A": "Blue", "B": "Black", "C": "White", "D": "Red"},
     "answer": "D", "session_id": ["M4", "CX7"], "clue": ["M4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "The 'Halloween' black-bun burger creative belonged to which brand?",
     "options": {"A": "Wendy's", "B": "McDonald's", "C": "Burger King", "D": "Carl's Jr"},
     "answer": "C", "session_id": ["BK4", "CX8"], "clue": ["BK4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "The spider-shaped snack ad belonged to which brand?",
     "options": {"A": "McDonald's", "B": "Dunkin'", "C": "Burger King", "D": "KFC"},
     "answer": "B", "session_id": ["D3", "CX9"], "clue": ["D3:1"]},
    {"point": [["X4"], ["Y3"]],
     "question": "Which pair of names appeared on the Share-a-Coke bottles in that image?",
     "options": {"A": "Mike and Sarah", "B": "Kylie and Luke", "C": "John and Mary", "D": "There were no names"},
     "answer": "B", "session_id": ["CC2", "CX10"], "clue": ["CC2:1"]},
    {"point": [["X4"], ["Y3"]],
     "question": "How many bottles were actually shown in the Coca-Cola variant-lineup row?",
     "options": {"A": "Five bottles", "B": "Six bottles", "C": "Three bottles", "D": "Four bottles"},
     "answer": "D", "session_id": ["CC5", "CX11"], "clue": ["CC5:1"]},
    {"point": [["X4"], ["Y3"]],
     "question": "What kind of container was shown in the Starbucks 'Buy Me a Coffee' ad?",
     "options": {"A": "A travel tumbler", "B": "A green ceramic mug", "C": "A tall paper cup", "D": "A small clear espresso glass"},
     "answer": "D", "session_id": ["S3", "CX12"], "clue": ["S3:1"]},

    # ===== Category E: temporal (8) =====
    {"point": [["X2"], ["Y3"]],
     "question": "Did the Coca-Cola Citi soccer sponsorship ad come BEFORE or AFTER the Burger King 'crown' high-heels ad?",
     "options": {"A": "Before the BK crown ad", "B": "Neither was shown", "C": "On the same day", "D": "After the BK crown ad"},
     "answer": "A", "session_id": ["CC1", "BK1"], "clue": ["CC1:1", "BK1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Did the Dunkin' '25 cent donut' coin ad come BEFORE or AFTER the Starbucks 'Coffee Braille' accessibility ad?",
     "options": {"A": "Neither was shown", "B": "Same day", "C": "After", "D": "Before"},
     "answer": "C", "session_id": ["D5", "S4"], "clue": ["D5:1", "S4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Which brand was shown in the very FIRST ad of the entire conversation?",
     "options": {"A": "Burger King", "B": "Coca-Cola", "C": "Starbucks", "D": "Pepsi"},
     "answer": "B", "session_id": ["CC1"], "clue": ["CC1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Which brand was shown in the very LAST ad of the entire conversation?",
     "options": {"A": "Dunkin'", "B": "Burger King", "C": "Pepsi", "D": "Coca-Cola"},
     "answer": "B", "session_id": ["BK5"], "clue": ["BK5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Did we discuss the Pepsi camouflage-pattern can BEFORE or AFTER the McDonald's surreal goldfish-burger?",
     "options": {"A": "Before the goldfish-burger", "B": "Never discussed", "C": "Same day", "D": "After the goldfish-burger"},
     "answer": "D", "session_id": ["P3", "M3"], "clue": ["P3:1", "M3:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Among Coca-Cola's five ads, which was the LAST one shown chronologically?",
     "options": {"A": "Citi soccer sponsorship", "B": "Happiness couple", "C": "Share a Coke with Kylie and Luke", "D": "Good things come in fours (4-bottle lineup)"},
     "answer": "D", "session_id": ["CC5"], "clue": ["CC5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 6 brands, which was discussed FIRST chronologically (earliest first ad date)?",
     "options": {"A": "Pepsi", "B": "Dunkin'", "C": "Coca-Cola", "D": "McDonald's"},
     "answer": "C", "session_id": ["CC1"], "clue": ["CC1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Did the Burger King Halloween Whopper come BEFORE or AFTER the McDonald's Wi-Fi fries ad?",
     "options": {"A": "Same day", "B": "After the Wi-Fi fries ad", "C": "Never shown", "D": "Before the Wi-Fi fries ad"},
     "answer": "D", "session_id": ["BK4", "M5"], "clue": ["BK4:1", "M5:1"]},

    # ===== Category F: set-level aggregation within brand (12) =====
    {"point": [["X2"], ["Y3"]],
     "question": "Across all 5 Coca-Cola ads, how many used a SOLID RED background?",
     "options": {"A": "1", "B": "3", "C": "2", "D": "4"},
     "answer": "B", "session_id": ["CC3", "CC4", "CC5"], "clue": ["CC3:1", "CC4:1", "CC5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Pepsi ads, how many showed exactly ONE Pepsi can (not multiple, not zero)?",
     "options": {"A": "5", "B": "3", "C": "2", "D": "4"},
     "answer": "B", "session_id": ["P2", "P3", "P4"], "clue": ["P2:1", "P3:1", "P4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across the 5 Dunkin' ads, how many had FOOD as the visual centerpiece (not coffee, not a multi-panel layout)?",
     "options": {"A": "5", "B": "2", "C": "3", "D": "4"},
     "answer": "D", "session_id": ["D2", "D3", "D4", "D5"], "clue": ["D2:1", "D3:1", "D4:1", "D5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Starbucks ads, how many actually showed COFFEE (cup, drink, espresso)?",
     "options": {"A": "4", "B": "5", "C": "3", "D": "2"},
     "answer": "A", "session_id": ["S2", "S3", "S4", "S5"], "clue": ["S2:1", "S3:1", "S4:1", "S5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across the 5 McDonald's ads, how many used a RED or red-orange background?",
     "options": {"A": "4", "B": "5", "C": "3", "D": "2"},
     "answer": "C", "session_id": ["M1", "M4", "M5"], "clue": ["M1:1", "M4:1", "M5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Burger King ads, how many had an actual BURGER as the primary subject?",
     "options": {"A": "2", "B": "5", "C": "4", "D": "3"},
     "answer": "D", "session_id": ["BK2", "BK3", "BK4"], "clue": ["BK2:1", "BK3:1", "BK4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across the 5 Coca-Cola ads, how many showed a person's FACE in the frame?",
     "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
     "answer": "C", "session_id": ["CC1", "CC3", "CC4"], "clue": ["CC1:1", "CC3:1", "CC4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Pepsi ads, how many used a DARK BLUE background (not light blue, not red)?",
     "options": {"A": "4", "B": "2", "C": "5", "D": "3"},
     "answer": "D", "session_id": ["P1", "P2", "P3"], "clue": ["P1:1", "P2:1", "P3:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across the 5 Starbucks ads, how many were minimalist single-product / single-subject shots (no humans visible)?",
     "options": {"A": "2", "B": "4", "C": "5", "D": "3"},
     "answer": "D", "session_id": ["S1", "S2", "S3"], "clue": ["S1:1", "S2:1", "S3:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 McDonald's ads, how many used a SURREAL product visualization (e.g., goldfish-burger, fries-fireworks, fries-as-Wi-Fi)?",
     "options": {"A": "2", "B": "4", "C": "1", "D": "3"},
     "answer": "D", "session_id": ["M3", "M4", "M5"], "clue": ["M3:1", "M4:1", "M5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across all 5 Burger King ads, how many had a NON-burger primary subject (animal, fashion, etc.)?",
     "options": {"A": "4", "B": "3", "C": "1", "D": "2"},
     "answer": "D", "session_id": ["BK1", "BK5"], "clue": ["BK1:1", "BK5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "In the 5 Dunkin' ads, how many used a WHITE background?",
     "options": {"A": "1", "B": "2", "C": "0", "D": "3"},
     "answer": "B", "session_id": ["D2", "D4"], "clue": ["D2:1", "D4:1"]},
    {"point": [["X2", "X4"], ["Y3"]],
     "question": "Of the 5 Coca-Cola ads, how many showed BOTH a person's face AND a SOLID RED background?",
     "options": {"A": "2", "B": "4", "C": "1", "D": "3"},
     "answer": "A", "session_id": ["CC3", "CC4"], "clue": ["CC3:1", "CC4:1"]},
    {"point": [["X2", "X4"], ["Y3"]],
     "question": "Of the 5 Pepsi ads, how many had EXACTLY ONE Pepsi can AND a DARK BLUE background?",
     "options": {"A": "2", "B": "3", "C": "4", "D": "1"},
     "answer": "A", "session_id": ["P2", "P3"], "clue": ["P2:1", "P3:1"]},
    {"point": [["X2", "X4"], ["Y3"]],
     "question": "Across the 5 McDonald's ads, how many had BOTH a red background AND a surreal product visualization (e.g., fries-fireworks, Wi-Fi-fries)?",
     "options": {"A": "4", "B": "3", "C": "2", "D": "1"},
     "answer": "C", "session_id": ["M4", "M5"], "clue": ["M4:1", "M5:1"]},

    # ===== Category G: cross-brand comparative (8) =====
    {"point": [["X2"], ["Y3"]],
     "question": "Comparing Coca-Cola vs Pepsi, which brand had MORE ads featuring a person's face?",
     "options": {"A": "Pepsi", "B": "Equal counts", "C": "Neither showed any faces", "D": "Coca-Cola"},
     "answer": "D", "session_id": ["CC1", "CC3", "CC4", "P4"], "clue": ["CC1:1", "P4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Between McDonald's and Burger King, which brand had MORE surreal product visualizations (e.g., goldfish, fries-fireworks, fries-as-Wi-Fi, cow tower)?",
     "options": {"A": "Burger King", "B": "Equal", "C": "Neither", "D": "McDonald's"},
     "answer": "D", "session_id": ["M3", "M4", "M5", "BK5"], "clue": ["M3:1", "M4:1", "M5:1", "BK5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Which TWO brands had EXACTLY 3 red-background ads each in our deck?",
     "options": {"A": "Pepsi and McDonald's", "B": "Coca-Cola and McDonald's", "C": "Coca-Cola and Burger King", "D": "Dunkin' and Starbucks"},
     "answer": "B", "session_id": ["CC3", "CC4", "CC5", "M1", "M4", "M5"], "clue": ["CC5:1", "M5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Among the 6 brands, which brand's 5 ads showed the WIDEST variety of background colors (5 visibly different backgrounds)?",
     "options": {"A": "Coca-Cola", "B": "Pepsi", "C": "Starbucks", "D": "Burger King"},
     "answer": "D", "session_id": ["BK1", "BK2", "BK3", "BK4", "BK5"], "clue": ["BK5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Between Starbucks and Dunkin', which brand had MORE minimalist single-subject product photography?",
     "options": {"A": "Neither", "B": "Dunkin'", "C": "Starbucks", "D": "Equal counts"},
     "answer": "C", "session_id": ["S1", "S2", "S3", "D2", "D4"], "clue": ["S1:1", "D2:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Comparing Coca-Cola and Pepsi, which brand showed MORE distinct product formats (glass bottle vs plastic bottle vs can vs cup)?",
     "options": {"A": "Neither showed multiple formats", "B": "Coca-Cola", "C": "Equal", "D": "Pepsi"},
     "answer": "B", "session_id": ["CC1", "CC2", "CC4", "P1", "P5"], "clue": ["CC4:1", "P5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across all 30 ads, how many showed a person actively HOLDING the product in their hands?",
     "options": {"A": "5", "B": "2", "C": "4", "D": "3"},
     "answer": "C", "session_id": ["CC2", "CC3", "CC4", "P4"], "clue": ["CC2:1", "P4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 6 brands in our discussion, how many distinct PRODUCT CATEGORIES did they cover (cola / coffee / fast food etc.)?",
     "options": {"A": "5", "B": "4", "C": "3", "D": "2"},
     "answer": "C", "session_id": ["CC1", "P1", "D1", "S1", "M1", "BK1"], "clue": ["CC1:1"]},
    {"point": [["X2", "X4"], ["Y3"]],
     "question": "Across all 30 ads, how many used a SOLID RED background AND showed a person's face in the same frame?",
     "options": {"A": "2", "B": "3", "C": "4", "D": "1"},
     "answer": "A", "session_id": ["CC3", "CC4"], "clue": ["CC3:1", "CC4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across all 30 ads, how many ads showed a person's face clearly visible in the frame?",
     "options": {"A": "4", "B": "6", "C": "5", "D": "3"},
     "answer": "A", "session_id": ["CC1", "CC3", "CC4", "P4"], "clue": ["CC1:1", "P4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Among the 6 brands, how many had ZERO ads in their 5-ad set with a person's face clearly visible?",
     "options": {"A": "3", "B": "1", "C": "4", "D": "2"},
     "answer": "A", "session_id": ["D1", "D2", "D3", "D4", "D5", "S1", "S2", "S3", "S5", "BK2", "BK3", "BK4", "BK5"], "clue": ["D1:1", "S1:1", "BK5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 6 brands, which had the MOST ads using its brand signature color as the primary background?",
     "options": {"A": "Dunkin' (orange/pink)", "B": "Pepsi (blue)", "C": "Coca-Cola (red)", "D": "Starbucks (green)"},
     "answer": "B", "session_id": ["P1", "P2", "P3", "P4"], "clue": ["P1:1", "P2:1", "P3:1"]},

    # ===== Category H: anomaly / one-of detection (8) =====
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Starbucks ads, only ONE featured a NON-coffee product. What product was it?",
     "options": {"A": "Tea", "B": "Frappuccino", "C": "Ethos Water", "D": "A pastry"},
     "answer": "C", "session_id": ["S1"], "clue": ["S1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Among the 5 McDonald's ads, only ONE showed MULTIPLE menu items lined up in a row. Which campaign was that?",
     "options": {"A": "Historical menu timeline", "B": "WiFi fries", "C": "Filet-O-Fish goldfish", "D": "Cannes fries fireworks"},
     "answer": "A", "session_id": ["M2"], "clue": ["M2:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Burger King ads, only ONE had a HUMAN element in the frame. Which one?",
     "options": {"A": "Cow tower stack", "B": "Crown around legs in red high heels", "C": "Halloween Whopper black bun", "D": "BK Breakfast Bowl + Muffin"},
     "answer": "B", "session_id": ["BK1"], "clue": ["BK1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Pepsi ads, only ONE was NOT a can-focused product shot. Which one?",
     "options": {"A": "Live for Now fountain cup", "B": "Single can REFRESH YOUR WORLD slogan", "C": "Camo-pattern can", "D": "Three-can hero shot"},
     "answer": "A", "session_id": ["P5"], "clue": ["P5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Coca-Cola ads, only ONE showed a multi-bottle variant lineup. Which one?",
     "options": {"A": "Good things come in fours", "B": "Citi soccer sponsorship", "C": "Happiness couple", "D": "Share a Coke (Kylie / Luke)"},
     "answer": "A", "session_id": ["CC5"], "clue": ["CC5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Dunkin' ads, only ONE was a multi-panel marketing case study. Which one?",
     "options": {"A": "Spider-shaped snack Halloween", "B": "Tomato + cheese pizza minimal", "C": "Flavor Radio (Korea)", "D": "Single bacon strip"},
     "answer": "C", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 BK ads, only ONE had an ANIMAL as the primary visual subject. Which one?",
     "options": {"A": "Cow tower stack", "B": "BURRRGERRR KING Whopper", "C": "Halloween Whopper", "D": "Crown around legs"},
     "answer": "A", "session_id": ["BK5"], "clue": ["BK5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 McDonald's ads, only ONE used a NON-red background as its primary backdrop. Which one?",
     "options": {"A": "Historical menu timeline", "B": "Filet-O-Fish goldfish", "C": "Salad with running woman", "D": "WiFi fries"},
     "answer": "B", "session_id": ["M3"], "clue": ["M3:1"]},

    # ===== Category J: negative counting (6) =====
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Pepsi ads, how many did NOT show a Pepsi can?",
     "options": {"A": "0", "B": "3", "C": "1", "D": "2"},
     "answer": "C", "session_id": ["P5"], "clue": ["P5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across the 5 Burger King ads, how many had NEITHER a burger NOR breakfast item as the primary subject?",
     "options": {"A": "1", "B": "2", "C": "0", "D": "3"},
     "answer": "B", "session_id": ["BK1", "BK5"], "clue": ["BK1:1", "BK5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 McDonald's ads, how many did NOT use a red or red-orange background?",
     "options": {"A": "2", "B": "0", "C": "1", "D": "3"},
     "answer": "A", "session_id": ["M2", "M3"], "clue": ["M2:1", "M3:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Across the 5 Dunkin' ads, how many did NOT have food as the visual centerpiece?",
     "options": {"A": "0", "B": "1", "C": "3", "D": "2"},
     "answer": "B", "session_id": ["D1"], "clue": ["D1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Starbucks ads, how many did NOT show a coffee cup or coffee drink?",
     "options": {"A": "0", "B": "1", "C": "2", "D": "3"},
     "answer": "B", "session_id": ["S1"], "clue": ["S1:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the 5 Coca-Cola ads, how many did NOT show a person's face clearly?",
     "options": {"A": "4", "B": "3", "C": "2", "D": "1"},
     "answer": "C", "session_id": ["CC2", "CC5"], "clue": ["CC2:1", "CC5:1"]},

    # ===== Category K: multi-step reasoning (6) =====
    {"point": [["X2"], ["Y3"]],
     "question": "Of McDonald's three surreal ads (goldfish-burger, fries-fireworks, Wi-Fi-fries), which one came in the MIDDLE chronologically?",
     "options": {"A": "All three on the same date", "B": "Fries-as-fireworks (Cannes)", "C": "Fries-as-Wi-Fi", "D": "Goldfish-burger (Filet-O-Fish)"},
     "answer": "B", "session_id": ["M3", "M4", "M5"], "clue": ["M3:1", "M4:1", "M5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Which brand had the ad whose background was easy to misremember as black even though the original frame was red?",
     "options": {"A": "McDonald's", "B": "Burger King", "C": "Pepsi", "D": "Starbucks"},
     "answer": "A", "session_id": ["M4", "CX7"], "clue": ["M4:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of Coca-Cola's three RED-background ads, which one came LAST chronologically?",
     "options": {"A": "Good things come in fours (variant lineup)", "B": "Perfection (ice cold)", "C": "Happiness couple", "D": "Citi soccer sponsorship"},
     "answer": "A", "session_id": ["CC3", "CC4", "CC5"], "clue": ["CC5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Among the two Starbucks follow-up checks, which ad was the one where the key uncertainty was the container type rather than the background?",
     "options": {"A": "Ethos Water heart pour", "B": "Best coffee paper cup on table", "C": "Coffee Braille accessibility", "D": "Buy Me a Coffee espresso glass with sparks"},
     "answer": "D", "session_id": ["S3", "CX12"], "clue": ["S3:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Of the THREE McDonald's ads with red backgrounds, which one was the EARLIEST chronologically?",
     "options": {"A": "Cannes fries fireworks", "B": "Wi-Fi fries", "C": "Salad with running woman", "D": "Filet-O-Fish goldfish"},
     "answer": "C", "session_id": ["M1", "M4", "M5"], "clue": ["M1:1", "M4:1", "M5:1"]},
    {"point": [["X2"], ["Y3"]],
     "question": "Which follow-up note was specifically about verifying the bottle count in a lineup ad?",
     "options": {"A": "Coca-Cola four-bottle variant lineup", "B": "Pepsi 3-can hero shot", "C": "Burger King breakfast pair", "D": "Starbucks 4-mug seasonal grid"},
     "answer": "A", "session_id": ["CC5", "CX11"], "clue": ["CC5:1"]},
]


# ----- Builders -----

def _build_brand_session(session_id: str) -> Dict[str, Any]:
    spec = BRAND_SESSIONS[session_id]
    dialogues = [
        {
            "round": f"{session_id}:1",
            "user": spec["intro_user"],
            "assistant": spec["intro_assistant"],
            "input_image": [f"../image/{IMAGE_DIR}/{spec['image_file']}"],
            "image_id": [f"{session_id}:IMG_001"],
        }
    ]
    for idx, (user_text, assistant_text) in enumerate(spec.get("extra_turns", []), start=2):
        dialogues.append({
            "round": f"{session_id}:{idx}",
            "user": user_text,
            "assistant": assistant_text,
        })
    return {
        "session_id": session_id,
        "date": spec["date"],
        "dialogues": dialogues,
    }


def _build_cx_session(session_id: str) -> Dict[str, Any]:
    spec = CROSS_BRAND_SESSIONS[session_id]
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
        "dialogues": dialogues,
    }


def build_dataset() -> Dict[str, Any]:
    ordered_sessions: List[Dict[str, Any]] = []
    for session_id in SESSION_PLAN:
        if session_id in BRAND_SESSIONS:
            ordered_sessions.append(_build_brand_session(session_id))
        elif session_id in CROSS_BRAND_SESSIONS:
            ordered_sessions.append(_build_cx_session(session_id))
        else:
            raise ValueError(f"Unknown session id: {session_id}")
    all_qas = copy.deepcopy(QA_ITEMS)
    if SELECTED_QA_INDICES is not None:
        all_qas = [all_qas[i - 1] for i in SELECTED_QA_INDICES]
    return {
        "character_profile": copy.deepcopy(CHARACTER_PROFILE),
        "multi_session_dialogues": ordered_sessions,
        "human-annotated QAs": all_qas,
    }


def main() -> None:
    data = build_dataset()
    OUTPUT_DIALOG.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DIALOG.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_sessions = len(data["multi_session_dialogues"])
    total_rounds = sum(len(s["dialogues"]) for s in data["multi_session_dialogues"])
    total_qas = len(data["human-annotated QAs"])
    print(f"[INFO] Wrote {OUTPUT_DIALOG}")
    print(f"[INFO] sessions={total_sessions} rounds={total_rounds} qas={total_qas}")


if __name__ == "__main__":
    main()
