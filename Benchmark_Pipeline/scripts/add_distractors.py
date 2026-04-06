#!/usr/bin/env python3
"""
Add distractor content to ComicScene dialogues.
Distractors are similar to QA answers but with key differences to confuse the model.

QA Answer → Distractor Strategy:
- "crowned character in water" → "helmeted figure near river"
- "Dinny hits tree" → "dinosaur bumps into rocks"
- "trade-in dinosaur larger" → "exchange beast was smaller"
- "today's special smaller than cheap" → "premium one was largest"
- "dinosaur drooling" → "creature panting/snorting"
- "bon-bons next to reclining" → "fruits beside seated figure"
- "running with basin on head" → "carrying bucket on shoulder"
- "panel 5" → mentions of "panel 3" or "panel 4"
"""

import json
from pathlib import Path

# Distractor dialogues to INSERT into existing sessions
# These contain information similar to but different from QA answers
DISTRACTOR_INSERTIONS = {
    # Early sessions - distractors for D1-D4 related QAs
    "D5": [
        {
            "user": "I remember seeing some characters with interesting headgear in those early pages.",
            "assistant": "Yes, various characters wear different headpieces. I recall seeing a helmeted warrior figure near what looked like a riverbank in one panel. The prehistoric setting often features characters with distinctive accessories."
        },
        {
            "user": "Were there any memorable animal moments?",
            "assistant": "There was a scene where a dinosaur stumbled into some boulders - quite a comedic moment. The physical comedy with the prehistoric creatures is a hallmark of Hamlin's style."
        }
    ],
    "D6": [
        {
            "user": "The dinosaur dealership concept is funny. What sizes did they have?",
            "assistant": "From what I gathered, there were various sizes. The premium display model appeared to be the largest of the lot. The one marked for exchange seemed smaller than expected - buyers always want an upgrade."
        }
    ],
    "D7": [
        {
            "user": "Any particular scenes with characters in or near water?",
            "assistant": "I think there was a figure with some kind of headband standing by a pond at one point. Water features appear occasionally in the strip to add variety to the prehistoric landscape."
        }
    ],
    # Middle sessions - more distractors
    "D9": [
        {
            "user": "How do the dinosaurs behave in the strip?",
            "assistant": "They show various behaviors. I noticed one creature panting heavily after exertion - very dog-like. Another was snorting in what seemed like frustration. Hamlin gave them personality."
        },
        {
            "user": "Any memorable physical comedy with the dinosaurs?",
            "assistant": "There was a good bit where a dinosaur knocked over some logs while turning around. The large creatures in small spaces always creates humor. Clumsiness is a recurring theme."
        }
    ],
    "D11": [
        {
            "user": "Tell me about the royal characters in prehistoric comics.",
            "assistant": "In Alley Oop, the chief or leader figures often appear just once or twice per storyline. I recall seeing a bare-headed leader giving directions to workers at one point - typical authority figure stuff."
        }
    ],
    "D13": [
        {
            "user": "How do characters carry things in the strip?",
            "assistant": "Creatively! I remember seeing someone carrying a bucket balanced on their shoulder - very practical for the setting. Characters often improvise with available materials."
        },
        {
            "user": "Any scenes of characters running?",
            "assistant": "Several chase scenes. In one, a character was running while carrying something on their back - maybe around panel 3 or 4 of that page. The action sequences are dynamic."
        }
    ],
    "D14": [
        {
            "user": "What about scenes showing characters at rest?",
            "assistant": "There are quieter moments. I recall a seated figure with what looked like fruits or vegetables beside them - a peaceful domestic scene. The strip balances action with character moments."
        }
    ],
    # Later sessions - distractors for D27-D29 related QAs
    "D16": [
        {
            "user": "How does the strip show power dynamics?",
            "assistant": "Through visual cues mostly. A robed figure arguing with a helmeted one suggests conflict between different social classes. The kingdom politics are often played for comedy."
        }
    ],
    "D17": [
        {
            "user": "Are there scenes of characters working versus relaxing?",
            "assistant": "Yes, both appear. Interestingly, I seem to recall a sequence where someone was working first, then shown resting later - the opposite of what you might expect in a 'lazy ruler' trope."
        }
    ],
    "D18": [
        {
            "user": "What happens when crowds get involved in the strip?",
            "assistant": "Crowd scenes add energy. There was one where people seemed to be cheering while a figure relaxed in a hut - celebrating perhaps? Hamlin uses crowds for emotional punctuation."
        }
    ],
    "D21": [
        {
            "user": "Any memorable confrontation scenes?",
            "assistant": "Several! I recall a scene where someone was pushed rather than struck - the distinction matters for the humor. Physical comedy walks a line between slapstick and violence."
        }
    ],
    "D22": [
        {
            "user": "How precise is the panel-by-panel storytelling?",
            "assistant": "Very deliberate. Each panel advances the story. I remember analyzing one page where significant action happened around panels 3 and 4 - the middle panels often carry key story beats."
        }
    ],
    "D24": [
        {
            "user": "Anything about characters and their dinosaur mounts?",
            "assistant": "The relationship between riders and mounts is interesting. I recall seeing a character who seemed to prefer the larger beast for riding - size equals status perhaps."
        }
    ]
}

def load_dialog_json(path):
    with open(path) as f:
        return json.load(f)

def save_dialog_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def add_distractors(data):
    """Add distractor dialogues to existing sessions."""

    for session in data['multi_session_dialogues']:
        sid = session['session_id']

        if sid in DISTRACTOR_INSERTIONS:
            # Get current max round number
            current_rounds = len(session['dialogues'])

            # Add distractor dialogues
            for i, distractor in enumerate(DISTRACTOR_INSERTIONS[sid], start=1):
                session['dialogues'].append({
                    "round": f"{sid}:{current_rounds + i}",
                    "user": distractor["user"],
                    "assistant": distractor["assistant"]
                })

    return data

def count_stats(data):
    """Count statistics."""
    total_rounds = sum(len(s['dialogues']) for s in data['multi_session_dialogues'])
    image_rounds = sum(
        1 for s in data['multi_session_dialogues']
        for d in s['dialogues'] if 'input_image' in d
    )
    return total_rounds, image_rounds

def main():
    path = Path("/common/home/mg1998/MemoryBench/Benchmark_Pipeline/data/dialog/ComicScene_Alley_Oop_Draft.json")

    print(f"Loading: {path}")
    data = load_dialog_json(path)

    total_before, images_before = count_stats(data)
    print(f"Before: {total_before} rounds ({images_before} with images)")

    # Add distractors
    data = add_distractors(data)

    total_after, images_after = count_stats(data)
    print(f"After: {total_after} rounds ({images_after} with images)")
    print(f"Added: {total_after - total_before} distractor rounds")

    # Show which sessions were modified
    print("\nDistractors added to sessions:")
    for sid in sorted(DISTRACTOR_INSERTIONS.keys(), key=lambda x: int(x[1:])):
        count = len(DISTRACTOR_INSERTIONS[sid])
        print(f"  {sid}: +{count} distractor round(s)")

    # Save
    save_dialog_json(data, path)
    print(f"\nSaved: {path}")

    # Show sample distractors
    print("\n=== Sample Distractors ===")
    print("\n[D5] Distractor for 'crowned in water':")
    print(f"  → 'helmeted warrior near riverbank'")
    print("\n[D6] Distractor for 'trade-in larger':")
    print(f"  → 'premium model was largest, exchange one seemed smaller'")
    print("\n[D9] Distractor for 'dinosaur drooling':")
    print(f"  → 'creature panting, another snorting'")
    print("\n[D13] Distractor for 'running with basin, panel 5':")
    print(f"  → 'carrying bucket on shoulder, around panel 3 or 4'")

if __name__ == "__main__":
    main()
