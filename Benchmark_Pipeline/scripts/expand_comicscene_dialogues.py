#!/usr/bin/env python3
"""
Expand ComicScene dialog with text-only rounds to increase memory burden.
Strategy:
- Add 2-3 text rounds after each image session
- Insert 5-8 filler sessions between D4 and D27
- Total: ~40-60 new text-only rounds
- Topics: 70% comic-related, 30% tangential (dinosaurs, prehistoric life, etc.)
- AVOID: Any information that could answer the QAs
"""

import json
from pathlib import Path

# Text-only dialogue templates
# These discuss the comic WITHOUT revealing QA-relevant visual details

INTRA_SESSION_DIALOGUES = {
    "D1": [
        {
            "user": "What do you think of the art style so far? It has that classic newspaper strip look.",
            "assistant": "The linework is very characteristic of early 1930s comic strips. V.T. Hamlin had a distinctive style that balanced detailed prehistoric settings with expressive character work. The panel layouts are efficient for daily newspaper formats."
        },
        {
            "user": "I've heard Alley Oop was one of the first adventure strips. Is that true?",
            "assistant": "It was among the pioneering adventure strips, yes. Starting in 1932, it helped establish the adventure genre alongside strips like Tarzan and Buck Rogers. The prehistoric setting gave Hamlin freedom to create fantastical scenarios while grounding them in a semi-realistic world."
        }
    ],
    "D2": [
        {
            "user": "The dinosaurs in this strip are interesting. They seem almost like vehicles or pets.",
            "assistant": "That's a good observation. Hamlin portrayed dinosaurs as domesticated creatures in Moo, similar to how we might use horses or cars today. It's an anachronistic but charming approach that makes the prehistoric setting more relatable."
        },
        {
            "user": "Were dinosaurs depicted accurately in 1930s media?",
            "assistant": "Not by modern standards. In the 1930s, dinosaurs were often shown as slow, tail-dragging reptiles. Scientific understanding has changed dramatically since then. But for storytelling purposes, Hamlin's dinosaurs work well as narrative devices."
        }
    ],
    "D3": [
        {
            "user": "This dealership concept is funny. A prehistoric used dinosaur lot.",
            "assistant": "It's one of Hamlin's clever anachronisms. By transplanting modern commercial concepts into a prehistoric setting, he created humor through incongruity. Car dealerships were becoming ubiquitous in 1930s America, so readers would immediately get the joke."
        },
        {
            "user": "Do you think the humor holds up today?",
            "assistant": "Much of it does. The core comedy comes from universal human experiences - being sold something, dealing with salespeople, wanting an upgrade. Those situations transcend the era. The specific 1930s references might be lost on modern readers though."
        }
    ],
    "D4": [
        {
            "user": "Test drives existed even for dinosaurs apparently!",
            "assistant": "Another modern concept mapped onto prehistory. The test drive became standard practice for automobiles in the 1920s. By 1933 when this strip ran, readers would find it amusing to see the same ritual applied to dinosaur purchases."
        },
        {
            "user": "What other modern concepts did Hamlin use in the strip?",
            "assistant": "He incorporated politics, fashion, commerce, and social hierarchies. The kingdom of Moo has kings, guards, and social classes. Later stories even involved time travel, allowing direct commentary on contemporary society."
        }
    ],
    "D27": [
        {
            "user": "The political aspects of the strip are more visible in these later pages.",
            "assistant": "Yes, the power dynamics become central to the narrative. Hamlin often used the kingdom setting to satirize real-world politics. Leadership, authority, and the relationship between rulers and citizens were recurring themes."
        },
        {
            "user": "Was political commentary common in comic strips of that era?",
            "assistant": "Absolutely. Comic strips reached millions of readers daily, making them powerful vehicles for social commentary. Some strips were overtly political, while others like Alley Oop wove themes more subtly into adventure narratives."
        }
    ],
    "D28": [
        {
            "user": "The dynamics between characters seem to be shifting in this part of the story.",
            "assistant": "Character relationships evolve throughout the strip's run. Hamlin developed his cast over years, with alliances and conflicts driving the narrative. This serial storytelling kept readers coming back to see what happened next."
        },
        {
            "user": "How did daily strips handle ongoing storylines back then?",
            "assistant": "Each daily strip needed to be somewhat self-contained while advancing a larger arc. Artists balanced recapping for new readers with progression for regulars. Sunday strips often had separate, more elaborate stories."
        }
    ],
    "D29": [
        {
            "user": "Things seem to be getting more chaotic in the story.",
            "assistant": "Escalation is a classic storytelling technique. Hamlin built tension through increasingly complicated situations. The resolution often required cleverness from the protagonist rather than brute force, which appealed to Depression-era readers."
        },
        {
            "user": "Were readers invested in these continuing stories?",
            "assistant": "Very much so. Newspapers received letters about popular strips, and circulation could be affected by which comics a paper carried. Some readers followed strips for decades, watching characters grow and change over time."
        }
    ]
}

# Filler sessions between D4 and D27 (the time gap)
FILLER_SESSIONS = [
    {
        "session_id": "D5",
        "date": "1933-01-05",
        "dialogues": [
            {
                "user": "I've been thinking about the dinosaur species shown in the comic. Are they based on real ones?",
                "assistant": "Hamlin took creative liberties. Some dinosaurs resemble sauropods or theropods, but they're not meant to be scientifically accurate representations. The strip predates many paleontological discoveries that shaped modern dinosaur depictions."
            },
            {
                "user": "What did scientists think dinosaurs looked like in the 1930s?",
                "assistant": "The prevailing view was that dinosaurs were cold-blooded, slow-moving reptiles. They were often depicted standing upright with dragging tails. The idea of feathered dinosaurs or warm-blooded metabolisms came much later."
            }
        ]
    },
    {
        "session_id": "D8",
        "date": "1933-01-08",
        "dialogues": [
            {
                "user": "I'm curious about the setting. Where is the Kingdom of Moo supposed to be located?",
                "assistant": "Hamlin never pinned down a specific location. It's a fictional prehistoric land that exists outside normal geography. This vagueness gave him freedom to include various dinosaur types and landscapes without worrying about accuracy."
            },
            {
                "user": "Did other prehistoric fiction of that era do the same thing?",
                "assistant": "Many did. Films like 'One Million B.C.' and novels took similar liberties. Audiences weren't as concerned with scientific accuracy as they might be today. Entertainment value trumped paleontological correctness."
            },
            {
                "user": "What about the name 'Moo' - any significance?",
                "assistant": "It's likely just a humorous, simple name that sounds vaguely primitive. Hamlin had a talent for naming - 'Alley Oop' itself is a phrase from circus acrobatics, suggesting leaps and tumbles befitting an action hero."
            }
        ]
    },
    {
        "session_id": "D12",
        "date": "1933-01-12",
        "dialogues": [
            {
                "user": "How long did Alley Oop run as a comic strip?",
                "assistant": "It had an remarkably long run. V.T. Hamlin created it in 1932 and drew it until 1971. Other artists continued it afterward. At various points, it was one of the most widely syndicated strips in America."
            },
            {
                "user": "That's impressive longevity. What kept readers interested for so long?",
                "assistant": "Hamlin evolved the strip over time. He introduced time travel in 1939, allowing adventures across history. This expanded storytelling possibilities enormously while keeping the core characters familiar."
            }
        ]
    },
    {
        "session_id": "D15",
        "date": "1933-01-15",
        "dialogues": [
            {
                "user": "Speaking of prehistoric life, what's your take on how cavemen are usually portrayed in media?",
                "assistant": "The stereotypical 'caveman' is largely a fiction - club-wielding brutes in animal skins. Real prehistoric humans were sophisticated tool-makers with complex social structures. But the simplified image makes for accessible storytelling."
            },
            {
                "user": "Alley Oop fits that stereotype somewhat, doesn't he?",
                "assistant": "In appearance yes, but Hamlin gave him more depth. Alley is brave, loyal, and sometimes clever. He's not just a brute - he has relationships, faces moral dilemmas, and grows as a character over the strip's run."
            },
            {
                "user": "Were there other caveman characters in comics at the time?",
                "assistant": "Some, but none as successful. Alley Oop really defined the archetype for comics. Later characters like B.C. and The Flintstones owe a debt to Hamlin's pioneering work, even if they took different approaches."
            }
        ]
    },
    {
        "session_id": "D18",
        "date": "1933-01-18",
        "dialogues": [
            {
                "user": "I wonder what the daily routine was like for comic strip artists in the 1930s.",
                "assistant": "It was demanding work. Artists typically produced six daily strips plus a Sunday strip each week. They needed to write, pencil, ink, and letter everything, often working months ahead of publication."
            },
            {
                "user": "Did they work alone or have assistants?",
                "assistant": "Many successful artists hired assistants for backgrounds, lettering, or inking. Hamlin worked largely alone for much of his career, which is quite impressive given the strip's detailed prehistoric settings."
            }
        ]
    },
    {
        "session_id": "D21",
        "date": "1933-01-21",
        "dialogues": [
            {
                "user": "The 1930s must have been an interesting time for comics generally.",
                "assistant": "It was a golden age of newspaper comics. Strips reached enormous audiences before television. Artists like Hamlin, Alex Raymond, and Milton Caniff were celebrities. Their work influenced generations of cartoonists."
            },
            {
                "user": "How did the Great Depression affect comics?",
                "assistant": "Interestingly, comics thrived during the Depression. They were affordable entertainment that provided escapism. Adventure strips with exotic settings were especially popular - people wanted to imagine worlds beyond their economic troubles."
            },
            {
                "user": "So Alley Oop's prehistoric adventures served that escapist function?",
                "assistant": "Exactly. A world of dinosaurs and simple conflicts was far removed from unemployment lines and bank failures. The humor helped too - people needed to laugh during difficult times."
            }
        ]
    },
    {
        "session_id": "D24",
        "date": "1933-01-24",
        "dialogues": [
            {
                "user": "Before we continue with the later pages, I'm curious - did Alley Oop have a romantic interest?",
                "assistant": "Yes, Oola became his girlfriend and eventually wife in the strip. She was introduced early and became a central character. Their relationship provided ongoing storylines and humanized the action-oriented protagonist."
            },
            {
                "user": "Were female characters common in adventure strips of that era?",
                "assistant": "They existed but were often secondary. Strips like Flash Gordon had Dale Arden, and Terry and the Pirates featured strong female characters. The treatment varied - some were damsels in distress, others were more capable."
            }
        ]
    },
    {
        "session_id": "D26",
        "date": "1933-01-26",
        "dialogues": [
            {
                "user": "Ready to look at more pages tomorrow. These later ones seem to have a different tone.",
                "assistant": "Stories often shift in tone as they develop. The early pages establish characters and setting, while later pages complicate the situation and build toward resolution. It's a natural narrative progression."
            },
            {
                "user": "Do you think the strip was planned out in advance or improvised?",
                "assistant": "Probably a mix. Hamlin likely had general story arcs in mind but worked out details as he went. Daily strip artists often adjusted based on reader response and their own creative impulses. The format encouraged serialized storytelling."
            }
        ]
    }
]

def load_dialog_json(path):
    with open(path) as f:
        return json.load(f)

def save_dialog_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def expand_dialogues(data):
    """Add text-only rounds to existing sessions and insert filler sessions."""

    expanded_sessions = []

    for session in data['multi_session_dialogues']:
        sid = session['session_id']

        # Keep the original session
        expanded_session = session.copy()

        # Add intra-session text dialogues after the image round
        if sid in INTRA_SESSION_DIALOGUES:
            original_dialogues = list(session['dialogues'])
            text_dialogues = INTRA_SESSION_DIALOGUES[sid]

            for i, td in enumerate(text_dialogues, start=2):
                original_dialogues.append({
                    "round": f"{sid}:{i}",
                    "user": td["user"],
                    "assistant": td["assistant"]
                    # No image_id, input_image, or image_caption = text-only
                })

            expanded_session['dialogues'] = original_dialogues

        expanded_sessions.append(expanded_session)

        # After D4, insert filler sessions before D27
        if sid == "D4":
            for filler in FILLER_SESSIONS:
                filler_session = {
                    "session_id": filler["session_id"],
                    "date": filler["date"],
                    "dialogues": []
                }
                for i, d in enumerate(filler["dialogues"], start=1):
                    filler_session["dialogues"].append({
                        "round": f"{filler['session_id']}:{i}",
                        "user": d["user"],
                        "assistant": d["assistant"]
                    })
                expanded_sessions.append(filler_session)

    data['multi_session_dialogues'] = expanded_sessions
    return data

def count_rounds(data):
    """Count total rounds."""
    total = 0
    text_only = 0
    with_image = 0

    for session in data['multi_session_dialogues']:
        for d in session['dialogues']:
            total += 1
            if 'input_image' in d:
                with_image += 1
            else:
                text_only += 1

    return total, text_only, with_image

def main():
    input_path = Path("/common/home/mg1998/MemoryBench/Benchmark_Pipeline/data/dialog/ComicScene_Alley_Oop_Draft.json")
    output_path = input_path  # Overwrite in place

    print(f"Loading: {input_path}")
    data = load_dialog_json(input_path)

    # Count before
    total_before, text_before, image_before = count_rounds(data)
    print(f"Before: {total_before} rounds ({image_before} with images, {text_before} text-only)")

    # Expand
    data = expand_dialogues(data)

    # Count after
    total_after, text_after, image_after = count_rounds(data)
    print(f"After: {total_after} rounds ({image_after} with images, {text_after} text-only)")
    print(f"Added: {total_after - total_before} text-only rounds")

    # Show session structure
    print("\nExpanded session structure:")
    for session in data['multi_session_dialogues']:
        sid = session['session_id']
        rounds = len(session['dialogues'])
        has_image = any('input_image' in d for d in session['dialogues'])
        marker = "📷" if has_image else "💬"
        print(f"  {marker} {sid}: {rounds} round(s)")

    # Save
    save_dialog_json(data, output_path)
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
