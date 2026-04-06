#!/usr/bin/env python3
"""
Rebuild ComicScene dialog with:
- 13 comic pages (original 7 + 6 new: pages 8, 10, 12, 15, 20, 25)
- Text-only dialogue rounds after each image session
- Natural transition filler sessions between image sessions
- Total target: ~100 rounds for maximum memory burden
"""

import json
from pathlib import Path

# Image sessions with associated dialogues
IMAGE_SESSIONS = [
    # Original early pages
    {
        "session_id": "D1", "date": "1933-01-01", "page": 1,
        "user": "I'm reading an Alley Oop comic for fun and want your help following it page by page. Here's the first page I'm looking at.",
        "assistant": "Sounds good. I will follow along with you and keep this page in mind as the story continues.",
        "follow_up": [
            ("What do you think of the art style so far? It has that classic newspaper strip look.",
             "The linework is very characteristic of early 1930s comic strips. V.T. Hamlin had a distinctive style that balanced detailed prehistoric settings with expressive character work."),
            ("I've heard Alley Oop was one of the first adventure strips. Is that true?",
             "It was among the pioneering adventure strips, yes. Starting in 1932, it helped establish the adventure genre alongside strips like Tarzan and Buck Rogers.")
        ]
    },
    {
        "session_id": "D2", "date": "1933-01-02", "page": 2,
        "user": "Here's the next page. Help me keep track of the story as it develops.",
        "assistant": "Okay. I'll keep this next page in context with the earlier one.",
        "follow_up": [
            ("The dinosaurs in this strip are interesting. They seem almost like vehicles or pets.",
             "That's a good observation. Hamlin portrayed dinosaurs as domesticated creatures in Moo, similar to how we might use horses or cars today."),
            ("Were dinosaurs depicted accurately in 1930s media?",
             "Not by modern standards. In the 1930s, dinosaurs were often shown as slow, tail-dragging reptiles. Scientific understanding has changed dramatically since then.")
        ]
    },
    {
        "session_id": "D3", "date": "1933-01-03", "page": 3,
        "user": "I kept reading and got to this page. Please keep following the comic with me.",
        "assistant": "Sure. I'll keep this page connected to the earlier parts of the comic.",
        "follow_up": [
            ("This dealership concept is funny. A prehistoric used dinosaur lot.",
             "It's one of Hamlin's clever anachronisms. By transplanting modern commercial concepts into a prehistoric setting, he created humor through incongruity."),
            ("Do you think the humor holds up today?",
             "Much of it does. The core comedy comes from universal human experiences - being sold something, dealing with salespeople, wanting an upgrade.")
        ]
    },
    {
        "session_id": "D4", "date": "1933-01-04", "page": 4,
        "user": "Here's the next page in the same comic sequence. Help me stay oriented as I read through it.",
        "assistant": "Will do. I'll keep this page in the running story context.",
        "follow_up": [
            ("Test drives existed even for dinosaurs apparently!",
             "Another modern concept mapped onto prehistory. The test drive became standard practice for automobiles in the 1920s."),
            ("What other modern concepts did Hamlin use in the strip?",
             "He incorporated politics, fashion, commerce, and social hierarchies. The kingdom of Moo has kings, guards, and social classes.")
        ]
    },
    # New middle pages
    {
        "session_id": "D8", "date": "1933-01-08", "page": 8,
        "user": "I skipped ahead a few pages. Here's where I'm at now in the story.",
        "assistant": "Got it. I'll incorporate this page into our ongoing reading of the comic.",
        "follow_up": [
            ("The story seems to be developing. Are these daily strips or weekly?",
             "These were daily strips, published Monday through Saturday in newspapers. Each strip advanced the story incrementally, keeping readers coming back."),
            ("How did readers keep track of ongoing storylines?",
             "Most readers followed religiously. Missing a day meant missing plot points. Some newspapers would print brief recaps, but generally readers were expected to follow along.")
        ]
    },
    {
        "session_id": "D10", "date": "1933-01-10", "page": 10,
        "user": "Continuing through the story. Here's another page I want to keep track of.",
        "assistant": "Noted. I'll keep this page in context with everything we've seen so far.",
        "follow_up": [
            ("The pacing seems different from modern comics.",
             "Daily strips had unique pacing constraints. Each strip needed a small hook or gag while advancing a larger narrative. It's a lost art form in many ways."),
            ("Did Hamlin write and draw everything himself?",
             "For most of his career, yes. He was both writer and artist, which gave the strip a consistent voice and vision throughout its run.")
        ]
    },
    {
        "session_id": "D12", "date": "1933-01-12", "page": 12,
        "user": "Here's another page from my reading session. The plot is getting more involved.",
        "assistant": "I see. I'll track this page along with the others as the story continues to develop.",
        "follow_up": [
            ("How long did typical story arcs last in strips like this?",
             "Arcs could run anywhere from a week to several months. Hamlin balanced episodic humor with longer adventure narratives. Readers appreciated both."),
            ("Were there recurring villains or antagonists?",
             "Yes, King Guzzle was a frequent foil. Various prehistoric creatures and rival tribes also provided conflict. The strip had a rich cast of supporting characters.")
        ]
    },
    {
        "session_id": "D15", "date": "1933-01-15", "page": 15,
        "user": "Moving along in the comic. Here's where I am now.",
        "assistant": "Understood. I'll keep this page connected to the overall narrative we've been following.",
        "follow_up": [
            ("The backgrounds are quite detailed for a daily strip.",
             "Hamlin put considerable effort into his prehistoric environments. The lush vegetation and rocky landscapes helped establish Moo as a believable world."),
            ("Did he have reference materials for prehistoric settings?",
             "He likely consulted natural history books and museum displays. The 1930s saw growing public interest in dinosaurs, so resources were available.")
        ]
    },
    {
        "session_id": "D20", "date": "1933-01-20", "page": 20,
        "user": "Getting further into the story now. Here's this page.",
        "assistant": "Alright. I'll integrate this with everything we've read so far in the comic.",
        "follow_up": [
            ("The character expressions are quite animated.",
             "Hamlin excelled at expressive faces. Even in small panels, characters conveyed clear emotions. This was crucial for daily strips where space was limited."),
            ("How did the strip influence later cartoonists?",
             "Many artists cite Alley Oop as an influence. The blend of humor and adventure, the prehistoric setting, and the character designs all left marks on the medium.")
        ]
    },
    {
        "session_id": "D25", "date": "1933-01-25", "page": 25,
        "user": "Almost caught up to where I was before. Here's this page from the comic.",
        "assistant": "Got it. I'll keep this in mind as we approach the later pages you mentioned earlier.",
        "follow_up": [
            ("We're getting close to those later pages now.",
             "Yes, the story should be building toward the events you'll show me soon. These intermediate pages provide important context."),
            ("Do you notice any changes in the art style as the strip progressed?",
             "Artists often refine their style over time. Hamlin's linework became more confident, and his compositions more dynamic as he settled into the strip.")
        ]
    },
    # Original later pages
    {
        "session_id": "D27", "date": "1933-01-27", "page": 27,
        "user": "I jumped ahead to a later part of the same comic and want help following what's going on there too.",
        "assistant": "Okay. I'll treat this as a later part of the same storyline and keep track of it with you.",
        "follow_up": [
            ("The political aspects of the strip are more visible in these later pages.",
             "Yes, the power dynamics become central to the narrative. Hamlin often used the kingdom setting to satirize real-world politics."),
            ("Was political commentary common in comic strips of that era?",
             "Absolutely. Comic strips reached millions of readers daily, making them powerful vehicles for social commentary.")
        ]
    },
    {
        "session_id": "D28", "date": "1933-01-28", "page": 28,
        "user": "Here's another later page from that Alley Oop storyline. Help me keep the thread straight.",
        "assistant": "Got it. I'll keep following the storyline with this page too.",
        "follow_up": [
            ("The dynamics between characters seem to be shifting in this part of the story.",
             "Character relationships evolve throughout the strip's run. Hamlin developed his cast over years, with alliances and conflicts driving the narrative."),
            ("How did daily strips handle ongoing storylines back then?",
             "Each daily strip needed to be somewhat self-contained while advancing a larger arc. Artists balanced recapping for new readers with progression for regulars.")
        ]
    },
    {
        "session_id": "D29", "date": "1933-01-29", "page": 29,
        "user": "This is one more page from the same reading session. Stay with me on the plot and characters here too.",
        "assistant": "Will do. I'll keep this page aligned with the rest of the comic as we go.",
        "follow_up": [
            ("Things seem to be getting more chaotic in the story.",
             "Escalation is a classic storytelling technique. Hamlin built tension through increasingly complicated situations."),
            ("Were readers invested in these continuing stories?",
             "Very much so. Newspapers received letters about popular strips, and circulation could be affected by which comics a paper carried.")
        ]
    }
]

# Filler text-only sessions (natural transitions to unrelated topics)
FILLER_SESSIONS = [
    {
        "session_id": "D5", "date": "1933-01-05",
        "dialogues": [
            ("Taking a break from the comic. What else were people reading in the 1930s?",
             "The 1930s saw the rise of pulp magazines, detective fiction, and science fiction. Writers like Dashiell Hammett and Isaac Asimov were gaining popularity."),
            ("Were comic strips considered literature?",
             "Not by literary critics, but they had cultural impact rivaling many novels. Some strips were collected into books, though this was less common then."),
            ("What about movies? What was popular?",
             "The Golden Age of Hollywood was beginning. Monster movies like Frankenstein and King Kong captivated audiences. Sound films had just become standard.")
        ]
    },
    {
        "session_id": "D6", "date": "1933-01-06",
        "dialogues": [
            ("Speaking of King Kong, that movie has dinosaurs too.",
             "Yes, the stop-motion dinosaurs in King Kong were groundbreaking. Willis O'Brien's animation influenced generations of filmmakers."),
            ("How did they create those effects?",
             "Stop-motion animation with detailed miniatures. Each frame was photographed individually, with tiny movements between shots. Incredibly labor-intensive."),
            ("It's amazing what they achieved without computers.",
             "The creativity required was immense. Constraints often drive innovation. Many techniques developed then are still studied by filmmakers today.")
        ]
    },
    {
        "session_id": "D7", "date": "1933-01-07",
        "dialogues": [
            ("What was daily life like during the Depression era?",
             "For many, it was a struggle. Unemployment reached 25%. Families relied on community support, government programs, and whatever work they could find."),
            ("How did entertainment help people cope?",
             "Escapism was crucial. Radio shows, movies, and comic strips offered affordable escape from harsh realities. Laughter and adventure were valuable commodities."),
            ("Were comic strips expensive to access?",
             "Newspapers were cheap, often just a few cents. Comics were included, making them accessible entertainment. This democratization of storytelling was significant.")
        ]
    },
    {
        "session_id": "D9", "date": "1933-01-09",
        "dialogues": [
            ("I've been thinking about how different communication was back then.",
             "No internet, limited telephone access, radio was the cutting edge. News traveled slower, but people were deeply engaged with available media."),
            ("Did families gather around the newspaper like they later did with TV?",
             "In many homes, yes. The morning paper was a ritual. Different family members might claim different sections. Comics were often fought over."),
            ("What happened to newspapers that dropped popular strips?",
             "Reader backlash could be severe. Papers received angry letters, and some readers switched papers entirely. Strips had real commercial value.")
        ]
    },
    {
        "session_id": "D11", "date": "1933-01-11",
        "dialogues": [
            ("Let me ask about something different. What do you know about actual prehistoric life?",
             "Paleontology has revealed a rich history of life on Earth spanning billions of years. Dinosaurs dominated for about 165 million years before their extinction."),
            ("How accurate are comic depictions of prehistoric life?",
             "Generally not very accurate, but that's fine for entertainment. Scientific understanding continues to evolve anyway. We now know many dinosaurs had feathers."),
            ("Would Alley Oop look different if drawn today with modern knowledge?",
             "Likely yes. The dinosaurs would look different, the timeline would be corrected, but the humor and adventure could remain the same.")
        ]
    },
    {
        "session_id": "D13", "date": "1933-01-13",
        "dialogues": [
            ("What's your favorite type of prehistoric creature?",
             "As an AI, I don't have preferences, but I find the diversity fascinating. From tiny feathered dinosaurs to massive sauropods to bizarre sea creatures."),
            ("Tell me about some unusual prehistoric animals.",
             "Deinocheirus had enormous arms. Quetzalcoatlus was a flying reptile with a wingspan of a small airplane. Anomalocaris was a Cambrian predator that looked alien."),
            ("Were there prehistoric mammals alongside dinosaurs?",
             "Yes, small mammals coexisted with dinosaurs for millions of years. After the dinosaur extinction, mammals diversified rapidly into the forms we know today.")
        ]
    },
    {
        "session_id": "D14", "date": "1933-01-14",
        "dialogues": [
            ("Changing topics - what do you think about time travel stories?",
             "Time travel is a rich narrative device. It allows exploration of causality, history, and human nature. Many classic stories use it effectively."),
            ("Did Alley Oop ever involve time travel?",
             "Yes! In 1939, Hamlin introduced a time machine, allowing Alley to visit different historical periods. It greatly expanded storytelling possibilities."),
            ("That's a big shift from pure prehistoric adventure.",
             "It was controversial at first but proved popular. The strip could now comment on any historical era while keeping beloved characters.")
        ]
    },
    {
        "session_id": "D16", "date": "1933-01-16",
        "dialogues": [
            ("I've been reading about 1930s fashion. It's so different from today.",
             "The era had distinctive styles. Women's fashion featured longer hemlines after the 1920s. Men wore suits with wider lapels and high-waisted trousers."),
            ("How did fashion appear in comic strips?",
             "Contemporary strips showed current fashions. Period strips like Alley Oop invented their own aesthetics. The caveman look became iconic partly through comics."),
            ("Did Alley Oop merchandise his look?",
             "The character appeared on various products over the years. The distinctive appearance made him recognizable and marketable.")
        ]
    },
    {
        "session_id": "D17", "date": "1933-01-17",
        "dialogues": [
            ("What about music in the 1930s? What were people listening to?",
             "Jazz and swing were dominant. Big bands led by Benny Goodman, Duke Ellington, and Count Basie defined the era. Radio brought music into homes nationwide."),
            ("Were there songs about comics or cartoon characters?",
             "Some characters inspired songs. Betty Boop had musical connections. The intersection of comics and music was less developed than today but existed."),
            ("How did radio change entertainment consumption?",
             "Dramatically. Families gathered around radios for shows, news, and music. It created shared national experiences and launched celebrity culture.")
        ]
    },
    {
        "session_id": "D18", "date": "1933-01-18",
        "dialogues": [
            ("Let's talk about food. What did people eat in the 1930s?",
             "The Depression influenced diets significantly. Economical meals were essential. Casseroles, soups, and creative use of leftovers became common."),
            ("Were there any iconic foods from that era?",
             "Spam was introduced in 1937. Kraft Macaroni and Cheese appeared in 1937. These affordable, shelf-stable foods became American staples."),
            ("What about restaurants and dining out?",
             "Fine dining existed but was inaccessible to most. Diners and lunch counters served affordable meals. The automat was a popular innovation.")
        ]
    },
    {
        "session_id": "D19", "date": "1933-01-19",
        "dialogues": [
            ("I'm curious about transportation in the 1930s.",
             "Automobiles were becoming more common but still a luxury for many. Trains remained crucial for long-distance travel. Aviation was exciting but limited."),
            ("How did people commute to work?",
             "Streetcars and buses in cities, walking for many. Suburbs were growing but not yet dominant. The car-centric culture was still developing."),
            ("Did comics reflect transportation changes?",
             "Often, yes. Adventure strips featured planes, trains, and automobiles. The modern world was exciting, and comics captured that enthusiasm.")
        ]
    },
    {
        "session_id": "D21", "date": "1933-01-21",
        "dialogues": [
            ("What were homes like in the 1930s?",
             "Many lacked modern conveniences we take for granted. Refrigerators were replacing iceboxes. Indoor plumbing wasn't universal. Radio was the electronic centerpiece."),
            ("How did people spend evenings without TV or internet?",
             "Reading, radio programs, board games, conversation. Community events, church activities, and visiting neighbors. Entertainment was more social and local."),
            ("That sounds both limiting and appealing somehow.",
             "There's nostalgia for simpler times, though we shouldn't romanticize hardship. Each era has trade-offs. The 1930s had community but also economic devastation.")
        ]
    },
    {
        "session_id": "D22", "date": "1933-01-22",
        "dialogues": [
            ("Tell me about sports in the 1930s.",
             "Baseball was king. Babe Ruth was a living legend. Boxing drew huge audiences. College football was immensely popular. The Olympics were politicized."),
            ("Were sports covered in comics?",
             "Sports strips existed and were popular. Joe Palooka, a boxing strip, was huge. Sports and comics both offered escapism and heroic narratives."),
            ("Did any comic characters play sports?",
             "Various strips incorporated athletics. The physical prowess of adventure heroes often extended to sporting contexts. Readers enjoyed seeing characters compete.")
        ]
    },
    {
        "session_id": "D23", "date": "1933-01-23",
        "dialogues": [
            ("What about science and technology advances in that era?",
             "Radio technology advanced rapidly. Early television experiments occurred. Aviation pushed boundaries. Scientific understanding grew despite economic constraints."),
            ("Were people excited about technological progress?",
             "Generally yes. The 1933 World's Fair in Chicago celebrated progress. Technology promised a better future, which was hopeful during hard times."),
            ("Did science fiction comics exist then?",
             "Buck Rogers debuted in 1929 and was hugely influential. Flash Gordon started in 1934. Science fiction comics helped shape public imagination about the future.")
        ]
    },
    {
        "session_id": "D24", "date": "1933-01-24",
        "dialogues": [
            ("We've wandered far from Alley Oop. Should we get back to the comic?",
             "Whenever you're ready. These tangents provide interesting context for the era when the strip was created. Understanding the times enriches the reading."),
            ("I have a few more pages to show you from the later part of the story.",
             "I'm ready to continue. We've covered quite a bit of ground with the earlier pages. Let's see how the story develops."),
            ("The next pages should wrap up this particular storyline.",
             "Story conclusions are often where Hamlin shined. The resolution of conflicts and character arcs made for satisfying reading.")
        ]
    },
    {
        "session_id": "D26", "date": "1933-01-26",
        "dialogues": [
            ("One more thing before those final pages - what made Alley Oop special compared to other strips?",
             "The unique prehistoric setting, likeable characters, blend of humor and adventure, and Hamlin's distinctive art all contributed. It carved its own niche."),
            ("Would you recommend it to modern readers?",
             "For those interested in comic history, absolutely. The humor is dated in places but the craft is evident. It's a window into how visual storytelling developed."),
            ("Alright, let me show you those last few pages now.",
             "I'm ready. With all the context we've built up, it should be interesting to see how this arc concludes.")
        ]
    }
]

def build_dialog_json():
    """Build the complete expanded dialog JSON."""

    # Load existing JSON for character_profile and QAs
    with open("/common/home/mg1998/MemoryBench/Benchmark_Pipeline/data/dialog/ComicScene_Alley_Oop_Draft.json") as f:
        existing = json.load(f)

    sessions = []

    # Process all sessions in chronological order
    all_session_ids = set()

    # Add image sessions
    for img_session in IMAGE_SESSIONS:
        sid = img_session["session_id"]
        all_session_ids.add(sid)

        session = {
            "session_id": sid,
            "date": img_session["date"],
            "dialogues": []
        }

        # First round with image
        session["dialogues"].append({
            "round": f"{sid}:1",
            "user": img_session["user"],
            "assistant": img_session["assistant"],
            "image_id": [f"{sid}:IMG_001"],
            "input_image": [f"ComicScene_Alley_Oop_Draft/Alley_Oop_Page_{img_session['page']}.jpg"],
            "image_caption": [f"Alley Oop comic page {img_session['page']}."]
        })

        # Follow-up text rounds
        for i, (user, assistant) in enumerate(img_session.get("follow_up", []), start=2):
            session["dialogues"].append({
                "round": f"{sid}:{i}",
                "user": user,
                "assistant": assistant
            })

        sessions.append(session)

    # Add filler sessions
    for filler in FILLER_SESSIONS:
        sid = filler["session_id"]
        all_session_ids.add(sid)

        session = {
            "session_id": sid,
            "date": filler["date"],
            "dialogues": []
        }

        for i, (user, assistant) in enumerate(filler["dialogues"], start=1):
            session["dialogues"].append({
                "round": f"{sid}:{i}",
                "user": user,
                "assistant": assistant
            })

        sessions.append(session)

    # Sort sessions by date
    sessions.sort(key=lambda s: s["date"])

    # Build final structure
    result = {
        "character_profile": existing["character_profile"],
        "multi_session_dialogues": sessions,
        "human-annotated QAs": existing["human-annotated QAs"]
    }

    return result

def count_stats(data):
    """Count statistics about the dialog."""
    total_rounds = 0
    image_rounds = 0
    text_rounds = 0
    total_sessions = len(data["multi_session_dialogues"])
    image_sessions = 0

    for session in data["multi_session_dialogues"]:
        has_image = False
        for d in session["dialogues"]:
            total_rounds += 1
            if "input_image" in d:
                image_rounds += 1
                has_image = True
            else:
                text_rounds += 1
        if has_image:
            image_sessions += 1

    return {
        "total_sessions": total_sessions,
        "image_sessions": image_sessions,
        "text_sessions": total_sessions - image_sessions,
        "total_rounds": total_rounds,
        "image_rounds": image_rounds,
        "text_rounds": text_rounds
    }

def main():
    output_path = Path("/common/home/mg1998/MemoryBench/Benchmark_Pipeline/data/dialog/ComicScene_Alley_Oop_Draft.json")

    print("Building expanded dialog JSON...")
    data = build_dialog_json()

    stats = count_stats(data)
    print(f"\nStatistics:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"    - With images: {stats['image_sessions']}")
    print(f"    - Text-only: {stats['text_sessions']}")
    print(f"  Total rounds: {stats['total_rounds']}")
    print(f"    - With images: {stats['image_rounds']}")
    print(f"    - Text-only: {stats['text_rounds']}")

    print("\nSession structure:")
    for session in data["multi_session_dialogues"]:
        sid = session["session_id"]
        rounds = len(session["dialogues"])
        has_image = any("input_image" in d for d in session["dialogues"])
        marker = "📷" if has_image else "💬"
        print(f"  {marker} {sid} ({session['date']}): {rounds} round(s)")

    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
