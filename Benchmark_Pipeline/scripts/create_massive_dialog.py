#!/usr/bin/env python3
"""
Create a MASSIVE dialog JSON to fill the context window.
Target: ~100K tokens
- 35 comic pages = ~38K tokens
- Need ~60K tokens of text = ~240K characters
"""

import json
from pathlib import Path

# Very long text passages on various topics
# Each passage is designed to be 2000-5000 characters

LONG_PASSAGES = {
    "comic_history_1": """The history of comic strips in America is a fascinating journey that began in the late 19th century. The Yellow Kid, created by Richard F. Outcault in 1895, is often credited as one of the first comic strip characters. This character appeared in Joseph Pulitzer's New York World and later in William Randolph Hearst's New York Journal, sparking what became known as 'yellow journalism.' The term actually derived from the character's distinctive yellow nightshirt.

The early 1900s saw an explosion of comic creativity. Winsor McCay's Little Nemo in Slumberland, which debuted in 1905, pushed the boundaries of what comics could achieve visually. McCay's elaborate art nouveau style and dream-logic narratives influenced countless artists. His work demonstrated that comics could be high art, not merely disposable entertainment.

George Herriman's Krazy Kat, running from 1913 to 1944, achieved critical acclaim that few comics have matched. The strip's surreal humor, poetic language, and constantly shifting desert landscapes made it a favorite among intellectuals and artists. The love triangle between Krazy Kat, Ignatz Mouse, and Officer Pupp explored themes of love, rejection, and persistence in ways that continue to resonate.

The 1920s brought adventure strips to prominence. Roy Crane's Wash Tubbs introduced sustained action narratives to the comics page. Harold Gray's Little Orphan Annie combined adventure with strong political commentary, reflecting the conservative views of its creator. These strips demonstrated that comics could engage with serious themes while entertaining mass audiences.

By the 1930s, the adventure strip had fully matured. Chester Gould's Dick Tracy brought crime drama to the funny pages with unprecedented violence and moral clarity. Alex Raymond's Flash Gordon offered science fiction spectacle that rivaled the pulp magazines. Milton Caniff's Terry and the Pirates combined adventure, romance, and geopolitical intrigue in ways that influenced generations of storytellers.""",

    "comic_history_2": """The syndication system that developed in the early 20th century transformed comics from local newspaper features into national phenomena. King Features Syndicate, United Feature Syndicate, and others distributed strips to hundreds of newspapers simultaneously. This system created enormous audiences - by the 1930s, popular strips reached tens of millions of readers daily.

The economics of syndication shaped comic content significantly. Strips needed broad appeal to succeed in diverse markets. Violence couldn't be too graphic, politics couldn't be too partisan, and humor couldn't be too regional. These constraints pushed artists toward universal themes and accessible storytelling.

Sunday comics developed as a separate art form. The larger format and color printing allowed for more elaborate artwork. Many artists produced distinct Sunday strips that didn't connect to their daily continuity. The Sunday comics section became a treasured family ritual, with parents and children sharing the pages each week.

The production demands were grueling. Artists typically worked six to eight weeks ahead of publication. They produced six daily strips plus a Sunday strip each week - about 360 daily strips and 52 Sunday strips per year. Many worked twelve-hour days, seven days a week. The pressure drove some to alcoholism or breakdown. Others, like Charles Schulz with Peanuts decades later, maintained the pace for fifty years.

Comic strip merchandising became big business. Characters appeared on toys, clothing, food products, and countless other items. Radio adaptations brought comic characters to life in new media. The Lone Ranger, though originating on radio, demonstrated the cross-media potential that comics would soon exploit. By the late 1930s, comic characters were among the most recognizable figures in American popular culture.""",

    "dinosaur_science_1": """Paleontology in the 1930s was experiencing a relative lull between two great ages of dinosaur discovery. The 'Bone Wars' of the late 19th century, when Othniel Charles Marsh and Edward Drinker Cope competed fiercely to discover and name new species, had established many of the dinosaurs we know today. The early 20th century brought important discoveries in Africa and Asia, but public fascination had somewhat waned.

The understanding of dinosaurs in the 1930s was fundamentally different from today's knowledge. Scientists believed dinosaurs were cold-blooded, sluggish creatures that dragged their tails on the ground. The famous Charles R. Knight paintings, which shaped public perception for decades, depicted dinosaurs in swampy environments, standing in water to support their massive bodies. We now know this was entirely wrong.

Tyrannosaurus rex, discovered in 1902 by Barnum Brown, captured public imagination but was poorly understood. Scientists debated whether it was a predator or scavenger. Its tiny arms puzzled researchers - some suggested they were vestigial, others that they served unknown functions. The debate continues today, though we now have much better evidence for T. rex as an apex predator.

Brontosaurus, named by Marsh in 1879, was considered invalid by the 1930s, merged with Apatosaurus. However, the name persisted in popular culture. Recent research has actually reinstated Brontosaurus as a valid genus, vindicating generations of children who loved the 'thunder lizard.' Scientific names often prove more fluid than the public realizes.

The Chinese discoveries of the 1920s and 1930s added important new species to the fossil record. Roy Chapman Andrews' expeditions to Mongolia uncovered Velociraptor, Protoceratops, and the first known dinosaur eggs. These discoveries revolutionized understanding of dinosaur reproduction and behavior. The expeditions also found early mammal fossils that would prove crucial for understanding mammalian evolution.""",

    "dinosaur_science_2": """The classification of dinosaurs underwent significant revision throughout the 20th century. Harry Seeley's 1888 division of dinosaurs into Saurischia (lizard-hipped) and Ornithischia (bird-hipped) remained the standard framework. However, the relationships within these groups were constantly debated and revised.

Sauropods - the long-necked giants like Brachiosaurus and Diplodocus - were particularly poorly understood in the 1930s. Scientists couldn't explain how such massive animals could have functioned. The semi-aquatic hypothesis suggested they lived in water to support their weight. We now know sauropods were fully terrestrial, with sophisticated respiratory systems and weight-bearing adaptations.

Theropods, the group including T. rex and eventually birds, were recognized as related to birds as early as Thomas Henry Huxley's work in the 1860s. However, the connection was downplayed for decades. The discovery of Archaeopteryx provided clear evidence of dinosaur-bird links, but many scientists resisted the implications. It wasn't until the 1960s and 1970s that the dinosaur-bird connection became widely accepted.

The extinction of non-avian dinosaurs remained mysterious in the 1930s. Various theories proposed climate change, disease, competition from mammals, or dietary changes. The asteroid impact hypothesis wouldn't be proposed until 1980, revolutionizing our understanding of mass extinctions. The 1930s scientific community couldn't have imagined that a cosmic catastrophe ended the dinosaur era.

Paleontological techniques in the 1930s were primitive by modern standards. Fossils were extracted with pickaxes and dynamite, often damaging specimens. Dating relied on relative stratigraphy rather than radiometric techniques. CT scanning, computer modeling, and molecular analysis were decades away. Yet the foundational work of this era established the framework that modern paleontology would build upon.""",

    "prehistoric_life": """The history of life on Earth extends far beyond the dinosaurs that capture popular imagination. Life began approximately 3.8 billion years ago, with the first cells appearing in Earth's primordial oceans. For billions of years, life remained microscopic - single-celled organisms slowly transforming the planet's atmosphere and chemistry.

The Cambrian Explosion, roughly 540 million years ago, saw the rapid diversification of complex multicellular life. In geological terms, 'rapid' means tens of millions of years, but this was remarkably fast for evolution. The Burgess Shale deposits in Canada preserve extraordinary fossils from this period - bizarre creatures like Anomalocaris, Opabinia, and Hallucigenia that don't fit neatly into modern animal categories.

The colonization of land began with plants around 470 million years ago, followed by arthropods and eventually vertebrates. The first tetrapods - four-limbed vertebrates - emerged from fish ancestors around 375 million years ago. Tiktaalik, discovered in 2004 but theorized for decades, represents this transition beautifully, with features of both fish and amphibians.

The Permian period saw the rise of synapsids - the lineage that would eventually produce mammals. These creatures, often misleadingly called 'mammal-like reptiles,' dominated terrestrial ecosystems before the dinosaurs. The Permian-Triassic extinction event, the worst mass extinction in Earth's history, nearly ended their reign. Over 90% of marine species and 70% of terrestrial species perished.

The Mesozoic Era - the Age of Dinosaurs - lasted from about 252 to 66 million years ago. This era is divided into the Triassic, Jurassic, and Cretaceous periods. Early dinosaurs were small and competed with other reptile groups. By the late Triassic, dinosaurs had begun their rise to dominance. The extinction of competing groups at the end of the Triassic opened ecological niches that dinosaurs would fill spectacularly in the Jurassic and Cretaceous.""",

    "depression_era_1": """The Great Depression that began in 1929 fundamentally transformed American society. The stock market crash of October 1929 triggered a catastrophic economic collapse, but the underlying causes ran deeper - agricultural overproduction, industrial overcapacity, speculation, and structural weaknesses in the banking system all contributed.

By 1933, unemployment reached approximately 25% of the workforce. Millions of Americans lost their homes, their savings, and their hope. Bread lines and soup kitchens became common sights in major cities. Hoovervilles - shanty towns named mockingly after President Herbert Hoover - appeared in vacant lots and public parks.

Franklin D. Roosevelt's election in 1932 brought dramatic changes. The New Deal programs attempted to provide relief, recovery, and reform. The Civilian Conservation Corps employed young men in conservation work. The Works Progress Administration funded arts, construction, and infrastructure projects. Social Security, established in 1935, created a safety net that persists today.

Entertainment during the Depression served crucial psychological functions. Movies offered escape into glamorous worlds far from economic hardship. Hollywood's golden age produced enduring classics - musicals, comedies, and adventures that let audiences forget their troubles for a few hours. Ticket prices remained low enough for most to afford occasional visits.

Radio transformed home entertainment during the 1930s. Families gathered around receivers for news, music, drama, and comedy. Programs like Amos 'n' Andy, The Lone Ranger, and The Shadow attracted millions of listeners. President Roosevelt's fireside chats used radio to communicate directly with Americans, building support for New Deal programs and restoring confidence in government.""",

    "depression_era_2": """The cultural production of the 1930s reflected the era's anxieties and aspirations. Literature ranged from John Steinbeck's social realism to the escapist adventures of pulp magazines. The Grapes of Wrath, published in 1939, captured the Dust Bowl migration with unflinching honesty. Meanwhile, The Shadow, Doc Savage, and countless other pulp heroes offered thrilling escapism.

Music evolved significantly during the Depression. Jazz continued its development, with swing becoming the dominant popular style by mid-decade. Big bands led by Benny Goodman, Duke Ellington, Count Basie, and Glenn Miller provided music for dancing and listening. The jukebox, introduced in 1927, spread recorded music to bars, diners, and restaurants across America.

Fashion reflected economic constraints and changing values. Women's hemlines dropped from the short skirts of the 1920s to more modest lengths. Elegant simplicity replaced the elaborate ornamentation of the previous decade. Men's suits featured wider lapels and higher-waisted trousers. Practicality and durability became important considerations for clothing purchases.

Sports provided distraction and heroes during difficult times. Baseball remained America's national pastime, with players like Babe Ruth, Lou Gehrig, and Joe DiMaggio achieving legendary status. Jesse Owens' triumph at the 1936 Berlin Olympics provided a powerful response to Nazi racial ideology. Boxing matches, particularly Joe Louis's victories, drew enormous audiences and radio listenership.

The New Deal's cultural programs created lasting artistic legacies. The Federal Writers' Project employed thousands of writers to document American life, producing state guides, folklore collections, and oral histories. The Federal Art Project supported visual artists, producing murals and sculptures that still adorn public buildings. The Federal Theatre Project staged productions across the country before its controversial termination in 1939.""",

    "entertainment_history_1": """The entertainment industry of the 1930s operated very differently from today's media landscape. Without television, internet, or streaming services, entertainment options were more limited but perhaps more communal. Families and communities gathered for shared experiences that created lasting memories and cultural touchstones.

Movie theaters were palaces of dreams. The great picture palaces built in the 1920s featured elaborate architecture, ornate decoration, and thousands of seats. Going to the movies was an event, complete with newsreels, cartoons, and sometimes live performances before the main feature. The studio system controlled production, distribution, and exhibition, creating a vertically integrated industry.

The major studios - MGM, Paramount, Warner Bros., RKO, and Fox - each developed distinctive styles and star rosters. MGM boasted it had 'more stars than there are in heaven.' Warner Bros. specialized in gritty urban dramas and musicals. Paramount cultivated sophisticated comedy and European elegance. These studio personalities shaped audience expectations and loyalty.

The Production Code, enforced from 1934, restricted movie content significantly. Violence couldn't be gratuitous, crime couldn't pay, and sexual content was heavily regulated. These restrictions pushed filmmakers toward clever indirection and sophisticated subtext. Some argue the constraints produced more creative solutions; others see them as censorship that infantilized audiences.

The transition to sound, completed by the early 1930s, transformed filmmaking aesthetics. Early talkies were often stage-bound, with stationary cameras recording theatrical performances. Gradually, filmmakers rediscovered mobility and visual storytelling. By the late 1930s, technical sophistication had returned, combined with the new possibilities of synchronized sound and dialogue.""",

    "entertainment_history_2": """Vaudeville, the variety show format that had dominated live entertainment for decades, declined rapidly in the 1930s. Movie theaters, which had once included vaudeville acts, increasingly showed only films. Radio provided free entertainment at home, reducing the incentive to attend live performances. Many vaudeville performers transitioned to radio, film, or nightclub work.

The nightclub scene flourished despite - or perhaps because of - Prohibition's repeal in 1933. The Cotton Club in Harlem showcased African American performers for predominantly white audiences, launching careers while perpetuating segregation. Jazz clubs in cities across America provided venues for musical innovation and social mixing that challenged conventional boundaries.

Amusement parks offered affordable family entertainment. Coney Island in New York remained a popular destination, though competition from newer parks increased. The 1933 Chicago World's Fair, Century of Progress, attracted millions of visitors with its celebration of scientific and industrial achievement. World's fairs combined entertainment with education, showcasing technological wonders and cultural exhibits.

Circuses still toured America, though the industry faced increasing challenges. The Ringling Bros. and Barnum & Bailey Circus remained the most prestigious, traveling by train to cities across the country. Smaller circuses struggled with competition from movies and radio. The circus would continue its long decline through subsequent decades.

Board games and parlor games enjoyed renewed popularity as families sought inexpensive home entertainment. Monopoly, patented in 1935, became an enormous success despite - or perhaps because of - its themes of real estate speculation and financial ruin. Jigsaw puzzles, card games, and word games helped families pass long evenings together.""",

    "science_technology_1": """Scientific and technological progress continued despite the Depression's economic constraints. The 1930s saw significant advances in physics, chemistry, medicine, and engineering. These developments would reshape the world in subsequent decades, though their full implications weren't always recognized at the time.

Physics underwent revolutionary changes in the early 20th century, and the 1930s saw continued development of quantum mechanics and relativity. The neutron was discovered in 1932, and artificial radioactivity was produced in 1934. These discoveries would prove crucial for nuclear physics development. Albert Einstein, having fled Nazi Germany, continued his work in America, though his quest for a unified field theory remained unfulfilled.

Aviation advanced rapidly during the 1930s. Charles Lindbergh's 1927 transatlantic flight had captured global imagination, and the following decade saw commercial aviation begin to mature. Passenger airlines connected major cities, though air travel remained expensive and somewhat risky. The China Clipper and other flying boats crossed oceans, shrinking the globe.

Automotive technology improved steadily. Cars became more reliable, comfortable, and affordable. Streamlined designs reflected both aesthetic preferences and engineering advances. The automobile transformed American geography, enabling suburban development and changing urban planning assumptions. By 1940, most American families owned at least one car.

Radio technology advanced significantly. FM radio was invented in 1933, though it wouldn't become commercially significant for decades. Television experiments progressed, with limited broadcasts occurring by the late 1930s. The BBC began regular television service in 1936, and RCA demonstrated television at the 1939 New York World's Fair. The television age was imminent.""",

    "science_technology_2": """Medical advances during the 1930s saved countless lives and laid groundwork for further progress. Sulfonamide drugs, the first antibiotics, were introduced in 1935, revolutionizing treatment of bacterial infections. Before antibiotics, simple infections could prove fatal; afterward, many previously deadly conditions became manageable.

Vitamin research clarified the role of nutrition in health. Deficiency diseases like scurvy, rickets, and pellagra were better understood and increasingly preventable. Food fortification programs added vitamins to common foods. Public health campaigns promoted better nutrition, though Depression-era poverty limited many families' access to adequate diets.

Psychiatric treatment remained primitive and often harmful by modern standards. Lobotomy, introduced in 1935, was promoted as a treatment for mental illness despite devastating effects on patients. Insulin shock therapy and electroconvulsive therapy were also introduced during this decade. Mental illness carried enormous stigma, and institutionalization was common.

Agricultural science addressed the Dust Bowl crisis that devastated the Great Plains. Soil conservation techniques, crop rotation, and drought-resistant varieties were developed and promoted. The Soil Conservation Service, established in 1935, worked to prevent erosion and restore damaged farmland. These efforts helped prevent future agricultural disasters, though they couldn't undo the damage already done.

Engineering achievements included major infrastructure projects. The Hoover Dam, completed in 1936, demonstrated massive construction capabilities while providing electricity and water to the growing Southwest. The Golden Gate Bridge, finished in 1937, became an iconic symbol of American engineering prowess. The Empire State Building, completed in 1931, remained the world's tallest building for decades.""",

    "art_culture_1": """The visual arts of the 1930s reflected the era's social concerns and aesthetic debates. Realism dominated American painting, with artists like Thomas Hart Benton, Grant Wood, and John Steuart Curry celebrating rural American life. Their work, sometimes called Regionalism, offered an alternative to European modernism that many Americans found alien.

The Federal Art Project employed thousands of artists during the Depression. Murals painted in post offices, schools, and government buildings across America depicted local history, industry, and community life. These works brought art to communities that had never had public art before. Some murals celebrated labor and working people in ways that later seemed politically charged.

Photography emerged as a powerful documentary medium. The Farm Security Administration employed photographers including Dorothea Lange, Walker Evans, and Gordon Parks to document rural poverty. Their images of Dust Bowl migrants and impoverished farmers remain iconic. These photographs shaped public perception of Depression hardships and built support for government assistance programs.

Surrealism and other modernist movements continued developing in Europe, with some influence reaching America. Salvador Dalí gained attention with his melting clocks and dream imagery. The Museum of Modern Art in New York, founded in 1929, promoted modernist art to American audiences. However, abstract and experimental art remained less popular in America than traditional and realist approaches.

Architecture transitioned from the elaborate ornamentation of earlier decades toward modernist simplicity. Art Deco, which had flourished in the 1920s, remained influential but began giving way to International Style modernism. The Chrysler Building and Empire State Building represented Art Deco's culmination. Frank Lloyd Wright continued his innovative work, completing Fallingwater in 1935.""",

    "art_culture_2": """Literary production during the 1930s reflected economic hardship, political turmoil, and social change. Proletarian literature, sympathetic to working-class struggles, gained prominence. Writers like John Dos Passos, James T. Farrell, and Richard Wright explored class conflict and social injustice. The Communist Party attracted many intellectuals, though most eventually became disillusioned.

The Southern Renaissance brought new attention to Southern literature. William Faulkner's experimental novels, including The Sound and the Fury and Absalom, Absalom!, established him as a major American writer. Robert Penn Warren, Eudora Welty, and other Southern writers developed distinctive voices exploring regional history and identity.

Genre fiction flourished in pulp magazines and paperback originals. Hardboiled detective fiction, pioneered by Dashiell Hammett and developed by Raymond Chandler, offered cynical perspectives on American society. Science fiction evolved from simple adventure stories toward more sophisticated speculation. Horror writers like H.P. Lovecraft developed influential mythologies that continue to inspire.

Children's literature produced enduring classics. Laura Ingalls Wilder's Little House series began publication in 1932. Dr. Seuss published his first book in 1937. These and other works shaped childhood reading for generations. Comic books emerged as a distinct medium, with Superman's debut in 1938 launching the superhero genre.

The Federal Writers' Project produced remarkable documentation of American life. State guidebooks provided detailed information about history, geography, and culture. Oral history projects recorded narratives from former slaves and other marginalized communities. These materials preserve voices that might otherwise have been lost to history."""
}

# Generate additional filler text by repeating and varying themes
def generate_filler_text(topic, variant):
    """Generate additional filler text on given topics."""
    fillers = {
        ("dinosaurs", 1): """Let me tell you more about the fascinating world of prehistoric creatures. The Mesozoic Era, spanning from roughly 252 to 66 million years ago, was divided into three periods: the Triassic, Jurassic, and Cretaceous. Each period saw different dominant species and ecological conditions.

The Triassic period saw the first true dinosaurs emerge, though they were relatively small and competed with other reptile groups. Early dinosaurs like Eoraptor and Herrerasaurus were bipedal predators no larger than modern dogs. The end-Triassic extinction event cleared ecological niches that dinosaurs would fill.

The Jurassic period brought the classic dinosaurs of popular imagination. Massive sauropods like Brachiosaurus and Diplodocus dominated landscapes. Predators like Allosaurus hunted in complex ecosystems. The first birds evolved from small feathered dinosaurs, with Archaeopteryx representing this transition.

The Cretaceous period saw dinosaur diversity reach its peak. Tyrannosaurus rex, Triceratops, and countless other species evolved during this time. Flowering plants appeared and diversified, changing terrestrial ecosystems dramatically. The period ended abruptly with the asteroid impact that caused the mass extinction.""",

        ("dinosaurs", 2): """Dinosaur behavior remains largely speculative, but fossil evidence provides some clues. Trackways show some dinosaurs traveled in herds, suggesting social behavior. Nesting sites reveal parental care in some species. Bone pathologies indicate fighting and disease.

The question of warm-bloodedness versus cold-bloodedness has been debated for decades. Evidence now suggests dinosaurs had metabolisms intermediate between modern reptiles and birds. Some may have been fully warm-blooded, others more reptilian. The diversity of dinosaurs likely meant diversity in physiology.

Dinosaur intelligence varied widely. Some theropods had brain-to-body ratios approaching modern birds, suggesting relatively high intelligence. Others had tiny brains relative to body size. Tool use and complex communication remain speculative but not impossible for some species.

The colors of dinosaurs were completely unknown until recently. Fossilized melanosomes now allow reconstruction of some dinosaur colors. Many had stripes, spots, or iridescent feathers. The image of dinosaurs as uniformly gray or green is giving way to more vibrant reconstructions.""",

        ("comics", 1): """The relationship between comics and other media has always been complex. Comic strips influenced early cinema, with many films adapting popular characters. The sequential storytelling of comics shares structural similarities with film editing. Both media developed during the same era, learning from each other.

Newspaper editors wielded enormous power over comic content. They could demand changes to storylines, character designs, or dialogue. Some strips were canceled for political content or controversial storylines. The pressure to appeal to broad audiences constrained creative expression.

International comics developed differently from American strips. European bandes dessinées allowed more mature content and longer narratives. Japanese manga evolved distinct storytelling conventions. Cross-cultural influence increased throughout the 20th century, though American strips dominated globally until recent decades.

The economics of cartooning changed dramatically over time. Early cartoonists could become wealthy from syndication deals. Later consolidation reduced payments and creative control. Today, webcomics offer new distribution models, though monetization remains challenging.""",

        ("comics", 2): """Comic strip artists developed distinctive visual vocabularies. Motion lines indicated movement. Speech balloons evolved from simple circles to varied shapes indicating tone. Sound effects became visual elements integrated into compositions. These conventions, now universal, were invented and refined through experimentation.

The daily deadline shaped comic strip storytelling profoundly. Each strip needed to function independently while advancing longer narratives. Recapping previous events without boring regular readers required skill. The Sunday strip format allowed more elaborate storytelling but reached different audiences.

Censorship affected comics throughout their history. The Comics Code Authority, established in 1954, regulated content for decades. Earlier, newspaper editors and public pressure shaped acceptable content. Comics often pushed boundaries subtly, using humor and fantasy to address serious topics.

The preservation of original comic art was neglected for decades. Newspapers discarded strips after publication. Artists gave away or lost originals. Only later did collectors and institutions recognize the historical and artistic value of original comic art. Many early strips survive only as newspaper clippings.""",

        ("1930s", 1): """Daily life in the 1930s differed dramatically from modern experience. Refrigerators were replacing iceboxes, but many homes still relied on ice delivery. Indoor plumbing wasn't universal, especially in rural areas. Electricity reached most urban homes but remained unavailable in much of rural America.

Communication was slower and more deliberate. Long-distance telephone calls were expensive luxuries. Letters took days to arrive. News traveled by radio and newspaper rather than instant digital updates. This slower pace shaped expectations and social rhythms.

Transportation options were more limited. Most travel was by train for long distances, car for medium distances, and foot or streetcar for local movement. Air travel existed but remained uncommon and expensive. The automobile was transforming American geography, but highway systems were primitive by modern standards.

Food came from different sources than today. Supermarkets were just emerging; most shopping happened at specialty shops - butchers, bakers, greengrocers. Home canning and preservation were essential skills. Seasonal availability affected diet more than it does today.""",

        ("1930s", 2): """Fashion in the 1930s reflected both economic constraints and changing aesthetics. The short skirts of the 1920s gave way to longer hemlines. Women's fashion emphasized elegant simplicity. Men wore suits with wider lapels and high-waisted trousers. Casual dress was less common than today.

Social conventions were more formal in many ways. Forms of address were more rigid. Dating followed prescribed patterns. Gender roles were more strictly defined. Yet the Depression also broke down some barriers, as economic necessity pushed women into workforce roles previously reserved for men.

Education was less universal than today. High school completion rates were lower, and college was an elite privilege. Vocational training was common. Literacy rates were high in urban areas but lower in rural and immigrant communities. Public libraries served crucial educational functions.

Healthcare was more limited and more dangerous. Antibiotics didn't exist until mid-decade. Surgery was riskier. Many conditions we now treat easily were then fatal or disabling. Medical costs, while lower in absolute terms, could still devastate families without insurance."""
    }
    return fillers.get((topic, variant), "")


def create_image_session(page_num, date):
    """Create a session for a comic page."""
    return {
        "session_id": f"D{page_num}",
        "date": date,
        "dialogues": [
            {
                "round": f"D{page_num}:1",
                "user": f"Here's page {page_num} of the Alley Oop comic I'm reading.",
                "assistant": f"I see page {page_num}. I'll keep track of what happens on this page as we continue through the comic.",
                "image_id": [f"D{page_num}:IMG_001"],
                "input_image": [f"ComicScene_Alley_Oop_Draft/Alley_Oop_Page_{page_num}.jpg"],
                "image_caption": [f"Alley Oop comic page {page_num}."]
            }
        ]
    }


def create_text_session(session_id, date, dialogues):
    """Create a text-only session with long passages."""
    session = {
        "session_id": session_id,
        "date": date,
        "dialogues": []
    }
    for i, (user, assistant) in enumerate(dialogues, start=1):
        session["dialogues"].append({
            "round": f"{session_id}:{i}",
            "user": user,
            "assistant": assistant
        })
    return session


def build_massive_dialog():
    """Build a massive dialog JSON with all 35 pages and extensive text."""

    # Load existing for QAs and character profile
    with open("/common/home/mg1998/MemoryBench/Benchmark_Pipeline/data/dialog/ComicScene_Alley_Oop_Draft.json") as f:
        existing = json.load(f)

    sessions = []
    day = 1

    # Pages 1-10 with heavy text between them
    for page in range(1, 11):
        # Image session
        sessions.append(create_image_session(page, f"1933-01-{day:02d}"))
        day += 1

        # Text session after each page
        if page == 1:
            text_session = create_text_session(
                f"T{page}", f"1933-01-{day:02d}",
                [
                    ("Tell me about the history of comic strips.", LONG_PASSAGES["comic_history_1"]),
                    ("That's fascinating. What else can you tell me?", LONG_PASSAGES["comic_history_2"]),
                ]
            )
            sessions.append(text_session)
            day += 1
        elif page == 2:
            text_session = create_text_session(
                f"T{page}", f"1933-01-{day:02d}",
                [
                    ("I'm curious about dinosaur science.", LONG_PASSAGES["dinosaur_science_1"]),
                    ("Tell me more about prehistoric life.", LONG_PASSAGES["dinosaur_science_2"]),
                ]
            )
            sessions.append(text_session)
            day += 1
        elif page == 3:
            text_session = create_text_session(
                f"T{page}", f"1933-01-{day:02d}",
                [
                    ("What was life like during the Great Depression?", LONG_PASSAGES["depression_era_1"]),
                    ("How did entertainment help people cope?", LONG_PASSAGES["depression_era_2"]),
                ]
            )
            sessions.append(text_session)
            day += 1
        elif page == 5:
            text_session = create_text_session(
                f"T{page}", f"1933-01-{day:02d}",
                [
                    ("Tell me about entertainment in the 1930s.", LONG_PASSAGES["entertainment_history_1"]),
                    ("What other entertainment was available?", LONG_PASSAGES["entertainment_history_2"]),
                ]
            )
            sessions.append(text_session)
            day += 1
        elif page == 7:
            text_session = create_text_session(
                f"T{page}", f"1933-01-{day:02d}",
                [
                    ("What about science and technology?", LONG_PASSAGES["science_technology_1"]),
                    ("Any other scientific advances?", LONG_PASSAGES["science_technology_2"]),
                ]
            )
            sessions.append(text_session)
            day += 1
        elif page == 9:
            text_session = create_text_session(
                f"T{page}", f"1933-01-{day:02d}",
                [
                    ("Tell me about art and culture.", LONG_PASSAGES["art_culture_1"]),
                    ("What about literature?", LONG_PASSAGES["art_culture_2"]),
                ]
            )
            sessions.append(text_session)
            day += 1

    # Pages 11-20 with more text
    for page in range(11, 21):
        sessions.append(create_image_session(page, f"1933-01-{day:02d}"))
        day += 1

        if page in [12, 15, 18]:
            topic = "dinosaurs" if page == 12 else ("comics" if page == 15 else "1930s")
            variant = 1 if page in [12, 15] else 2
            filler = generate_filler_text(topic, variant)
            if filler:
                text_session = create_text_session(
                    f"T{page}", f"1933-01-{day:02d}",
                    [
                        (f"Let's discuss more about {topic}.", filler),
                        ("Interesting. Can you elaborate?", LONG_PASSAGES.get("prehistoric_life", "The history continues...")),
                    ]
                )
                sessions.append(text_session)
                day += 1

    # Pages 21-35 with remaining text
    for page in range(21, 36):
        sessions.append(create_image_session(page, f"1933-02-{(page-20):02d}"))

        if page in [22, 25, 28, 31, 34]:
            topic = ["dinosaurs", "comics", "1930s", "dinosaurs", "comics"][([22, 25, 28, 31, 34].index(page))]
            variant = 2 if page > 28 else 1
            filler = generate_filler_text(topic, variant)
            if filler:
                text_session = create_text_session(
                    f"T{page}", f"1933-02-{(page-19):02d}",
                    [
                        (f"More thoughts on {topic}?", filler),
                    ]
                )
                sessions.append(text_session)

    result = {
        "character_profile": existing["character_profile"],
        "multi_session_dialogues": sessions,
        "human-annotated QAs": existing["human-annotated QAs"]
    }

    return result


def main():
    output_path = Path("/common/home/mg1998/MemoryBench/Benchmark_Pipeline/data/dialog/ComicScene_Alley_Oop_Draft.json")

    print("Building massive dialog JSON...")
    data = build_massive_dialog()

    # Count statistics
    total_chars = 0
    total_rounds = 0
    image_rounds = 0

    for session in data["multi_session_dialogues"]:
        for d in session["dialogues"]:
            total_rounds += 1
            total_chars += len(d.get("user", ""))
            total_chars += len(d.get("assistant", ""))
            if "input_image" in d:
                image_rounds += 1

    text_tokens = total_chars / 4
    image_tokens = image_rounds * 1100
    total_tokens = text_tokens + image_tokens

    print(f"\n=== Statistics ===")
    print(f"Total sessions: {len(data['multi_session_dialogues'])}")
    print(f"Total rounds: {total_rounds}")
    print(f"  - With images: {image_rounds}")
    print(f"  - Text-only: {total_rounds - image_rounds}")
    print(f"\n=== Token Estimates ===")
    print(f"Text characters: {total_chars:,}")
    print(f"Text tokens: ~{int(text_tokens):,}")
    print(f"Image tokens: ~{int(image_tokens):,}")
    print(f"TOTAL: ~{int(total_tokens):,} tokens")

    if total_tokens > 100000:
        print(f"\n⚠️  Approaching 128K limit!")

    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
