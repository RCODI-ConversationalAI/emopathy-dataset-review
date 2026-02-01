import os
import rispy
import bibtexparser
import re
import logging

# Set up logging to trace the process
logging.basicConfig(filename='process_trace.log', level=logging.INFO, format='%(message)s')

def analyze_databases(database_dir):
    global_results = {
        'emotion': {'papers': set(), 'duplicates': 0},
        'empathy': {'papers': set(), 'duplicates': 0},
        'total_papers': set(),
        'total_duplicates': 0
    }
    database_results = {}

    patterns_ris = {
        'empathy': re.compile(r'empath\w*', re.IGNORECASE),
    }

    patterns_bib = {
        'empathy': re.compile(r'empath\w*', re.IGNORECASE),
        'emotion': re.compile(r'emot\w*', re.IGNORECASE),
        'task': re.compile(r'(detect\w*|recog\w*|predict\w*|classi\w*|regress\w*|generat\w*)', re.IGNORECASE)
    }

    global_title_set = set()

    for filename in os.listdir(database_dir):
        file_path = os.path.join(database_dir, filename)
        if filename.endswith('.ris'):
            database_name = filename[:-4]
            logging.info(f"Processing RIS database: {database_name}")
            database_results[database_name] = analyze_ris_database(
                file_path,
                global_results, global_title_set,
                patterns_ris
            )
        elif filename.endswith('.bib'):
            database_name = filename[:-4]
            logging.info(f"Processing BibTeX database: {database_name}")
            database_results[database_name] = analyze_bib_database(
                file_path,
                global_results, global_title_set,
                patterns_bib
            )

    return database_results, global_results

def analyze_ris_database(file_path, global_results, global_title_set, patterns):
    with open(file_path, 'r', encoding='utf-8') as bibliography_file:
        entries = list(rispy.load(bibliography_file))

    local_results = {
        'total_papers': len(entries),
        'emotion': {'papers': set(), 'duplicates': 0},
        'empathy': {'papers': set(), 'duplicates': 0},
    }

    logging.info(f"Number of papers in {os.path.basename(file_path)}: {len(entries)}")

    for entry in entries:
        title = entry.get('title', '').strip().lower()
        abstract = entry.get('abstract', '').strip().lower()
        combined_text = f"{title} {abstract}"

        if not title:
            logging.warning("Entry without a title found. Skipping.")
            continue

        # Categorize the paper
        if patterns['empathy'].search(combined_text):
            category = 'empathy'
        else:
            category = 'emotion'

        if title in global_title_set:
            global_results['total_duplicates'] += 1
            global_results[category]['duplicates'] += 1
            local_results[category]['duplicates'] += 1
            logging.info(f"Duplicate found in category {category}: {title}")
            continue

        global_title_set.add(title)
        global_results['total_papers'].add(title)

        global_results[category]['papers'].add(title)
        local_results[category]['papers'].add(title)

    return local_results

def analyze_bib_database(file_path, global_results, global_title_set, patterns):
    with open(file_path, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    entries = bib_database.entries
    local_results = {
        'total_papers': 0,  # Initialize to zero, will increment after filtering
        'emotion': {'papers': set(), 'duplicates': 0},
        'empathy': {'papers': set(), 'duplicates': 0},
    }

    logging.info(f"Total entries in {os.path.basename(file_path)}: {len(entries)}")

    for entry in entries:
        title = entry.get('title', '').strip().lower()
        abstract = entry.get('abstract', '').strip().lower()
        combined_text = f"{title} {abstract}"

        if not title:
            logging.warning("Entry without a title found. Skipping.")
            continue

        # Categorize the paper based on given criteria
        empathy_match = patterns['empathy'].search(combined_text) and patterns['task'].search(combined_text)
        emotion_match = patterns['emotion'].search(combined_text) and patterns['task'].search(combined_text)

        if empathy_match:
            category = 'empathy'
        elif emotion_match:
            category = 'emotion'
        else:
            logging.info(f"Entry does not match any category: {title}")
            continue  # Skip entries that do not match the criteria

        # Increment total_papers after filtering
        local_results['total_papers'] += 1

        if title in global_title_set:
            global_results['total_duplicates'] += 1
            global_results[category]['duplicates'] += 1
            local_results[category]['duplicates'] += 1
            logging.info(f"Duplicate found in category {category}: {title}")
            continue

        global_title_set.add(title)
        global_results['total_papers'].add(title)

        global_results[category]['papers'].add(title)
        local_results[category]['papers'].add(title)

    logging.info(f"Number of papers matching criteria in {os.path.basename(file_path)}: {local_results['total_papers']}")

    return local_results

def print_and_save_stats(database_results, global_results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        def write_line(line):
            print(line)
            f.write(line + '\n')

        write_line("Statistics by database:")
        for db, results in database_results.items():
            write_line(f"\nDatabase: {db}")
            write_line(f"Total papers in database: {results['total_papers']}")
            write_line(f"Empathy papers: {len(results['empathy']['papers'])}")
            write_line(f"Emotion papers: {len(results['emotion']['papers'])}")
            write_line(f"Duplicates in 'empathy': {results['empathy']['duplicates']}")
            write_line(f"Duplicates in 'emotion': {results['emotion']['duplicates']}")

        write_line("\nOverall Statistics:")
        write_line(f"Total unique papers across all databases: {len(global_results['total_papers'])}")
        write_line(f"Total duplicates removed: {global_results['total_duplicates']}")
        write_line(f"Total duplicates in 'empathy': {global_results['empathy']['duplicates']}")
        write_line(f"Total duplicates in 'emotion': {global_results['emotion']['duplicates']}")
        write_line(f"Total empathy papers: {len(global_results['empathy']['papers'])}")
        write_line(f"Total emotion papers: {len(global_results['emotion']['papers'])}")

def main():
    database_dir = '/Volumes/ssd/01-ckj-postdoc/LLM-CCR/boolean-search/all-zot-items'
    database_results, global_results = analyze_databases(database_dir)

    stats_output_file = 'analysis_statistics.txt'
    print_and_save_stats(database_results, global_results, stats_output_file)
    print(f"Statistics saved to {stats_output_file}")
    print("Process tracing saved to 'process_trace.log'")

if __name__ == "__main__":
    main()