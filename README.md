# Documentation for `main.py`
# Currently Outdated

## Overview
This script is designed to load mentee and mentor data from CSV files, process the data, and match mentees with mentors based on various criteria. The matches are then saved to an output CSV file.

## Dependencies
- pandas
- numpy
- typing
- scipy
- sentence_transformers
- sklearn
- mentoringClasses (custom module containing `Mentee` and `Mentor` classes)

## Functions

### `load_mentees_from_csv(mentee_csv) -> List[Mentee]`
Loads mentee data from a CSV file and returns a list of `Mentee` objects.

- **Parameters:**
  - `mentee_csv` (str): Path to the mentee CSV file.
- **Returns:**
  - List of `Mentee` objects.

### `load_mentors_from_csv(mentors_csv) -> List[Mentor]`
Loads mentor data from a CSV file and returns a list of `Mentor` objects.

- **Parameters:**
  - `mentors_csv` (str): Path to the mentor CSV file.
- **Returns:**
  - List of `Mentor` objects.

### `match_mentees_and_mentors(mentees: List[Mentee], mentors: List[Mentor], output_csv='matches.csv')`
Matches mentees with mentors based on various criteria and saves the matches to a CSV file.

- **Parameters:**
  - `mentees` (List[Mentee]): List of `Mentee` objects.
  - `mentors` (List[Mentor]): List of `Mentor` objects.
  - `output_csv` (str, optional): Path to the output CSV file. Default is 'matches.csv'.
- **Returns:**
  - None

## Matching Process
1. **Load Sentence Transformer Model:**
   - Uses `SentenceTransformer('all-MiniLM-L6-v2')` to encode text data for similarity comparison.

2. **Separate Mentors:**
   - Mentors are divided into those who can take only one mentee (`mentors_single`) and those who can take multiple mentees (`mentors_multiple`).

3. **Define Capacity:**
   - Sets the capacity for each mentor based on their ability to take multiple mentees.

4. **Perform Matching:**
   - Matches are performed in two phases:
     - **Phase 1:** Matches mentees with mentors who can take only one mentee.
     - **Phase 2:** Matches remaining mentees with mentors who can take multiple mentees.
   - Uses cosine similarity to compare text embeddings of mentee and mentor descriptions.
   - Applies hard constraints based on gender and origin preferences.
   - Uses the Hungarian algorithm (`linear_sum_assignment`) to find the optimal matching based on compatibility scores.

5. **Save Matches:**
   - Saves the matches to the specified output CSV file.
   - Prints unmatched mentees, if any.

## Usage
To run the script, execute the following command:

python main.py

This will load mentee and mentor data from `mentees.csv` and `mentors.csv`, perform the matching, and save the results to `matches.csv`.

## Example CSV Structure
### Mentees CSV
| id | full_name | gender | contact_email | locations | intro | looking_for | state_of_origin | country_of_origin | gender_preference | prefers_from_origin |
|----|-----------|--------|---------------|-----------|-------|-------------|-----------------|-------------------|-------------------|---------------------|
| 1  | John Doe  | male   | john@example.com | loc1;loc2 | Intro text | Looking for text | VIC | AU | true | true |

### Mentors CSV
| id | full_name | gender | contact_email | locations | multiple | intro | state_of_origin | country_of_origin | gender_preference |
|----|-----------|--------|---------------|-----------|----------|-------|-----------------|-------------------|-------------------|
| 1  | Jane Smith| female | jane@example.com | loc1;loc2 | true | Intro text | VIC | AU | false |

## Notes
- Ensure that the `mentoringClasses` module is available and contains the `Mentee` and `Mentor` classes.
- Adjust the CSV file paths as needed.


# TO DO
Hub for monash medical students and graduate doctors - find research projects, opportunities, mentoring connections, career advice
Web platform
Make profile, link socials, upload resume

