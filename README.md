# Mentor-Mentee Matching System Documentation

## Overview

This Python script implements a mentor-mentee matching system using various data processing and machine learning techniques. The system reads mentor and mentee data from CSV files, processes the information, and uses a combination of text similarity and other criteria to create optimal matches.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Data Structures](#data-structures)
3. [Functions](#functions)
4. [Matching Algorithm](#matching-algorithm)
5. [Usage](#usage)

## Dependencies

The script requires the following Python libraries:

- pandas
- numpy
- scipy
- sentence_transformers
- sklearn

Ensure these are installed before running the script.

## Data Structures

### Mentor

```python
@dataclass
class Mentor:
    id: int
    full_name: str
    gender: str
    contact_email: str
    locations: Set[str]
    capacity: int = 1
    intro: str = 'Not provided'
    state_of_origin: str = "VIC"
    country_of_origin: str = "AU"
    gender_preference: bool = False
```

### Mentee

```python
@dataclass
class Mentee:
    id: int
    full_name: str
    gender: str
    contact_email: str
    locations: Set[str]
    intro: str = 'Not provided'
    looking_for: str = 'Not provided'
    state_of_origin: str = "VIC"
    country_of_origin: str = "AU"
    gender_preference: bool = False  
    prefers_from_origin: bool = True
```

## Functions

### `load_mentees_from_csv(mentee_csv) -> List[Mentee]`

Loads mentee data from a CSV file and returns a list of Mentee objects.

### `load_mentors_from_csv(mentors_csv) -> List[Mentor]`

Loads mentor data from a CSV file and returns a list of Mentor objects.

### `match_mentees_and_mentors(mentees: List[Mentee], mentors: List[Mentor], output_csv='matches.csv')`

The main function that performs the matching algorithm and outputs the results to a CSV file.

## Matching Algorithm

The matching algorithm uses the following steps:

1. Expand mentors based on their capacity.
2. Generate text embeddings for mentor intros and mentee "looking for" statements using SentenceTransformer.
3. Calculate compatibility scores based on various factors:
   - Gender preference match
   - Origin preference match
   - Location overlap
   - State and country match
   - Text similarity between mentor intro and mentee "looking for" statement
4. Create a cost matrix based on compatibility scores.
5. Apply the Hungarian Algorithm (linear sum assignment) to find optimal matches.
6. Save matches to a CSV file and report unmatched mentees.

### Scoring Factors

- `GENDER_MISMATCH_PENALTY`: 20
- `ORIGIN_MISMATCH_PENALTY`: 5
- `LOCATION_WEIGHT`: 3
- `STATE_MATCH_WEIGHT`: 2
- `COUNTRY_MATCH_WEIGHT`: 1
- `TEXT_SIMILARITY_WEIGHT`: 5

## Usage

1. Prepare two CSV files: `mentees.csv` and `mentors.csv` with the required fields.
2. Run the script:

```python
if __name__ == "__main__":
    mentees = load_mentees_from_csv('mentees.csv')
    mentors = load_mentors_from_csv('mentors.csv')
    match_mentees_and_mentors(mentees, mentors, output_csv='matches.csv')
```

3. The script will output the matches to `matches.csv` and print information about the matching process, including any unmatched mentees.

## Note

This system assumes certain defaults and scoring weights. Adjust these values in the `match_mentees_and_mentors` function to fine-tune the matching process for specific use cases.
