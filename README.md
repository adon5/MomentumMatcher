# Momentum Matcher

This was a one-time script for matching mentors and mentees for the [MUMUS Momentum mentoring program](https://www.mumusmomentum.org/) based on Google form responses, exported to csv. The mentors and mentees had no knowledge about each other (i.e. they did not rank each other, they just input information about themselves and preferences). View the full script in main.py. 

It is not very useful for anyone unless you have columns strictly in the format specified. Today, you could just boot up Claude Code or Cursor Agent, etc to write your own script.

## Installation

This still uses requirements.txt so:  

Set up a virtual environment:  
`python -m venv .venv`  

Run the virtual environment:  
`source .venv/bin/activate` (Linux or MacOS)  
    or  
`.venv\Scripts\activate` (Windows)  

Install dependencies from requirements.txt:  
`pip install -r requirements.txt`

## How it works

1. Expand mentors based on their capacity
2. Generate text embeddings for mentor intros and mentee "looking for" statements using sentence transformers.
3. Calculate compatibility scores, an integer:
   - Gender preference match: `GENDER_MISMATCH_PENALTY = 20`
   - Origin preference match: `ORIGIN_MISMATCH_PENALTY = 5`
   - Location overlap: `LOCATION_WEIGHT = 3`
   - State and country match: `STATE_MATCH_WEIGHT = 2`, `COUNTRY_MATCH_WEIGHT = 1`
   - Text similarity between mentor intro and mentee "looking for" statement: `TEXT_SIMILARITY_WEIGHT = 5`
4. Create a cost matrix based on compatibility scores and apply linear sum assignment
5. Save matches to a CSV file and report unmatched mentees.

*We have to create dummy mentors to handle multiple mentees to one mentor. We could model it as a network flow problem instead.*
*We use nested for loops to generate a rectangular matrix. It is very inefficient compared to just vectorizing the scores and comparisons. I will not change it for historical preservation purposes.*

## Usage

1. Prepare mentees.csv and mentors.csv with the required fields.
   See mentees.csv, mentors.csv as example data. See forms.txt for the form text.
3. Run the script, after ensuring you are in the right directory:
   
    `py main.py` (Windows)  
        or  
    `python3 main.py` (MacOS/Linux)

4. Matches will be output to matches.csv. Details about the matching process will be output into the terminal, including any unmatched mentees.


