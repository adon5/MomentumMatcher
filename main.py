import pandas as pd
import numpy as np
from typing import List, Set
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    gender_preference: bool = False  # True means prefer OWN gender -> This is an assumption we might want to change.

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

def load_mentees_from_csv(mentee_csv) -> List[Mentee]:
    mentee_dataframe = pd.read_csv(mentee_csv)
    mentees = []
    for index, row in mentee_dataframe.iterrows():
        id = int(row['id'])
        full_name = str(row['full_name'])
        gender = str(row['gender']).strip().lower()
        contact_email = str(row['contact_email'])

        if pd.notnull(row.get('locations')):
            locations = set(map(str.strip, str(row['locations']).split(';')))
        else:
            locations = set()

        intro = str(row.get('intro', 'Not provided'))
        looking_for = str(row.get('looking_for', 'Not provided'))
        state_of_origin = str(row.get('state_of_origin', 'VIC'))
        country_of_origin = str(row.get('country_of_origin', 'AU'))

        gender_preference = row.get('gender_preference', False)
        if isinstance(gender_preference, str):
            gender_preference = gender_preference.strip().lower() == 'true'
        else:
            gender_preference = bool(gender_preference)

        prefers_from_origin = row.get('prefers_from_origin', True)
        if isinstance(prefers_from_origin, str):
            prefers_from_origin = prefers_from_origin.strip().lower() == 'true'
        else:
            prefers_from_origin = bool(prefers_from_origin)

        mentee = Mentee(
            id=id,
            full_name=full_name,
            gender=gender,
            contact_email=contact_email,
            locations=locations,
            intro=intro,
            looking_for=looking_for,
            state_of_origin=state_of_origin,
            country_of_origin=country_of_origin,
            gender_preference=gender_preference,
            prefers_from_origin=prefers_from_origin
        )
        mentees.append(mentee)
    return mentees

def load_mentors_from_csv(mentors_csv) -> List[Mentor]:
    mentors_dataframe = pd.read_csv(mentors_csv)
    mentors = []
    for index, row in mentors_dataframe.iterrows():
        id = int(row['id'])
        full_name = str(row['full_name'])
        gender = str(row['gender']).strip().lower()
        contact_email = str(row['contact_email'])

        if pd.notnull(row.get('locations')):
            locations = set(map(str.strip, str(row['locations']).split(';')))
        else:
            locations = set()

        capacity = row.get('capacity', 1)
        if pd.isnull(capacity) or capacity == '':
            capacity = 1
        else:
            capacity = int(capacity)

        intro = str(row.get('intro', 'Not provided'))
        state_of_origin = str(row.get('state_of_origin', 'VIC'))
        country_of_origin = str(row.get('country_of_origin', 'AU'))

        gender_preference = row.get('gender_preference', False)
        if isinstance(gender_preference, str):
            gender_preference = gender_preference.strip().lower() == 'true'
        else:
            gender_preference = bool(gender_preference)

        mentor = Mentor(
            id=id,
            full_name=full_name,
            gender=gender,
            contact_email=contact_email,
            locations=locations,
            capacity=capacity,
            intro=intro,
            state_of_origin=state_of_origin,
            country_of_origin=country_of_origin,
            gender_preference=gender_preference
        )
        mentors.append(mentor)
    return mentors

def match_mentees_and_mentors(mentees: List[Mentee], mentors: List[Mentor], output_csv='matches.csv'):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    expanded_mentors = []
    mentor_id_to_indices = {}
    idx = 0
    for mentor in mentors:
        mentor_id_to_indices[mentor.id] = []
        for i in range(mentor.capacity):
            expanded_mentors.append(mentor)
            mentor_id_to_indices[mentor.id].append(idx)
            idx += 1

    mentor_intros = [mentor.intro for mentor in expanded_mentors]
    mentor_embeddings = model.encode(mentor_intros)
    mentee_lookings = [mentee.looking_for for mentee in mentees]
    mentee_embeddings = model.encode(mentee_lookings)

    mentee_indices = []
    mentor_indices = []
    compatibility_scores = []

    score_breakdown = {}

    GENDER_MISMATCH_PENALTY = 20
    ORIGIN_MISMATCH_PENALTY = 5
    LOCATION_WEIGHT = 3
    STATE_MATCH_WEIGHT = 2
    COUNTRY_MATCH_WEIGHT = 1
    TEXT_SIMILARITY_WEIGHT = 5

    for mentee_idx, mentee in enumerate(mentees):
        for mentor_idx, mentor in enumerate(expanded_mentors):
            score = 0
            gender_mismatch_penalty = 0
            origin_mismatch_penalty = 0
            location_score = 0
            state_match_score = 0
            country_match_score = 0
            text_similarity_score = 0

            if mentee.gender_preference and mentee.gender != mentor.gender:
                gender_mismatch_penalty -= GENDER_MISMATCH_PENALTY
            if mentor.gender_preference and mentee.gender != mentor.gender:
                gender_mismatch_penalty -= GENDER_MISMATCH_PENALTY

            score += gender_mismatch_penalty

            if mentee.prefers_from_origin and (
                mentee.state_of_origin != mentor.state_of_origin or
                mentee.country_of_origin != mentor.country_of_origin
            ):
                origin_mismatch_penalty -= ORIGIN_MISMATCH_PENALTY

            score += origin_mismatch_penalty

            location_overlap = len(mentee.locations & mentor.locations)
            location_score = location_overlap * LOCATION_WEIGHT  
            score += location_score

            if mentee.state_of_origin == mentor.state_of_origin:
                state_match_score = STATE_MATCH_WEIGHT  
            else:
                state_match_score = 0
            score += state_match_score

            if mentee.country_of_origin == mentor.country_of_origin:
                country_match_score = COUNTRY_MATCH_WEIGHT  
            else:
                country_match_score = 0
            score += country_match_score

            mentee_embedding = mentee_embeddings[mentee_idx]
            mentor_embedding = mentor_embeddings[mentor_idx]
            text_similarity = cosine_similarity(
                [mentee_embedding],
                [mentor_embedding]
            )[0][0]
            text_similarity_score = text_similarity * TEXT_SIMILARITY_WEIGHT
            score += text_similarity_score

            mentee_indices.append(mentee_idx)
            mentor_indices.append(mentor_idx)
            compatibility_scores.append(score)

            key = (mentee_idx, mentor_idx)
            score_breakdown[key] = {
                'total_score': score,
                'gender_mismatch_penalty': gender_mismatch_penalty,
                'origin_mismatch_penalty': origin_mismatch_penalty,
                'location_score': location_score,
                'state_match_score': state_match_score,
                'country_match_score': country_match_score,
                'text_similarity_score': text_similarity_score
            }

    num_mentees = len(mentees)
    num_mentors = len(expanded_mentors)
    cost_matrix = np.full((num_mentees, num_mentors), np.inf)

    for i in range(len(compatibility_scores)):
        mentee_idx = mentee_indices[i]
        mentor_idx = mentor_indices[i]
        score = compatibility_scores[i]
        cost_matrix[mentee_idx, mentor_idx] = -score

    # Replace any infinite values with a large finite number
    cost_matrix[np.isinf(cost_matrix)] = 1e6

    # Apply the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    assigned_mentors = {}
    for mentee_idx, mentor_idx in zip(row_ind, col_ind):
        mentee = mentees[mentee_idx]
        mentor = expanded_mentors[mentor_idx]
        total_score = -cost_matrix[mentee_idx, mentor_idx]
        mentor_id = mentor.id

        assigned_count = assigned_mentors.get(mentor_id, 0)
        if assigned_count < mentor.capacity:
            breakdown = score_breakdown[(mentee_idx, mentor_idx)]
            matches.append({
                'mentee_id': mentee.id,
                'mentor_id': mentor.id,
                'mentee_name': mentee.full_name,
                'mentor_name': mentor.full_name,
                'mentee_email': mentee.contact_email,
                'mentor_email': mentor.contact_email,
                'mentee_intro': mentee.intro,
                'mentor_intro': mentor.intro,
                'compatibility_score': total_score,
                'gender_mismatch_penalty': breakdown['gender_mismatch_penalty'],
                'origin_mismatch_penalty': breakdown['origin_mismatch_penalty'],
                'location_score': breakdown['location_score'],
                'state_match_score': breakdown['state_match_score'],
                'country_match_score': breakdown['country_match_score'],
                'text_similarity_score': breakdown['text_similarity_score']
            })
            assigned_mentors[mentor_id] = assigned_count + 1
        else:
            # Mentor capacity reached, mentee remains unmatched
            continue

    matched_mentee_ids = {match['mentee_id'] for match in matches}
    unmatched_mentees = [mentee for mentee in mentees if mentee.id not in matched_mentee_ids]

    if matches:
        matches_df = pd.DataFrame(matches)
        matches_df.to_csv(output_csv, index=False)
        print(f"{len(matches)} matches were made.")
        print(f"Optimal matches have been saved to {output_csv}")
    else:
        print("No matches could be made.")

    if unmatched_mentees:
        print("The following mentees could not be matched:")
        for mentee in unmatched_mentees:
            print(f"- {mentee.full_name} (ID: {mentee.id})")


if __name__ == "__main__":
    mentees = load_mentees_from_csv('mentees.csv')
    mentors = load_mentors_from_csv('mentors.csv')
    match_mentees_and_mentors(mentees, mentors, output_csv='matches.csv')
