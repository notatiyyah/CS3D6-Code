def resolve_overlaps_longest_span(entities):
    """
    Sorts entities to prioritize the longest span. 
    Drops any nested/shorter entities that intersect with an already-accepted span.
    Returns the original entity dictionaries untouched.
    """
    if not entities:
        return []
        
    # Sort by start index ascending, then by span length DESCENDING.
    entities.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    
    resolved_entities = []
    
    for current_ent in entities:
        is_overlapping = False
        c_start, c_end = current_ent["start"], current_ent["end"]
        
        # Check against spans we've already accepted
        for accepted_ent in resolved_entities:
            a_start, a_end = accepted_ent["start"], accepted_ent["end"]
            
            # Mathematical condition for span intersection
            if max(c_start, a_start) < min(c_end, a_end):
                is_overlapping = True
                break
                
        # If it doesn't collide, keep the whole original dictionary
        if not is_overlapping:
            resolved_entities.append(current_ent)
            
    return resolved_entities

def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0