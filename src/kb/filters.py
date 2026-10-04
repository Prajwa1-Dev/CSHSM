import re

def safety_filter(counter_speech: str) -> tuple[bool, str]:
    """
    Apply conservative safety rules to counter-speech.
    Returns (is_safe, rejection_reason)
    """
    cs_lower = counter_speech.lower()
    
    # 1. Hate reinforcement (e.g., agreeing with the hate text explicitly without reframing)
    if re.search(r"\b(i agree|you are right|exactly|that's true)\b", cs_lower) and not re.search(r"but|however|although", cs_lower):
        return False, "hate reinforcement"
        
    # 2. Counter-hate (attacking the speaker directly)
    if re.search(r"\b(you are an idiot|you're stupid|shut up|fuck off|fuck you|go to hell)\b", cs_lower):
        return False, "counter-hate"
        
    # 3. Threats / Violent Escalation
    if re.search(r"\b(i will kill|we should kill|die|deserve to die|hope you die)\b", cs_lower):
        return False, "threats or violent escalation"
        
    # 4. Discriminatory / Harmful content (Profanity/slurs that aren't clearly quoted)
    slurs = ["nigger", "faggot", "retard", "kike", "chink", "spic"]
    for slur in slurs:
        if slur in cs_lower:
            # Check if it's quoted:
            if f'"{slur}"' not in cs_lower and f"'{slur}'" not in cs_lower:
                return False, "discriminatory language"
                
    return True, ""

def leakage_filter(hate_text: str, test_set_texts: set, val_set_texts: set) -> tuple[bool, str]:
    """
    Check if the hate_text overlaps exactly with HASTIKA test or validation sets.
    """
    ht_norm = hate_text.strip().lower()
    
    if ht_norm in test_set_texts:
        return False, "Leakage: Exact match in HASTIKA Test Set"
    if ht_norm in val_set_texts:
        return False, "Leakage: Exact match in HASTIKA Validation Set"
        
    return True, ""
