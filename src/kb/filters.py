import re
from typing import Tuple


def _normalize_quotes(text: str) -> str:
    """Normalize curved/smart quotes to standard single/double quotes."""
    return text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def safety_filter(counter_speech: str) -> Tuple[bool, str]:
    """
    Apply conservative safety rules to counter-speech.
    Supports English and code-mixed Romanized Kannada (Kanglish) across six categories:
      Gender, Political, Religion, Geo-political (Regional), Violence, and Others.

    Safety Gates:
      1. Hate Reinforcement: Explicit agreement with hateful premise without reframing.
      2. Counter-Hate: Attacking, insulting, or shaming the original speaker.
      3. Threats & Violent Escalation: Threats of violence, bodily harm, killing, or death commands.
      4. Discriminatory Language: Unquoted slurs, abusive profanity, or communal/regional hostility.

    Returns:
      (is_safe, rejection_reason): Tuple of boolean status and standardized reason string.
    """
    if not counter_speech or not counter_speech.strip():
        return False, "empty content"

    cs_norm = _normalize_quotes(counter_speech)
    cs_lower = cs_norm.lower()

    # =========================================================================
    # 1. Hate reinforcement (agreeing with hate premise without reframing)
    # =========================================================================
    en_agree = r"\b(i agree|you are right|exactly|that's true)\b"
    kn_agree = (
        r"\b(naan[u]?\s*agree\s*maadt[ie]n[i|e]|naavu\s*agree\s*maadt[ie]v[i|e]|"
        r"correct\s*aagi\s*helidr[i|e]|correct\s*helidr[i|e]|"
        r"sariyaagi\s*helidr[i|e]|sari\s*helidr[i|e]|"
        r"nimm[a]?\s*mathu\s*nija|neevu\s*helidd[u]?\s*nija|"
        r"idu\s*nija|idu\s*sathyavada\s*mathu)\b"
    )
    reframing = r"\b(but|however|although|aadre|aadare|aadag[iy]u|adre|adare|aadroo|aadru)\b"

    has_agreement = bool(re.search(en_agree, cs_lower) or re.search(kn_agree, cs_lower))
    has_reframing = bool(re.search(reframing, cs_lower))

    if has_agreement and not has_reframing:
        return False, "hate reinforcement"

    # =========================================================================
    # 2. Counter-hate (attacking, demeaning, or shaming the speaker directly)
    # =========================================================================
    en_counter_hate = r"\b(you are an idiot|you're stupid|shut up|fuck off|fuck you|go to hell)\b"
    if re.search(en_counter_hate, cs_lower):
        return False, "counter-hate"

    kn_counter_hate = (
        # Commands to shut up / vulgar dismissals
        r"\b(baay[i]?\s*much[u|o|i]?|baay[i]?\s*muchkond\w*|bayi\s*much[u|o|i]?|bayi\s*muchkond\w*)\b|"
        r"\b(th?ika\s*much[u|o|i]?|th?ika\s*muchkond\w*)\b|"
        # Direct insults to intelligence/sanity
        r"\b(neen[u]?\s*(obba\s*)?(dadd[a|i]|murkh[a|e]|moorkh[a|e]|huchh?a|bewarsi|bevarsi|lofer|loafer))\b|"
        r"\b(dadd[a|i|ara]*\s*(nan\s*maga\w*|thara)|murkh[a|e|ara]*\s*(nan\s*maga\w*|thara)|huchh?a\s*nan\s*maga\w*|bewarsi\s*nan\s*maga\w*|bevarsi\s*nan\s*maga\w*)\b|"
        r"\b(buddi\s*illad\w*|buddhi\s*illad\w*|aklu\s*illa|taleli\s*buddhi\s*illa)\b|"
        # Abusive shaming / personal insults
        r"\b(ninna\s*janmakk[e]?\s*(benki|thoo)|nin\s*janmakk[e]?\s*(benki|thoo)|shame\s*on\s*you)\b|"
        r"\b(manaketta|maana\s*ketta|maanagetta)\b|"
        r"\b(lowde|lavde|laude|lavada|gaandu|shede|shedey|sede)\b"
    )
    if re.search(kn_counter_hate, cs_lower):
        return False, "counter-hate"

    # =========================================================================
    # 3. Threats / Violent Escalation
    # =========================================================================
    en_threats = r"\b(i will kill|we should kill|die|deserve to die|hope you die)\b"
    if re.search(en_threats, cs_lower):
        return False, "threats or violent escalation"

    kn_threats = (
        # Calls to kill / elimination
        r"\b(kollbeku|kollabeku|kond\s*haak\w*|kondu\s*haak\w*)\b|"
        r"\b(saayis\w*|sayis\w*|saais\w*)\b|"
        r"\b(pra?ana\s*(theg[ie]\w*|teg[ie]\w*))\b|"
        r"\b(kole\s*maad\w*|murder\s*maad\w*)\b|"
        # Violent commands to die
        r"\b(hog[i]?\s*saay\w*|neen[u]?\s*saay\w*|saayle|saay[o|i|ri]\s*neen[u]?)\b|"
        r"\b(saayod[u]?\s*vaasi|saayabeku|saaybeku)\b|"
        # Severe physical violence
        r"\b(hoded[u]?\s*haak\w*|kutt[i]?\s*haak\w*|bad[i]?\s*haak\w*)\b|"
        r"\b(kai\s*kaal[u]?\s*muri\w*|kaal\s*kith\w*)\b|"
        r"\b(badith[ie]ni|badibeku|badi\s*beku)\b|"
        r"\b(sutt[u]?\s*haak\w*|benki\s*haak\w*)\b|"
        r"\b(rape\s*maad\w*)\b"
    )
    if re.search(kn_threats, cs_lower):
        return False, "threats or violent escalation"

    # =========================================================================
    # 4. Discriminatory / Harmful content (Slurs & communal/regional hostility)
    # =========================================================================
    slurs = [
        # English slurs
        "nigger", "faggot", "retard", "kike", "chink", "spic",
        # Gender-based Kanglish slurs
        "soole", "sule", "sooley", "suley",
        "soolemaga", "sulemaga", "soole maga", "sule maga",
        "soole magane", "sule magane",
        "bolimaga", "boli maga", "bolimagane", "boli magane",
        "bholimaga", "bholimagane",
        "ranamunde", "ranamundhe", "rana munde", "rana mundhe",
        "soole munde", "soole mundhe",
        "munde maga", "mundhe maga", "munde magane", "mundhe magane",
        "veshye", "veshe",
        # Communal slurs
        "katua", "katuva", "katwa", "katuve", "rice bag", "ricebag",
        # Regional slurs
        "gulti", "gultis", "konga", "kongaru", "bhaiyya kachada"
    ]
    for slur in slurs:
        if slur in cs_lower:
            # Check if it's quoted (single or double quotes):
            if f'"{slur}"' not in cs_lower and f"'{slur}'" not in cs_lower:
                return False, "discriminatory language"

    # Communal collective dehumanization and expulsion regexes
    kn_communal_hostility = (
        r"\b(ella|ellaru)\s+(muslims?|musalman\w*|christians?|hindus?)\s+(terrorists?|ugravaadi\w*|drohi\w*)\b|"
        r"\b(deshadinda|rajyadinda)\s+(odisi|horage\s*haaki|haakbeku|thalli)\b"
    )
    if re.search(kn_communal_hostility, cs_lower):
        return False, "discriminatory language"

    # Regional / Xenophobic expulsion hostile demands
    kn_regional_hostility = (
        r"\b(karnataka|bengaluru|bangalore)\s*(bittu\s*tolagi|bittu\s*odi|dinda\s*odisi|dinda\s*horage\s*haaki)\b|"
        r"\b(illinda\s*tolagi|illinda\s*odisi|illinda\s*horag[e]?\s*haaki)\b"
    )
    if re.search(kn_regional_hostility, cs_lower):
        return False, "discriminatory language"

    return True, ""

def leakage_filter(hate_text: str, test_set_texts: set, val_set_texts: set) -> tuple[bool, str]:
    """
    Check if the hate_text overlaps exactly with HASTIKA test or validation sets.
    Raises ValueError if reference sets are missing, None, or empty to prevent
    unverified reference sets from silently passing leakage audits.
    """
    if not test_set_texts or not val_set_texts:
        raise ValueError(
            "HASTIKA test and validation reference sets must both be non-empty collections to perform leakage filtering."
        )

    ht_norm = hate_text.strip().lower()
    
    if ht_norm in test_set_texts:
        return False, "Leakage: Exact match in HASTIKA Test Set"
    if ht_norm in val_set_texts:
        return False, "Leakage: Exact match in HASTIKA Validation Set"
        
    return True, ""
