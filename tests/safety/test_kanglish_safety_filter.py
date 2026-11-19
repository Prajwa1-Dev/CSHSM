"""
CSHSM Milestone G5: Unit Tests for Code-Mixed Kannada (Kanglish) Safety Filtering.

Tests deterministic upstream and defense-in-depth safety filtering for:
  - English and Romanized Kannada (Kanglish) expressions
  - Hate reinforcement (with and without reframing)
  - Counter-hate (attacking, insulting, or shaming the commenter)
  - Threats and violent escalation (killing, death commands, bodily harm)
  - Discriminatory language (gender-based, communal, and regional slurs)
  - Quotation tolerance (educational / counter-narrative mentions)
  - Non-false-positive preservation for respectful disagreement and sensitive discussions
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.kb.filters import safety_filter


# =====================================================================
# 1. Empty & Whitespace Validation
# =====================================================================

def test_empty_and_whitespace_rejection():
    safe, reason = safety_filter("")
    assert safe is False
    assert reason == "empty content"

    safe, reason = safety_filter("   \n\t  ")
    assert safe is False
    assert reason == "empty content"


# =====================================================================
# 2. Hate Reinforcement (Kanglish & English)
# =====================================================================

def test_kanglish_hate_reinforcement_without_reframing():
    unsafe_examples = [
        "Correct aagi helidri, avaru haage kettavaru.",
        "Naanu agree maadtini, avara bagge neevu heliddu nija.",
        "Sariyaagi helidri, aa community janaru haage.",
        "Idu sathyavada mathu, avaru sariyalla.",
        "Nimm mathu nija, avaru eppatigu badalagalla."
    ]
    for text in unsafe_examples:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "hate reinforcement"


def test_kanglish_hate_reinforcement_with_contrastive_reframing_is_safe():
    safe_examples = [
        "Correct aagi helidri, aadre broad stereotypes use maaduvudhu thappu.",
        "Nimm mathu nija, but naavu ellarannu dooshisbaardu.",
        "Idu nija aadare prathiyobba vyakthiyannu avara kelsa nodi judge maadbeku.",
        "Naan agree maadtini, however generalise maaduvudhu sariyalla.",
        "Sari helidri, aadru shantiyinda vishaya charchisbeku."
    ]
    for text in safe_examples:
        safe, reason = safety_filter(text)
        assert safe is True, f"Expected safe for reframed text: '{text}', got reason: '{reason}'"
        assert reason == ""


# =====================================================================
# 3. Counter-Hate & Direct Personal Attacks (Kanglish)
# =====================================================================

def test_kanglish_counter_hate_shut_up_commands():
    shut_up_examples = [
        "Baayi muchkond hogu illinda.",
        "Baay muchu, ninage arivu illa.",
        "Bayi muchkond iru, bereyavara bagge maathadbedi.",
        "Thika muchu, ninage enu gothilla.",
        "Tika muchkond hogu, ninna opinion yaargu beda."
    ]
    for text in shut_up_examples:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "counter-hate"


def test_kanglish_counter_hate_intelligence_and_sanity_insults():
    insult_examples = [
        "Neenu obba dadda, enu thiliyade post maadbedi.",
        "Murkha nan maga neenu, shantiyinda iru.",
        "Daddi thara maathadbedi, factual aagi nodi.",
        "Taleli buddhi illa ninage, waste fellow.",
        "Buddi illada manushya neenu.",
        "Huchha nan magane, baayi muchu.",
        "Neenu obba bewarsi, ninna janmakke thoo."
    ]
    for text in insult_examples:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "counter-hate"


def test_kanglish_counter_hate_shaming_and_vulgarity():
    vulgar_examples = [
        "Ninna janmakke benki haaka, thumba kettavanu.",
        "Manaketta vyakti neenu, shameful behavior.",
        "Lowde ninage basic respect gothilla.",
        "Gaandu thara behave maadbedi."
    ]
    for text in vulgar_examples:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "counter-hate"


# =====================================================================
# 4. Threats and Violent Escalation (Kanglish)
# =====================================================================

def test_kanglish_killing_threats_and_calls_for_elimination():
    killing_threats = [
        "Ninnanu saayisthini husharagi iru.",
        "Sayistini ninna road nalli.",
        "Avarannu kollbeku, kondu haakbeku.",
        "Ninna praana thegithini nodo.",
        "Prana tegitheeni ninage dhairya idre baa.",
        "Avaranna murder maadbeku, saayisbeku."
    ]
    for text in killing_threats:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "threats or violent escalation"


def test_kanglish_violent_death_commands():
    death_commands = [
        "Hogi saayri ellaru, illi bandu keda bedi.",
        "Neenu saayle, ninna thara janarinda desha haalaagide.",
        "Hog saayo neenu, yaargu beda.",
        "Ninna thara irorige saayodu vaasi."
    ]
    for text in death_commands:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "threats or violent escalation"


def test_kanglish_physical_assault_threats():
    assault_threats = [
        "Ninna kai kaalu murithini road nalli sigadre.",
        "Hodedu haakthini ninnanu illi bandre.",
        "Ninnanu kutti haakbeku.",
        "Ninna kaal kithakthini.",
        "Avara manege benki haakbeku.",
        "Rape maadbeku antha bedarike haakodu thappu."  # Note: rape threats
    ]
    for text in assault_threats:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "threats or violent escalation"


# =====================================================================
# 5. Discriminatory Language across Categories (Kanglish)
# =====================================================================

def test_kanglish_gender_slurs():
    gender_slurs = [
        "Avalu obba soole, avala maathu kelabedi.",
        "Boli magane ninna maathu nillsi.",
        "Ranamunde thara aadbedi.",
        "Munde maga neenu, avalu obba veshye.",
        "Soole maga thara maathadbedi."
    ]
    for text in gender_slurs:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "discriminatory language"


def test_kanglish_communal_slurs_and_hostility():
    communal_hate = [
        "Katuva galige illi sthaana illa.",
        "Katwa galannu deshadinda odisi.",
        "Ella musalmanaru terrorists, avarannu rajyadinda odisi.",
        "Rice bag convert galu deshadrohigalu."
    ]
    for text in communal_hate:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "discriminatory language"


def test_kanglish_regional_and_xenophobic_hostility():
    regional_hate = [
        "Gultis ellaru Karnataka bittu tolagi.",
        "Kongaru Bangalore dinda odisi bega.",
        "Illinda tolagi, bere rajyadinda bandu kachada maadbedi.",
        "Karnataka dinda horage haaki ella outsiders annu."
    ]
    for text in regional_hate:
        safe, reason = safety_filter(text)
        assert safe is False, f"Expected unsafe for: '{text}'"
        assert reason == "discriminatory language"


# =====================================================================
# 6. Quotation Tolerance (Educational & Rebuttal Mentions)
# =====================================================================

def test_quoted_slurs_are_tolerated():
    quoted_examples = [
        "Do not address anyone as 'soole', treat every person with dignity.",
        'Using derogatory words like "katua" destroys communal harmony.',
        "Calling someone 'boli maga' is completely unacceptable in civil dialogue.",
        "We should never use slurs like 'gulti' against our fellow citizens.",
        "Avarannu “soole” athava “katua” antha kariyodu thappu."
    ]
    for text in quoted_examples:
        safe, reason = safety_filter(text)
        assert safe is True, f"Expected safe for quoted text: '{text}', got: '{reason}'"
        assert reason == ""


# =====================================================================
# 7. Non-False-Positive Preservation (Safe Kanglish Counter-Speech)
# =====================================================================

def test_safe_kanglish_counter_speech_preservation():
    canonical_safe = [
        # Canonical KB V2 sample: contains 'mundide'
        "Driving skill gender mele depend aagalla. Proper practice, road awareness mattu traffic rules paalisidre yaaru aadru safe driver aagbahudu. Bengaluru traffic nalli saaviraaru mahileyaru two-wheeler, cab mattu BMTC bus galannu safely operate maadtiro real examples namma mundide.",
        # Education and gender respect
        "Vidyaabhyasa prathiyobba vyaktiya moola hakku. Higher education padedha mahileyaru financial independence thogothare mattu kutumbakke balavada support aagtaare. Maduve aadro kooda tamma career continue maaduvudhu avara aaykhe aagiratte, adannu naavu respect maadbeku.",
        # Gender equality: contains 'gandu' and 'hennu' in non-abusive context
        "Gandu mattu hennu ibbarigu samaana hakku mattu avakasha irabeku. Gender mele bhedabhava maaduvudu samvidhana virodhi.",
        # Professional leadership
        "Ondhu leadership position ge talupalu hard work, competence mattu performance mukhya. Avara talent nodade kevala avalu hennu antha baseless aropane maaduvudu professional culture ge sariyalla.",
        # Respectful disagreement with hateful ideas
        "Neevu heliddu thappu, yaava samudayakku ee thara aropane maadbaardu. Naavu ellarannu sammanadinda nodbeku.",
        # Inter-state harmony & Karnataka development
        "Karnataka ellarannu preetiyinda swagatisutte. Bere rajyadinda bandavaru kooda namma state development ge contribute maadtaare.",
        # Communal friendship & peace
        "Musalmanaru, Hindugalu mattu Christian-ru ellaru snehadinda baalabeku. Manushyatva ellakkinta doddadu.",
        # De-escalation & anti-hate plea
        "Shantiyinda maathadona, dvesha hanchuvudhu yaava samasyege parihara alla."
    ]
    for text in canonical_safe:
        safe, reason = safety_filter(text)
        assert safe is True, f"False positive detected on legitimate counter-speech: '{text}' (reason: '{reason}')"
        assert reason == ""
