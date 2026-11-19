"""
CSHSM Milestone G2: Centralized Kanglish Prompt Template.

Provides standardized, reusable prompt generation for counter-speech models
across Cloudflare Workers AI, Groq, and all configured generation adapters.

Guarantees:
- Natural Romanized Kannada (Kanglish) language requirements.
- Approximately 70–80% Kannada vocabulary and 20–30% English vocabulary.
- Strict Latin script enforcement (no Kannada Unicode script).
- Empathetic, respectful, de-escalating, non-aggressive tone.
- Prohibition of attacks, insults, threats, shaming, or hate reinforcement.
- Category-specific context for all 6 canonical CSHSM categories:
    Gender, Political, Religion, Geo-political, Violence, Others.
- 2–4 sentences output constraint without meta-commentary or quotation marks.
"""

from typing import Dict, List, Optional, Any
import re


CANONICAL_CATEGORIES: tuple[str, ...] = (
    "Gender",
    "Political",
    "Religion",
    "Geo-political",
    "Violence",
    "Others"
)

CATEGORY_ALIASES: Dict[str, str] = {
    "gender": "Gender",
    "women": "Gender",
    "men": "Gender",
    "lgbt": "Gender",
    "lgbtq": "Gender",
    "political": "Political",
    "politics": "Political",
    "party": "Political",
    "religion": "Religion",
    "religious": "Religion",
    "communal": "Religion",
    "geo-political": "Geo-political",
    "geopolitical": "Geo-political",
    "regional": "Geo-political",
    "migrants": "Geo-political",
    "violence": "Violence",
    "violent": "Violence",
    "threats": "Violence",
    "others": "Others",
    "other": "Others",
    "general": "Others"
}

CATEGORY_CONTEXT: Dict[str, str] = {
    "Gender": (
        "Focus on gender equality and mutual respect. Challenge gender stereotypes and "
        "misogynistic or discriminatory generalizations. Reiterate that leadership, dignity, "
        "and capability belong equally to all individuals regardless of gender."
    ),
    "Political": (
        "Focus on democratic values, constructive dialogue, and constitutional due process. "
        "Challenge broad accusations against political groups while encouraging civil disagreement "
        "and accountability through legal and institutional channels."
    ),
    "Religion": (
        "Focus on communal harmony, freedom of belief, and mutual respect among communities. "
        "Reject collective blame, broad stereotypes, and demonization of any religious group."
    ),
    "Geo-political": (
        "Focus on regional unity, cultural diversity, and peaceful coexistence. Challenge xenophobic "
        "or divisive rhetoric targeting migrants, linguistic communities, or neighboring regions, "
        "highlighting shared contributions and mutual growth."
    ),
    "Violence": (
        "Focus on de-escalation, empathy, and non-violence. Firmly reject any incitement to violence, "
        "threats, or retaliation, advocating for peaceful resolution and respect for the rule of law."
    ),
    "Others": (
        "Focus on universal human dignity, empathy, and factual understanding. Challenge prejudice "
        "and harmful generalizations calmly without attacking the commenter."
    )
}

BASE_SYSTEM_INSTRUCTION: str = (
    "You are an expert counter-speech generator for online social media conversations in Karnataka.\n"
    "Your objective is to produce effective, de-escalating counter-speech responding to hateful comments.\n\n"
    "Language & Script Requirements:\n"
    "- Write in natural conversational Romanized Kannada (Kanglish) using only the English/Latin alphabet.\n"
    "- Never use Kannada script (Unicode Kannada characters).\n"
    "- Use approximately 70–80% conversational Kannada vocabulary and 20–30% commonly used English words (e.g., 'respect', 'community', 'equal', 'education', 'future').\n"
    "- Use natural, spoken Kannada phrasing commonly used in Karnataka social media discussions. Avoid robotic or literal translations.\n\n"
    "Tone & Safety Constraints:\n"
    "- Maintain a calm, empathetic, and respectful tone that de-escalates hostility.\n"
    "- Challenge harmful stereotypes and generalizations factually without attacking, insulting, shaming, or threatening the original commenter.\n"
    "- Never reinforce the hateful premise and never agree with hate speech.\n"
    "- Generate only 2 to 4 concise, meaningful sentences.\n"
    "- Output exclusively the final counter-speech response. Do not include analysis, preambles, meta-commentary, or quotation marks."
)


def normalize_category(category: Optional[str]) -> str:
    """
    Normalizes input category string to one of the 6 canonical CSHSM categories.
    Defaults to 'Others' for empty or unrecognized categories.
    """
    if not category or not isinstance(category, str):
        return "Others"

    clean_cat = category.strip().lower()
    return CATEGORY_ALIASES.get(clean_cat, "Others")


def get_category_guidance(category: Optional[str]) -> str:
    """
    Returns specific contextual counter-speech guidance for a given category.
    """
    norm_cat = normalize_category(category)
    return CATEGORY_CONTEXT.get(norm_cat, CATEGORY_CONTEXT["Others"])


def build_system_prompt(category: Optional[str] = None) -> str:
    """
    Builds the standardized system prompt with Kanglish language rules and category context.
    """
    norm_cat = normalize_category(category)
    guidance = get_category_guidance(norm_cat)

    return (
        f"{BASE_SYSTEM_INSTRUCTION}\n\n"
        f"Topic Category: {norm_cat}\n"
        f"Category Guidance: {guidance}"
    )


def build_user_prompt(hateful_comment: str, category: Optional[str] = None) -> str:
    """
    Builds the user message instructing the model to generate counter-speech for the input comment.
    """
    norm_cat = normalize_category(category)
    guidance = get_category_guidance(norm_cat)
    cleaned_comment = hateful_comment.strip() if hateful_comment else ""

    return (
        f"Category: {norm_cat}\n"
        f"Context Guidance: {guidance}\n"
        f"Hateful comment: \"{cleaned_comment}\"\n\n"
        f"Generate a natural Romanized Kannada-English (Kanglish) counter-speech response in 2-4 sentences:"
    )


def build_chat_messages(
    hateful_comment: str,
    category: Optional[str] = None,
    system_instruction_override: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Constructs OpenAI/Cloudflare-compatible chat messages list:
    [
        {"role": "system", "content": ...},
        {"role": "user", "content": ...}
    ]
    """
    system_content = system_instruction_override or build_system_prompt(category)
    user_content = build_user_prompt(hateful_comment=hateful_comment, category=category)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


class KanglishPromptBuilder:
    """
    Reusable prompt builder class for CSHSM counter-speech generation models.
    """

    def __init__(
        self,
        base_system_instruction: str = BASE_SYSTEM_INSTRUCTION,
        category_contexts: Optional[Dict[str, str]] = None
    ):
        self.base_system_instruction = base_system_instruction
        self.category_contexts = category_contexts or CATEGORY_CONTEXT

    def get_system_prompt(self, category: Optional[str] = None) -> str:
        norm_cat = normalize_category(category)
        guidance = self.category_contexts.get(norm_cat, self.category_contexts["Others"])
        return (
            f"{self.base_system_instruction}\n\n"
            f"Topic Category: {norm_cat}\n"
            f"Category Guidance: {guidance}"
        )

    def get_user_prompt(self, hateful_comment: str, category: Optional[str] = None) -> str:
        norm_cat = normalize_category(category)
        guidance = self.category_contexts.get(norm_cat, self.category_contexts["Others"])
        cleaned = hateful_comment.strip() if hateful_comment else ""
        return (
            f"Category: {norm_cat}\n"
            f"Context Guidance: {guidance}\n"
            f"Hateful comment: \"{cleaned}\"\n\n"
            f"Generate a natural Romanized Kannada-English (Kanglish) counter-speech response in 2-4 sentences:"
        )

    def build_messages(
        self,
        prompt: str,
        category: Optional[str] = None
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.get_system_prompt(category)},
            {"role": "user", "content": self.get_user_prompt(prompt, category)}
        ]
