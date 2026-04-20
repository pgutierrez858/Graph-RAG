import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from ..llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERSON_NOISE = {
    "he", "she", "they", "his", "her", "their", "it",
    "einstein", "curie", "newton", "bohr",  # bare surnames that slip through
}


def _title_case_name(v: str) -> str:
    """Normalise a person name to 'Firstname Lastname' title-case."""
    return " ".join(w.capitalize() for w in v.strip().split()) if v else v


def _validate_person_name(v: str) -> str:
    """Reject pronouns and bare surnames; normalise to title-case."""
    if v.strip().lower() in _PERSON_NOISE:
        raise ValueError(f"'{v}' is not a valid full person name — use the complete name.")
    return _title_case_name(v)


# ---------------------------------------------------------------------------
# Domain-specific entity models
# ---------------------------------------------------------------------------

class Person(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Full name of the person in 'Firstname Lastname' format. "
            "Always expand pronouns and short references to the full name. "
            "Example: 'Albert Einstein', never 'Einstein', 'he', or 'his'."
        ),
        examples=["Albert Einstein", "Marie Curie", "Isaac Newton"],
    )
    nationality: Optional[str] = Field(
        None,
        description="Nationality or country of origin. Example: 'German', 'French'.",
        examples=["German", "Polish-French"],
    )
    birth_year: Optional[int] = Field(
        None,
        description="Four-digit birth year. Example: 1879.",
        examples=[1879],
    )
    death_year: Optional[int] = Field(
        None,
        description="Four-digit death year if the person is deceased. Example: 1955.",
        examples=[1955],
    )
    fields_of_work: List[str] = Field(
        default_factory=list,
        description=(
            "Academic or professional fields as a list of short strings. "
            "Example: ['theoretical physics', 'quantum mechanics']."
        ),
        examples=[["theoretical physics", "quantum mechanics"]],
    )
    description: str = Field(
        ...,
        description=(
            "One factual sentence in the third person about this person. "
            "Example: 'Albert Einstein was a German-born theoretical physicist.'"
        ),
        examples=["Albert Einstein was a German-born theoretical physicist."],
    )

    @field_validator("name")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        return _validate_person_name(v)


class ScientificTheory(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Official name of the theory or concept, title-cased. "
            "Example: 'Theory of General Relativity', 'Quantum Mechanics'."
        ),
        examples=["Theory of General Relativity", "Special Relativity"],
    )
    year_published: Optional[int] = Field(
        None,
        description="Four-digit year the theory was published or formalised. Example: 1915.",
        examples=[1915],
    )
    description: str = Field(
        ...,
        description=(
            "One factual sentence describing what the theory explains. "
            "Example: 'General relativity describes gravity as the curvature of space-time.'"
        ),
        examples=["General relativity describes gravity as the curvature of space-time."],
    )
    domain: Optional[str] = Field(
        None,
        description="Scientific domain. Example: 'physics', 'cosmology'.",
        examples=["physics"],
    )


class Publication(BaseModel):
    title: str = Field(
        ...,
        description=(
            "Title or widely-used short name of the paper or work, title-cased. "
            "Example: 'On the Electrodynamics of Moving Bodies'."
        ),
        examples=["On the Electrodynamics of Moving Bodies"],
    )
    year: Optional[int] = Field(
        None,
        description="Four-digit publication year. Example: 1905.",
        examples=[1905],
    )
    topic: Optional[str] = Field(
        None,
        description="Main subject in a few words. Example: 'special relativity'.",
        examples=["special relativity"],
    )
    description: str = Field(
        ...,
        description=(
            "One factual sentence about what the publication covers. "
            "Example: 'This paper introduced the special theory of relativity.'"
        ),
        examples=["This paper introduced the special theory of relativity."],
    )


class Institution(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Official name of the organisation, title-cased. "
            "Example: 'Swiss Patent Office', 'Institute for Advanced Study'."
        ),
        examples=["Swiss Patent Office", "Institute for Advanced Study"],
    )
    type: str = Field(
        ...,
        description=(
            "Category of the institution. "
            "One of: university, research_institute, patent_office, government_office, laboratory."
        ),
        examples=["patent_office", "research_institute"],
    )
    location: Optional[str] = Field(
        None,
        description="City or country. Example: 'Bern, Switzerland'.",
        examples=["Bern, Switzerland"],
    )
    description: str = Field(
        ...,
        description=(
            "One factual sentence about the institution. "
            "Example: 'The Swiss Patent Office is a government agency located in Bern.'"
        ),
        examples=["The Swiss Patent Office is a government agency located in Bern."],
    )


class Location(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Place name, title-cased. Prefer the most specific form available. "
            "Example: 'Ulm', 'Princeton, New Jersey', 'Germany'."
        ),
        examples=["Ulm", "Princeton, New Jersey", "Germany"],
    )
    type: str = Field(
        ...,
        description="One of: city, country, region, state.",
        examples=["city", "country"],
    )
    country: Optional[str] = Field(
        None,
        description="Country the location belongs to. Example: 'Germany', 'United States'.",
        examples=["Germany"],
    )
    description: str = Field(
        ...,
        description=(
            "One factual sentence providing geographic context. "
            "Example: 'Ulm is a city in Baden-Württemberg, Germany.'"
        ),
        examples=["Ulm is a city in Baden-Württemberg, Germany."],
    )


class Award(BaseModel):
    name: str = Field(
        ...,
        description=(
            "Official name of the award, title-cased. "
            "Do NOT include the year in the name — the year goes in the 'year' field. "
            "Example: 'Nobel Prize in Physics', never 'Nobel Prize in Physics 1921'."
        ),
        examples=["Nobel Prize in Physics"],
    )
    year: Optional[int] = Field(
        None,
        description="Four-digit year the award was granted. Example: 1921.",
        examples=[1921],
    )
    field: Optional[str] = Field(
        None,
        description="Discipline or category. Example: 'physics'.",
        examples=["physics"],
    )
    reason: Optional[str] = Field(
        None,
        description=(
            "Short reason or citation. "
            "Example: 'explanation of the photoelectric effect'."
        ),
        examples=["explanation of the photoelectric effect"],
    )
    description: str = Field(
        ...,
        description=(
            "One factual sentence about the award. "
            "Example: 'The Nobel Prize in Physics was awarded to Albert Einstein in 1921 "
            "for his discovery of the law of the photoelectric effect.'"
        ),
        examples=["The Nobel Prize in Physics was awarded to Albert Einstein in 1921 "
                  "for his discovery of the law of the photoelectric effect."],
    )

    @field_validator("name")
    @classmethod
    def strip_year_suffix(cls, v: str) -> str:
        # Remove trailing year that the model sometimes appends: "Nobel Prize in Physics 1921"
        return re.sub(r"\s+\d{4}$", "", v.strip())


# ---------------------------------------------------------------------------
# Domain-specific relationship models
# ---------------------------------------------------------------------------

def _person_ref_validator(v: str) -> str:
    return _validate_person_name(v)


class BornIn(BaseModel):
    person: str = Field(
        ...,
        description="Full name of the person. Example: 'Albert Einstein'.",
        examples=["Albert Einstein"],
    )
    location: str = Field(
        ...,
        description="Name of the birth location. Example: 'Ulm'.",
        examples=["Ulm"],
    )
    year: Optional[int] = Field(None, description="Birth year. Example: 1879.", examples=[1879])

    @field_validator("person")
    @classmethod
    def validate_person(cls, v: str) -> str:
        return _person_ref_validator(v)


class WorkedAt(BaseModel):
    person: str = Field(..., description="Full name. Example: 'Albert Einstein'.", examples=["Albert Einstein"])
    institution: str = Field(..., description="Institution name. Example: 'Swiss Patent Office'.", examples=["Swiss Patent Office"])
    from_year: Optional[int] = Field(None, description="Start year. Example: 1902.", examples=[1902])
    to_year: Optional[int] = Field(None, description="End year. Example: 1909.", examples=[1909])
    role: Optional[str] = Field(None, description="Role held. Example: 'patent clerk'.", examples=["patent clerk"])

    @field_validator("person")
    @classmethod
    def validate_person(cls, v: str) -> str:
        return _person_ref_validator(v)


class LivedIn(BaseModel):
    person: str = Field(..., description="Full name. Example: 'Albert Einstein'.", examples=["Albert Einstein"])
    location: str = Field(..., description="Location name. Example: 'Princeton, New Jersey'.", examples=["Princeton, New Jersey"])
    from_year: Optional[int] = Field(None, description="Year moved there.", examples=[1933])
    to_year: Optional[int] = Field(None, description="Year left, if known.", examples=[1955])

    @field_validator("person")
    @classmethod
    def validate_person(cls, v: str) -> str:
        return _person_ref_validator(v)


class Developed(BaseModel):
    person: str = Field(..., description="Full name. Example: 'Albert Einstein'.", examples=["Albert Einstein"])
    theory: str = Field(..., description="Theory name. Example: 'Theory of General Relativity'.", examples=["Theory of General Relativity"])
    year: Optional[int] = Field(None, description="Year of development. Example: 1915.", examples=[1915])

    @field_validator("person")
    @classmethod
    def validate_person(cls, v: str) -> str:
        return _person_ref_validator(v)


class Published(BaseModel):
    person: str = Field(..., description="Full name of the author. Example: 'Albert Einstein'.", examples=["Albert Einstein"])
    publication: str = Field(..., description="Publication title. Example: 'On the Electrodynamics of Moving Bodies'.", examples=["On the Electrodynamics of Moving Bodies"])
    year: Optional[int] = Field(None, description="Publication year. Example: 1905.", examples=[1905])
    while_at: Optional[str] = Field(None, description="Institution at time of publication. Example: 'Swiss Patent Office'.", examples=["Swiss Patent Office"])

    @field_validator("person")
    @classmethod
    def validate_person(cls, v: str) -> str:
        return _person_ref_validator(v)


class ReceivedAward(BaseModel):
    person: str = Field(..., description="Full name. Example: 'Albert Einstein'.", examples=["Albert Einstein"])
    award: str = Field(
        ...,
        description=(
            "Award name without year. Example: 'Nobel Prize in Physics'. "
            "Never include the year in the name."
        ),
        examples=["Nobel Prize in Physics"],
    )
    year: Optional[int] = Field(None, description="Year received. Example: 1921.", examples=[1921])
    reason: Optional[str] = Field(None, description="Reason for the award.", examples=["explanation of the photoelectric effect"])

    @field_validator("person")
    @classmethod
    def validate_person(cls, v: str) -> str:
        return _person_ref_validator(v)

    @field_validator("award")
    @classmethod
    def strip_year_from_award(cls, v: str) -> str:
        return re.sub(r"\s+\d{4}$", "", v.strip())


class ContributedTo(BaseModel):
    person: str = Field(..., description="Full name. Example: 'Albert Einstein'.", examples=["Albert Einstein"])
    field: str = Field(..., description="Scientific field. Example: 'quantum mechanics'.", examples=["quantum mechanics"])
    description: str = Field(
        ...,
        description=(
            "One factual sentence about the contribution. "
            "Example: 'Albert Einstein made foundational contributions to quantum mechanics.'"
        ),
        examples=["Albert Einstein made foundational contributions to quantum mechanics."],
    )

    @field_validator("person")
    @classmethod
    def validate_person(cls, v: str) -> str:
        return _person_ref_validator(v)


# ---------------------------------------------------------------------------
# Two-pass extraction schemas
# ---------------------------------------------------------------------------

def _filter_list(items: list, model_cls) -> list:
    """Return only items that pass model_cls validation; silently drop the rest."""
    valid = []
    for item in items:
        try:
            model_cls.model_validate(item if isinstance(item, dict) else item.model_dump())
            valid.append(item)
        except Exception as exc:
            logger.debug("Dropped invalid %s item: %s — %s", model_cls.__name__, item, exc)
    return valid


class _EntityPass(BaseModel):
    """Pass 1 — entities only (no relationships).  Smaller schema = better names."""
    persons: List[Person] = Field(default_factory=list)
    theories: List[ScientificTheory] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
    institutions: List[Institution] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    awards: List[Award] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def drop_invalid_entities(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        for field, model_cls in [
            ("persons",      Person),
            ("theories",     ScientificTheory),
            ("publications", Publication),
            ("institutions", Institution),
            ("locations",    Location),
            ("awards",       Award),
        ]:
            if isinstance(data.get(field), list):
                data[field] = _filter_list(data[field], model_cls)
        return data


class _RelationshipPass(BaseModel):
    """Pass 2 — relationships only, grounded to the entity names from Pass 1."""
    born_in: List[BornIn] = Field(default_factory=list)
    worked_at: List[WorkedAt] = Field(default_factory=list)
    lived_in: List[LivedIn] = Field(default_factory=list)
    developed: List[Developed] = Field(default_factory=list)
    published: List[Published] = Field(default_factory=list)
    received_awards: List[ReceivedAward] = Field(default_factory=list)
    contributed_to: List[ContributedTo] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def drop_invalid_relationships(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        for field, model_cls in [
            ("born_in",        BornIn),
            ("worked_at",      WorkedAt),
            ("lived_in",       LivedIn),
            ("developed",      Developed),
            ("published",      Published),
            ("received_awards", ReceivedAward),
            ("contributed_to", ContributedTo),
        ]:
            if isinstance(data.get(field), list):
                data[field] = _filter_list(data[field], model_cls)
        return data


# ---------------------------------------------------------------------------
# Top-level extraction container (public API — unchanged for TextProcessor)
# ---------------------------------------------------------------------------

class PhysicistGraphExtraction(BaseModel):
    persons: List[Person] = Field(default_factory=list)
    theories: List[ScientificTheory] = Field(default_factory=list)
    publications: List[Publication] = Field(default_factory=list)
    institutions: List[Institution] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    awards: List[Award] = Field(default_factory=list)

    # Relationships
    born_in: List[BornIn] = Field(default_factory=list)
    worked_at: List[WorkedAt] = Field(default_factory=list)
    lived_in: List[LivedIn] = Field(default_factory=list)
    developed: List[Developed] = Field(default_factory=list)
    published: List[Published] = Field(default_factory=list)
    received_awards: List[ReceivedAward] = Field(default_factory=list)
    contributed_to: List[ContributedTo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Structured response model for consolidation — forces a single sentence
# ---------------------------------------------------------------------------

class OneSentenceSummary(BaseModel):
    summary: str = Field(
        ...,
        description=(
            "Exactly one factual sentence in the third person. "
            "No preamble, no reasoning, no alternatives. "
            "Example: 'Albert Einstein was a German-born theoretical physicist.'"
        ),
        examples=["Albert Einstein was a German-born theoretical physicist."],
    )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    def __init__(self):
        self.client = OllamaClient()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract_entities_and_relationships(
            self,
            text: str,
            known_persons: Optional[List[str]] = None,
    ) -> PhysicistGraphExtraction:
        """
        Two-pass extraction:
          Pass 1 — entities only (smaller schema → cleaner names).
          Pass 2 — relationships only, grounded to the entity names from Pass 1
                   (prevents hallucinated or fragment-copied entity names).
        Returns the merged PhysicistGraphExtraction so TextProcessor is unchanged.
        """
        logger.info("Pass 1 — entity extraction (%d chars)", len(text))
        entities = self._extract_entities(text, known_persons=known_persons)
        logger.info(
            "Pass 1 ✓  persons=%d  theories=%d  publications=%d  "
            "institutions=%d  locations=%d  awards=%d",
            len(entities.persons), len(entities.theories),
            len(entities.publications), len(entities.institutions),
            len(entities.locations), len(entities.awards),
        )

        # Merge known_persons into the entity list for pass-2 relationship resolution
        # (handles chunks that refer to a person only by pronoun or bare surname)
        all_persons: List[str] = [p.name for p in entities.persons]
        if known_persons:
            for kp in known_persons:
                if kp not in all_persons:
                    all_persons.append(kp)

        logger.info("Pass 2 — relationship extraction (%d known entities, %d persons total)",
                    sum([len(entities.persons), len(entities.theories),
                         len(entities.publications), len(entities.institutions),
                         len(entities.locations), len(entities.awards)]),
                    len(all_persons))
        relationships = self._extract_relationships(text, entities, all_persons)
        logger.info(
            "Pass 2 ✓  born_in=%d  worked_at=%d  lived_in=%d  developed=%d  "
            "published=%d  received_awards=%d  contributed_to=%d",
            len(relationships.born_in), len(relationships.worked_at),
            len(relationships.lived_in), len(relationships.developed),
            len(relationships.published), len(relationships.received_awards),
            len(relationships.contributed_to),
        )

        return PhysicistGraphExtraction(
            **entities.model_dump(),
            **relationships.model_dump(),
        )

    # ------------------------------------------------------------------
    # Private: Pass 1 — entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(
            self, text: str, known_persons: Optional[List[str]] = None
    ) -> _EntityPass:
        system_prompt = (
            "You are a named entity extractor for scientific biographical text.\n\n"
            "Extract EVERY entity of these types that appears in the text:\n"
            "- persons       : full 'Firstname Lastname' format. "
            "Never use pronouns (he/his/she) or bare surnames alone.\n"
            "- theories      : named scientific theories or concepts "
            "(e.g. 'Theory of General Relativity', 'Special Relativity', 'Quantum Mechanics'). "
            "Include theories mentioned even briefly.\n"
            "- publications  : only when a specific paper title is given. "
            "Skip vague references like 'several papers' or 'groundbreaking work'.\n"
            "- institutions  : universities, research institutes, offices, laboratories.\n"
            "- locations     : cities, countries, regions.\n"
            "- awards        : official award name WITHOUT the year "
            "(e.g. 'Nobel Prize in Physics'). The year goes in the 'year' field.\n\n"
            "Be thorough — extract every entity you can identify, even if mentioned briefly."
        )
        context_hint = ""
        if known_persons:
            context_hint = (
                f"\nCONTEXT: The following persons are known from earlier parts of the document: "
                f"{known_persons}. Use their full names if the text refers to them by pronoun or surname."
            )
        return self.client.structured_output(
            prompt=f"Extract all named entities from this text:{context_hint}\n\n{text}",
            schema=_EntityPass,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Private: Pass 2 — relationship extraction (grounded)
    # ------------------------------------------------------------------

    def _extract_relationships(
            self,
            text: str,
            entities: _EntityPass,
            all_persons: Optional[List[str]] = None,
    ) -> _RelationshipPass:
        person_names    = all_persons or [p.name for p in entities.persons]
        theory_names    = [t.name for t in entities.theories]
        pub_titles      = [p.title for p in entities.publications]
        inst_names      = [i.name for i in entities.institutions]
        loc_names       = [l.name for l in entities.locations]
        award_names     = [a.name for a in entities.awards]

        entity_list = (
            f"Persons     : {person_names or '(none)'}\n"
            f"Theories    : {theory_names or '(none)'}\n"
            f"Publications: {pub_titles or '(none)'}\n"
            f"Institutions: {inst_names or '(none)'}\n"
            f"Locations   : {loc_names or '(none)'}\n"
            f"Awards      : {award_names or '(none)'}"
        )

        system_prompt = (
            "You are extracting typed relationships from scientific biographical text.\n\n"
            "ENTITY NAMES IDENTIFIED IN THIS TEXT:\n"
            f"{entity_list}\n\n"
            "Prefer the exact names above when the text refers to those entities. "
            "You may also use other entity names that appear in the text if they are "
            "involved in a relationship not covered by the list above.\n\n"
            "Relationship types to extract:\n"
            "- born_in        : person → location (birth place, optional year)\n"
            "- worked_at      : person → institution (dates, role if stated). "
            "Include any affiliation, employment, membership, or position, "
            "even phrased as 'joined', 'was appointed at', 'became a member of'.\n"
            "- lived_in       : person → location (dates if stated)\n"
            "- developed      : person → theory (year if stated)\n"
            "- published      : person → publication title (year, institution at time)\n"
            "- received_awards: person → award name WITHOUT year (year, reason if stated)\n"
            "- contributed_to : person → scientific field as free text\n\n"
            "CRITICAL: Always use the full person name ('Firstname Lastname'). "
            "Never use pronouns (he/she/his) or bare surnames. "
            "If the text only uses a pronoun or surname for a person, "
            "resolve it to the full name from the entity list above.\n\n"
            "Extract every relationship supported by the text, even if phrased indirectly."
        )

        return self.client.structured_output(
            prompt=f"Extract all relationships present in this text:\n\n{text}",
            schema=_RelationshipPass,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Consolidation helpers (Entity Resolution)
    # ------------------------------------------------------------------

    def summarize_entity(self, entity_name: str, descriptions: List[str]) -> str:
        """Merges multiple descriptions of the same entity into one summary sentence."""
        description_list = "\n- ".join(dict.fromkeys(descriptions))  # deduplicate, keep order
        prompt = (
            f"Write a single factual sentence in the third person summarising "
            f"'{entity_name}' using only the information below:\n- {description_list}"
        )
        result: OneSentenceSummary = self.client.structured_output(
            prompt=prompt,
            schema=OneSentenceSummary,
        )
        return result.summary

    def summarize_relationship(self, source: str, target: str, descriptions: List[str]) -> str:
        """Merges multiple descriptions of the same relationship into one summary sentence."""
        description_list = "\n- ".join(dict.fromkeys(descriptions))  # deduplicate, keep order
        prompt = (
            f"Write a single factual sentence describing the relationship between "
            f"'{source}' and '{target}' using only the information below:\n- {description_list}"
        )
        result: OneSentenceSummary = self.client.structured_output(
            prompt=prompt,
            schema=OneSentenceSummary,
        )
        return result.summary