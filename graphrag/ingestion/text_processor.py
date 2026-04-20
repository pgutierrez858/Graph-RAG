import uuid
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from ..graph.neo4j_manager import Neo4jManager
from ..utils.chunking import chunk_text
from ..utils.embeddings import EmbeddingGenerator
from .entity_extractor import EntityExtractor, PhysicistGraphExtraction

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(
            self,
            neo4j_manager: Neo4jManager,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
    ):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
        self.entity_extractor = EntityExtractor()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_document(
            self,
            text: str,
            document_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Processes a document: splits it into chunks, extracts domain-specific
        entities and typed relationships, and stores everything in Neo4j.
        """
        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        self._create_document_node(document_id, metadata)

        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        logger.info("Document '%s' → %d chunks", document_id, len(chunks))

        # Accumulate person names across chunks so later chunks can resolve
        # pronouns/surnames to the correct full name.
        known_persons: List[str] = []

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            logger.info("Chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            chunk_id = f"{document_id}_chunk_{i}"
            new_persons = self._process_chunk(chunk_id, chunk, document_id, i, known_persons)
            for p in new_persons:
                if p not in known_persons:
                    known_persons.append(p)

        # Deduplication must happen before consolidation so summaries are
        # written to the already-merged canonical node.
        self._deduplicate_entities()
        self._consolidate_entities()
        self._consolidate_relationships()

        logger.info("Document '%s' processed successfully.", document_id)

    # ------------------------------------------------------------------
    # Document & chunk nodes
    # ------------------------------------------------------------------

    def _create_document_node(self, document_id: str, metadata: Dict[str, Any]):
        query = """
        MERGE (d:Document {id: $document_id})
        SET d += $metadata
        """
        self.neo4j.execute_query(query, {"document_id": document_id, "metadata": metadata})

    def _process_chunk(
            self,
            chunk_id: str,
            text: str,
            document_id: str,
            index: int,
            known_persons: Optional[List[str]] = None,
    ) -> List[str]:
        """Processes one chunk and returns the person names extracted from it."""
        embedding = self.embedding_gen.embed_text(text)

        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.embedding = $embedding,
            c.index = $index
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "index": index,
            "document_id": document_id,
        })

        extraction: PhysicistGraphExtraction = (
            self.entity_extractor.extract_entities_and_relationships(
                text, known_persons=known_persons
            )
        )

        self._store_all_entities(extraction, chunk_id)
        self._store_all_relationships(extraction, chunk_id)
        return [p.name for p in extraction.persons]

    # ------------------------------------------------------------------
    # Entity storage — one method per node label
    # ------------------------------------------------------------------

    def _store_all_entities(self, extraction: PhysicistGraphExtraction, chunk_id: str):
        for p in extraction.persons:
            self._store_person(p.model_dump(), chunk_id)
        for t in extraction.theories:
            self._store_theory(t.model_dump(), chunk_id)
        for pub in extraction.publications:
            self._store_publication(pub.model_dump(), chunk_id)
        for inst in extraction.institutions:
            self._store_institution(inst.model_dump(), chunk_id)
        for loc in extraction.locations:
            self._store_location(loc.model_dump(), chunk_id)
        for award in extraction.awards:
            self._store_award(award.model_dump(), chunk_id)

    def _store_person(self, data: Dict[str, Any], chunk_id: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (p:Person {name: $name})
        ON CREATE SET
            p.nationality    = $nationality,
            p.birth_year     = $birth_year,
            p.death_year     = $death_year,
            p.fields_of_work = $fields_of_work
        SET p.description = CASE
            WHEN p.description IS NULL THEN [$description]
            ELSE p.description + [$description]
        END
        MERGE (c)-[:HAS_ENTITY]->(p)
        """
        self.neo4j.execute_query(query, {
            "chunk_id":      chunk_id,
            "name":          data.get("name"),
            "nationality":   data.get("nationality"),
            "birth_year":    data.get("birth_year"),
            "death_year":    data.get("death_year"),
            "fields_of_work": data.get("fields_of_work", []),
            "description":   data.get("description", ""),
        })

    def _store_theory(self, data: Dict[str, Any], chunk_id: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (t:ScientificTheory {name: $name})
        ON CREATE SET
            t.year_published = $year_published,
            t.domain         = $domain
        SET t.description = CASE
            WHEN t.description IS NULL THEN [$description]
            ELSE t.description + [$description]
        END
        MERGE (c)-[:HAS_ENTITY]->(t)
        """
        self.neo4j.execute_query(query, {
            "chunk_id":      chunk_id,
            "name":          data.get("name"),
            "year_published": data.get("year_published"),
            "domain":        data.get("domain"),
            "description":   data.get("description", ""),
        })

    def _store_publication(self, data: Dict[str, Any], chunk_id: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (pub:Publication {title: $title})
        ON CREATE SET
            pub.year  = $year,
            pub.topic = $topic
        SET pub.description = CASE
            WHEN pub.description IS NULL THEN [$description]
            ELSE pub.description + [$description]
        END
        MERGE (c)-[:HAS_ENTITY]->(pub)
        """
        self.neo4j.execute_query(query, {
            "chunk_id":    chunk_id,
            "title":       data.get("title"),
            "year":        data.get("year"),
            "topic":       data.get("topic"),
            "description": data.get("description", ""),
        })

    def _store_institution(self, data: Dict[str, Any], chunk_id: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (i:Institution {name: $name})
        ON CREATE SET
            i.type     = $type,
            i.location = $location
        SET i.description = CASE
            WHEN i.description IS NULL THEN [$description]
            ELSE i.description + [$description]
        END
        MERGE (c)-[:HAS_ENTITY]->(i)
        """
        self.neo4j.execute_query(query, {
            "chunk_id":    chunk_id,
            "name":        data.get("name"),
            "type":        data.get("type", "unknown"),
            "location":    data.get("location"),
            "description": data.get("description", ""),
        })

    def _store_location(self, data: Dict[str, Any], chunk_id: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (l:Location {name: $name})
        ON CREATE SET
            l.type    = $type,
            l.country = $country
        SET l.description = CASE
            WHEN l.description IS NULL THEN [$description]
            ELSE l.description + [$description]
        END
        MERGE (c)-[:HAS_ENTITY]->(l)
        """
        self.neo4j.execute_query(query, {
            "chunk_id":    chunk_id,
            "name":        data.get("name"),
            "type":        data.get("type", "unknown"),
            "country":     data.get("country"),
            "description": data.get("description", ""),
        })

    def _store_award(self, data: Dict[str, Any], chunk_id: str):
        query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (a:Award {name: $name})
        ON CREATE SET
            a.year   = $year,
            a.field  = $field,
            a.reason = $reason
        SET a.description = CASE
            WHEN a.description IS NULL THEN [$description]
            ELSE a.description + [$description]
        END
        MERGE (c)-[:HAS_ENTITY]->(a)
        """
        self.neo4j.execute_query(query, {
            "chunk_id":    chunk_id,
            "name":        data.get("name"),
            "year":        data.get("year"),
            "field":       data.get("field"),
            "reason":      data.get("reason"),
            "description": data.get("description", ""),
        })

    # ------------------------------------------------------------------
    # Relationship storage — one method per relationship type
    # ------------------------------------------------------------------

    def _store_all_relationships(self, extraction: PhysicistGraphExtraction, chunk_id: str):
        for rel in extraction.born_in:
            self._store_born_in(rel.model_dump(), chunk_id)
        for rel in extraction.worked_at:
            self._store_worked_at(rel.model_dump(), chunk_id)
        for rel in extraction.lived_in:
            self._store_lived_in(rel.model_dump(), chunk_id)
        for rel in extraction.developed:
            self._store_developed(rel.model_dump(), chunk_id)
        for rel in extraction.published:
            self._store_published(rel.model_dump(), chunk_id)
        for rel in extraction.received_awards:
            self._store_received_award(rel.model_dump(), chunk_id)
        for rel in extraction.contributed_to:
            self._store_contributed_to(rel.model_dump(), chunk_id)

    def _store_born_in(self, data: Dict[str, Any], chunk_id: str):
        if not data.get("person") or not data.get("location"):
            logger.warning("Skipping incomplete BORN_IN in chunk %s", chunk_id)
            return
        query = """
        MERGE (p:Person   {name: $person})
        MERGE (l:Location {name: $location})
        MERGE (p)-[r:BORN_IN]->(l)
        SET r.year = $year
        """
        self.neo4j.execute_query(query, {
            "person":   data["person"],
            "location": data["location"],
            "year":     data.get("year"),
        })

    def _store_worked_at(self, data: Dict[str, Any], chunk_id: str):
        if not data.get("person") or not data.get("institution"):
            logger.warning("Skipping incomplete WORKED_AT in chunk %s", chunk_id)
            return
        query = """
        MERGE (p:Person      {name: $person})
        MERGE (i:Institution {name: $institution})
        MERGE (p)-[r:WORKED_AT]->(i)
        SET r.from_year = $from_year,
            r.to_year   = $to_year,
            r.role      = $role
        """
        self.neo4j.execute_query(query, {
            "person":      data["person"],
            "institution": data["institution"],
            "from_year":   data.get("from_year"),
            "to_year":     data.get("to_year"),
            "role":        data.get("role"),
        })

    def _store_lived_in(self, data: Dict[str, Any], chunk_id: str):
        if not data.get("person") or not data.get("location"):
            logger.warning("Skipping incomplete LIVED_IN in chunk %s", chunk_id)
            return
        query = """
        MERGE (p:Person   {name: $person})
        MERGE (l:Location {name: $location})
        MERGE (p)-[r:LIVED_IN]->(l)
        SET r.from_year = $from_year,
            r.to_year   = $to_year
        """
        self.neo4j.execute_query(query, {
            "person":    data["person"],
            "location":  data["location"],
            "from_year": data.get("from_year"),
            "to_year":   data.get("to_year"),
        })

    def _store_developed(self, data: Dict[str, Any], chunk_id: str):
        if not data.get("person") or not data.get("theory"):
            logger.warning("Skipping incomplete DEVELOPED in chunk %s", chunk_id)
            return
        query = """
        MERGE (p:Person         {name: $person})
        MERGE (t:ScientificTheory {name: $theory})
        MERGE (p)-[r:DEVELOPED]->(t)
        SET r.year = $year
        """
        self.neo4j.execute_query(query, {
            "person": data["person"],
            "theory": data["theory"],
            "year":   data.get("year"),
        })

    def _store_published(self, data: Dict[str, Any], chunk_id: str):
        if not data.get("person") or not data.get("publication"):
            logger.warning("Skipping incomplete PUBLISHED in chunk %s", chunk_id)
            return
        query = """
        MERGE (p:Person      {name: $person})
        MERGE (pub:Publication {title: $publication})
        MERGE (p)-[r:PUBLISHED]->(pub)
        SET r.year     = $year,
            r.while_at = $while_at
        """
        self.neo4j.execute_query(query, {
            "person":      data["person"],
            "publication": data["publication"],
            "year":        data.get("year"),
            "while_at":    data.get("while_at"),
        })

    def _store_received_award(self, data: Dict[str, Any], chunk_id: str):
        if not data.get("person") or not data.get("award"):
            logger.warning("Skipping incomplete RECEIVED_AWARD in chunk %s", chunk_id)
            return
        query = """
        MERGE (p:Person {name: $person})
        MERGE (a:Award  {name: $award})
        MERGE (p)-[r:RECEIVED_AWARD]->(a)
        SET r.year   = $year,
            r.reason = $reason
        """
        self.neo4j.execute_query(query, {
            "person": data["person"],
            "award":  data["award"],
            "year":   data.get("year"),
            "reason": data.get("reason"),
        })

    def _store_contributed_to(self, data: Dict[str, Any], chunk_id: str):
        if not data.get("person") or not data.get("field"):
            logger.warning("Skipping incomplete CONTRIBUTED_TO in chunk %s", chunk_id)
            return
        query = """
        MERGE (p:Person {name: $person})
        MERGE (f:ScientificField {name: $field})
        MERGE (p)-[r:CONTRIBUTED_TO]->(f)
        SET r.description = $description
        """
        self.neo4j.execute_query(query, {
            "person":      data["person"],
            "field":       data["field"],
            "description": data.get("description", ""),
        })

    # ------------------------------------------------------------------
    # Entity deduplication (must run before consolidation)
    # ------------------------------------------------------------------

    def _deduplicate_entities(self):
        """
        Merges duplicate entity nodes created during multi-chunk ingestion.

        Pass 1 — case-insensitive exact match for all name-keyed labels.
        Pass 2 — Person surname-suffix match ("Einstein" → "Albert Einstein").
        Pass 3 — case-insensitive match for Publication (keyed by title).
        """
        logger.info("Starting entity deduplication...")
        for label in ["Person", "ScientificTheory", "Institution", "Location", "Award", "ScientificField"]:
            for canonical, dup in self._find_case_duplicates(label, "name"):
                logger.info("Dedup %s: '%s' ← '%s'", label, canonical, dup)
                self._merge_nodes(label, "name", canonical, dup)

        for full, short in self._find_surname_duplicates():
            logger.info("Dedup Person surname: '%s' ← '%s'", full, short)
            self._merge_nodes("Person", "name", full, short)

        for canonical, dup in self._find_case_duplicates("Publication", "title"):
            logger.info("Dedup Publication: '%s' ← '%s'", canonical, dup)
            self._merge_nodes("Publication", "title", canonical, dup)

    def _find_case_duplicates(self, label: str, key: str) -> List[tuple]:
        """Returns (canonical, duplicate) pairs for nodes sharing the same case-insensitive key."""
        query = f"""
        MATCH (a:{label}), (b:{label})
        WHERE id(a) < id(b) AND toLower(a.{key}) = toLower(b.{key})
        RETURN
          CASE WHEN size(a.{key}) >= size(b.{key}) THEN a.{key} ELSE b.{key} END AS canonical,
          CASE WHEN size(a.{key}) >= size(b.{key}) THEN b.{key} ELSE a.{key} END AS duplicate
        """
        return [(r["canonical"], r["duplicate"]) for r in self.neo4j.execute_query(query)]

    def _find_surname_duplicates(self) -> List[tuple]:
        """Returns (full_name, short_name) where full ENDS WITH ' ' + short (e.g. surname-only nodes)."""
        query = """
        MATCH (full:Person), (short:Person)
        WHERE id(full) <> id(short)
          AND full.name ENDS WITH (' ' + short.name)
        RETURN full.name AS full_name, short.name AS short_name
        """
        return [(r["full_name"], r["short_name"]) for r in self.neo4j.execute_query(query)]

    def _merge_nodes(self, label: str, key: str, canonical_val: str, dup_val: str):
        """Merges dup node into canonical: retargets all edges, copies missing props, deletes dup."""
        params = {"canonical": canonical_val, "dup": dup_val}

        # Retarget Chunk → entity HAS_ENTITY edges
        self.neo4j.execute_query(f"""
        MATCH (c:Chunk)-[:HAS_ENTITY]->(d:{label} {{{key}: $dup}})
        MATCH (n:{label} {{{key}: $canonical}})
        MERGE (c)-[:HAS_ENTITY]->(n)
        """, params)

        # Retarget outgoing domain relationships (only Person nodes have them)
        if label == "Person":
            for rel_type, tgt_label, tgt_key in [
                ("BORN_IN",        "Location",         "name"),
                ("WORKED_AT",      "Institution",       "name"),
                ("LIVED_IN",       "Location",          "name"),
                ("DEVELOPED",      "ScientificTheory",  "name"),
                ("PUBLISHED",      "Publication",       "title"),
                ("RECEIVED_AWARD", "Award",             "name"),
                ("CONTRIBUTED_TO", "ScientificField",   "name"),
            ]:
                self.neo4j.execute_query(f"""
                MATCH (d:Person {{name: $dup}})-[r:{rel_type}]->(t:{tgt_label})
                MATCH (n:Person {{name: $canonical}})
                MERGE (n)-[r2:{rel_type}]->(t)
                  ON CREATE SET r2 += properties(r)
                DELETE r
                """, params)

        # Copy missing scalar properties and merge description lists
        self._copy_node_properties(label, key, canonical_val, dup_val)

        # Remove the now-orphaned duplicate
        self.neo4j.execute_query(f"""
        MATCH (d:{label} {{{key}: $dup}})
        DETACH DELETE d
        """, params)

    def _copy_node_properties(self, label: str, key: str, canonical_val: str, dup_val: str):
        """Copies non-null properties from dup to canonical (canonical's existing values take precedence)."""
        params = {"canonical": canonical_val, "dup": dup_val}
        desc_merge = (
            "n.description = coalesce(n.description, []) + "
            "[x IN coalesce(d.description, []) WHERE NOT x IN coalesce(n.description, [])]"
        )

        prop_sets: Dict[str, str] = {
            "Person": (
                "n.nationality    = COALESCE(n.nationality, d.nationality), "
                "n.birth_year     = COALESCE(n.birth_year, d.birth_year), "
                "n.death_year     = COALESCE(n.death_year, d.death_year), "
                "n.fields_of_work = COALESCE(n.fields_of_work, d.fields_of_work), "
            ),
            "ScientificTheory": (
                "n.year_published = COALESCE(n.year_published, d.year_published), "
                "n.domain         = COALESCE(n.domain, d.domain), "
            ),
            "Institution": (
                "n.type     = COALESCE(n.type, d.type), "
                "n.location = COALESCE(n.location, d.location), "
            ),
            "Location": (
                "n.type    = COALESCE(n.type, d.type), "
                "n.country = COALESCE(n.country, d.country), "
            ),
            "Award": (
                "n.year   = COALESCE(n.year, d.year), "
                "n.field  = COALESCE(n.field, d.field), "
                "n.reason = COALESCE(n.reason, d.reason), "
            ),
            "Publication": (
                "n.year  = COALESCE(n.year, d.year), "
                "n.topic = COALESCE(n.topic, d.topic), "
            ),
        }

        extra_props = prop_sets.get(label, "")
        self.neo4j.execute_query(f"""
        MATCH (n:{label} {{{key}: $canonical}}), (d:{label} {{{key}: $dup}})
        SET {extra_props}{desc_merge}
        """, params)

    # ------------------------------------------------------------------
    # Entity & relationship consolidation (Entity Resolution)
    # ------------------------------------------------------------------

    def _consolidate_entities(self):
        """
        For every node label that accumulates descriptions across chunks,
        merge them into a single summary property.
        """
        labels = [
            "Person", "ScientificTheory", "Publication",
            "Institution", "Location", "Award",
        ]
        for label in labels:
            query = f"""
            MATCH (e:{label})
            WHERE size(e.description) > 1
            RETURN e.name AS name, e.description AS descriptions
            """
            # Publications use 'title' as their primary key
            if label == "Publication":
                query = """
                MATCH (e:Publication)
                WHERE size(e.description) > 1
                RETURN e.title AS name, e.description AS descriptions
                """

            nodes = self.neo4j.execute_query(query)
            for node in tqdm(nodes, desc=f"Consolidating {label}"):
                summary = self.entity_extractor.summarize_entity(
                    node["name"], node["descriptions"]
                )
                if label == "Publication":
                    update_query = """
                    MATCH (e:Publication {title: $name})
                    SET e.summary = $summary
                    """
                else:
                    update_query = f"""
                    MATCH (e:{label} {{name: $name}})
                    SET e.summary = $summary
                    """
                self.neo4j.execute_query(update_query, {
                    "name": node["name"],
                    "summary": summary,
                })

    def _consolidate_relationships(self):
        """
        For relationship types that carry accumulated description lists,
        merge those descriptions and store an avg_strength where applicable.
        """
        # Only DEVELOPED and PUBLISHED accumulate enough text to be worth merging;
        # the others are keyed on unique date properties so no list accumulates.
        rel_specs = [
            ("Person", "DEVELOPED",      "ScientificTheory", "name",  "name"),
            ("Person", "PUBLISHED",       "Publication",      "name",  "title"),
            ("Person", "CONTRIBUTED_TO",  "ScientificField",  "name",  "name"),
        ]
        for src_label, rel_type, tgt_label, src_key, tgt_key in rel_specs:
            query = f"""
            MATCH (s:{src_label})-[r:{rel_type}]->(t:{tgt_label})
            WHERE r.description IS NOT NULL AND size(r.description) > 1
            RETURN s.{src_key} AS source, t.{tgt_key} AS target,
                   r.description AS descriptions
            """
            rels = self.neo4j.execute_query(query)
            for rel in tqdm(rels, desc=f"Consolidating {rel_type}"):
                summary = self.entity_extractor.summarize_relationship(
                    rel["source"], rel["target"], rel["descriptions"]
                )
                update_query = f"""
                MATCH (s:{src_label} {{{src_key}: $source}})
                      -[r:{rel_type}]->
                      (t:{tgt_label} {{{tgt_key}: $target}})
                SET r.summary = $summary
                """
                self.neo4j.execute_query(update_query, {
                    "source":  rel["source"],
                    "target":  rel["target"],
                    "summary": summary,
                })