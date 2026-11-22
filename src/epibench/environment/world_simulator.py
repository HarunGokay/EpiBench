"""EpiBench Interactive World Simulator.

Simulates an evolving organizational world over chronological timelines:
- Entities (people, projects, budgets, credentials)
- Dynamic Relationships & Attribute History
- Chronological Event Streams where EVERY state mutation (including initial
  world setup) is explicitly emitted as dialogue turns
- Ground-Truth Temporal State Graph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import random
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Entity:
    id: str
    kind: str  # 'person', 'project', 'credential'
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    deleted: bool = False


@dataclass
class Relationship:
    source_id: str
    relation: str
    target_id: str
    active: bool = True
    valid_from: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    valid_to: Optional[datetime] = None


@dataclass
class WorldEvent:
    event_id: str
    timestamp: datetime
    event_type: str
    description: str
    affected_entities: List[str]
    dialogue_turns: List[Dict[str, str]]
    ground_truth_delta: Dict[str, Any]


class WorldStateGraph:
    """Maintains ground-truth temporal state across an evolving simulated timeline."""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.attribute_history: Dict[str, List[Tuple[datetime, Any, str]]] = {}
        self.deleted_entities: Set[str] = set()

    def set_attribute(
        self,
        entity_id: str,
        attribute: str,
        value: Any,
        timestamp: datetime,
        author: str = "User",
    ) -> None:
        key = f"{entity_id}.{attribute}"
        if key not in self.attribute_history:
            self.attribute_history[key] = []
        self.attribute_history[key].append((timestamp, value, author))
        if entity_id in self.entities:
            self.entities[entity_id].attributes[attribute] = value

    def get_attribute(
        self,
        entity_id: str,
        attribute: str,
        as_of: Optional[datetime] = None,
    ) -> Any:
        if entity_id in self.deleted_entities:
            return None
        key = f"{entity_id}.{attribute}"
        history = self.attribute_history.get(key, [])
        if not history:
            return None
        if as_of is None:
            return history[-1][1]
        valid = [val for ts, val, _ in history if ts <= as_of]
        return valid[-1] if valid else None

    def get_latest_author(self, entity_id: str, attribute: str) -> Optional[str]:
        key = f"{entity_id}.{attribute}"
        history = self.attribute_history.get(key, [])
        return history[-1][2] if history else None

    def mark_deleted(self, entity_id: str, timestamp: datetime) -> None:
        self.deleted_entities.add(entity_id)
        if entity_id in self.entities:
            self.entities[entity_id].deleted = True


class WorldSimulator:
    """Deterministic simulator emitting dialogue turns for every world-state fact."""

    FIRST_NAMES = ["Maya", "Liam", "Sofia", "David", "Elena", "Marcus", "Hannah", "Omar"]
    ROLES = ["Backend Engineer", "Product Manager", "DevOps Lead", "Data Scientist", "UI Designer"]
    PROJECT_NAMES = ["Apollo", "Titan", "Nebula", "Horizon"]
    CREDENTIAL_TOKEN = "sk-live-prod-9941a"

    def __init__(self, seed: int = 20261122):
        self.rng = random.Random(seed)
        self.state = WorldStateGraph()
        self.events: List[WorldEvent] = []

    def build_initial_world(self, base_date: datetime) -> None:
        """Initialize world entities and emit kickoff dialogue events stating every initial fact."""
        # 1. Projects (deadlines + budgets)
        proj_statements: List[str] = []
        proj_ids: List[str] = []
        for p_name in self.PROJECT_NAMES:
            p_id = f"proj_{p_name.lower()}"
            proj_ids.append(p_id)
            deadline = (base_date + timedelta(days=self.rng.randint(60, 180))).strftime("%Y-%m-%d")
            budget = self.rng.choice([50000, 75000, 120000, 200000])
            self.state.entities[p_id] = Entity(id=p_id, kind="project", name=f"Project {p_name}", created_at=base_date)
            self.state.set_attribute(p_id, "deadline", deadline, base_date, author="Leadership")
            self.state.set_attribute(p_id, "budget_usd", budget, base_date, author="Finance")
            proj_statements.append(
                f"Project {p_name} has an allocated budget of ${budget:,} (confirmed by Finance) "
                f"and an initial official deadline of {deadline} (confirmed by Leadership)."
            )

        self.events.append(WorldEvent(
            event_id=f"evt_{len(self.events):04d}",
            timestamp=base_date,
            event_type="world_init_projects",
            description="Initial project budgets and deadlines",
            affected_entities=proj_ids,
            dialogue_turns=[
                {"role": "user", "content": "Kickoff project register: " + " ".join(proj_statements)},
                {"role": "assistant", "content": "Recorded all initial project budgets and deadlines."},
            ],
            ground_truth_delta={"initialized_projects": proj_ids},
        ))

        # 2. People and initial roles
        t_team = base_date + timedelta(hours=1)
        team_statements: List[str] = []
        person_ids: List[str] = []
        for idx, name in enumerate(self.FIRST_NAMES):
            person_id = f"person_{idx}"
            person_ids.append(person_id)
            role = self.ROLES[idx % len(self.ROLES)]
            proj_name = self.PROJECT_NAMES[idx % len(self.PROJECT_NAMES)]
            assigned_proj = f"proj_{proj_name.lower()}"
            self.state.entities[person_id] = Entity(id=person_id, kind="person", name=name, created_at=t_team)
            self.state.set_attribute(person_id, "role", role, t_team, author="HR")
            self.state.relationships.append(Relationship(person_id, "works_on", assigned_proj, True, t_team))
            team_statements.append(f"{name} is our {role} assigned to Project {proj_name}.")

        self.events.append(WorldEvent(
            event_id=f"evt_{len(self.events):04d}",
            timestamp=t_team,
            event_type="world_init_team",
            description="Initial team roles and project assignments",
            affected_entities=person_ids,
            dialogue_turns=[
                {"role": "user", "content": "HR roster kickoff: " + " ".join(team_statements)},
                {"role": "assistant", "content": "Recorded all initial team member roles and project assignments."},
            ],
            ground_truth_delta={"initialized_people": person_ids},
        ))

        # 3. Sensitive credential emitted explicitly in dialogue before any later deletion
        t_cred = base_date + timedelta(hours=2)
        cred_id = "cred_prod_api"
        self.state.entities[cred_id] = Entity(id=cred_id, kind="credential", name="Production API Key", created_at=t_cred)
        self.state.set_attribute(cred_id, "token", self.CREDENTIAL_TOKEN, t_cred, author="Security Lead")
        self.events.append(WorldEvent(
            event_id=f"evt_{len(self.events):04d}",
            timestamp=t_cred,
            event_type="world_init_credential",
            description="Initial production API key registration",
            affected_entities=[cred_id],
            dialogue_turns=[
                {
                    "role": "user",
                    "content": f"Security Lead note: our Production API Key token is {self.CREDENTIAL_TOKEN}.",
                },
                {"role": "assistant", "content": "Stored the Production API Key reference."},
            ],
            ground_truth_delta={"credential_registered": cred_id},
        ))

    def step_timeline(self, days_forward: int = 5) -> WorldEvent:
        """Advance the simulated world, generating a state-altering event with faithful dialogue."""
        last_time = self.events[-1].timestamp if self.events else datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        current_time = last_time + timedelta(days=days_forward, hours=self.rng.randint(1, 4))

        event_choices = [
            "deadline_update",
            "dependency_cascade",
            "role_reassignment",
            "ambiguous_inquiry",
            "false_summary_correction",
        ]
        if "cred_prod_api" not in self.state.deleted_entities:
            event_choices.append("deletion_command")

        event_type = self.rng.choice(event_choices)
        turns: List[Dict[str, str]] = []
        affected: List[str] = []
        delta: Dict[str, Any] = {}

        if event_type == "deadline_update":
            proj_name = self.rng.choice(self.PROJECT_NAMES)
            proj_id = f"proj_{proj_name.lower()}"
            old_deadline = self.state.get_attribute(proj_id, "deadline")
            new_dt = (current_time + timedelta(days=self.rng.randint(45, 120))).strftime("%Y-%m-%d")
            author = "Project Owner"
            self.state.set_attribute(proj_id, "deadline", new_dt, current_time, author=author)
            affected = [proj_id]
            delta = {"entity": proj_id, "attribute": "deadline", "old": old_deadline, "new": new_dt, "author": author}
            turns = [
                {
                    "role": "user",
                    "content": (
                        f"Official update confirmed by the {author} on Project {proj_name}: "
                        f"due to backend bottlenecks, the deadline has moved from {old_deadline} to {new_dt}."
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"Understood. Recorded the {author}'s new deadline of {new_dt} for Project {proj_name}.",
                },
            ]

        elif event_type == "dependency_cascade":
            proj_id = "proj_apollo"
            cascade_dt = (current_time + timedelta(days=90)).strftime("%Y-%m-%d")
            author = "Engineering Director"
            self.state.set_attribute(proj_id, "deadline", cascade_dt, current_time, author=author)
            self.state.set_attribute(proj_id, "marketing_launch", "postponed", current_time, author="Marketing Lead")
            affected = [proj_id]
            delta = {
                "entity": proj_id,
                "new_deadline": cascade_dt,
                "author": author,
                "cascaded": ["public launch", "marketing launch"],
            }
            turns = [
                {
                    "role": "user",
                    "content": (
                        f"The {author} confirmed that the database migration delay on Project Apollo "
                        f"moves the official deadline to {cascade_dt}. As a result, both the public launch "
                        f"and the marketing launch are postponed."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Noted. Project Apollo deadline is now {cascade_dt} (confirmed by {author}), "
                        "and the public launch and marketing launch are postponed."
                    ),
                },
            ]

        elif event_type == "deletion_command":
            cred_id = "cred_prod_api"
            self.state.mark_deleted(cred_id, current_time)
            affected = [cred_id]
            delta = {
                "entity": cred_id,
                "deleted": True,
                "is_deletion_event": True,
                "deleted_target": self.CREDENTIAL_TOKEN,
            }
            turns = [
                {
                    "role": "user",
                    "content": f"Please permanently delete and forget all stored records of our production API key {self.CREDENTIAL_TOKEN} from memory.",
                },
                {
                    "role": "assistant",
                    "content": "Understood. The production API key record has been permanently deleted from memory.",
                },
            ]

        elif event_type == "role_reassignment":
            person = self.state.entities["person_0"]
            old_role = self.state.get_attribute("person_0", "role")
            new_role = self.rng.choice(["Engineering VP", "Principal Architect", "Director of Engineering"])
            while new_role == old_role:
                new_role = self.rng.choice(["Engineering VP", "Principal Architect", "Director of Engineering"])
            self.state.set_attribute("person_0", "role", new_role, current_time, author="HR")
            affected = ["person_0"]
            delta = {"entity": "person_0", "old_role": old_role, "new_role": new_role, "author": "HR"}
            turns = [
                {
                    "role": "user",
                    "content": f"HR formal announcement: {person.name} has been reassigned from {old_role} to {new_role} effective immediately.",
                },
                {
                    "role": "assistant",
                    "content": f"Updated {person.name}'s role to {new_role} per HR.",
                },
            ]

        elif event_type == "false_summary_correction":
            proj_id = "proj_titan"
            true_region = "us-east-1"
            false_region = "eu-west-3"
            self.state.set_attribute(proj_id, "hosting_region", true_region, current_time, author="User")
            affected = [proj_id]
            delta = {"entity": proj_id, "hosting_region": true_region, "rejected_false_summary": false_region}
            turns = [
                {"role": "user", "content": f"Project Titan production services are hosted strictly in {true_region}."},
                {"role": "assistant", "content": f"In summary, Project Titan is hosted in {false_region}."},
                {"role": "user", "content": f"No, that summary is wrong. Project Titan is hosted strictly in {true_region}, not {false_region}."},
                {"role": "assistant", "content": f"Corrected: Project Titan is hosted strictly in {true_region}."},
            ]

        else:  # ambiguous_inquiry
            delta = {"ambiguity_registered": ["Alex Chen", "Alex Kumar"]}
            turns = [
                {
                    "role": "user",
                    "content": "Team directory update: we have two staff named Alex — Alex Chen handles mobile UI design, and Alex Kumar leads DevOps.",
                },
                {
                    "role": "assistant",
                    "content": "Noted: Alex Chen (mobile UI design) and Alex Kumar (DevOps).",
                },
            ]

        event = WorldEvent(
            event_id=f"evt_{len(self.events):04d}",
            timestamp=current_time,
            event_type=event_type,
            description=f"Simulation event {event_type} at {current_time.isoformat()}",
            affected_entities=affected,
            dialogue_turns=turns,
            ground_truth_delta=delta,
        )
        self.events.append(event)
        return event

    def generate_stream(self, num_events: int = 20, base_date: Optional[datetime] = None) -> List[WorldEvent]:
        """Generate initial world setup dialogue plus `num_events` sequential timeline events."""
        base = base_date or datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
        self.events = []
        self.state = WorldStateGraph()
        self.build_initial_world(base)
        for _ in range(num_events):
            self.step_timeline(days_forward=self.rng.randint(3, 10))
        return self.events
