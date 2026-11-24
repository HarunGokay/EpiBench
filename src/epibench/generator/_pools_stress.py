"""Curated epistemic-stress pools and overflow-chain builders for EpiBench v2.2.0 (--profile stress).

Designed specifically to expose the structural limits of flat chunk-based RAG
without requiring expensive LLM judging:
- Track R (stress): 28 scattered timestamped revisions (~2,600+ tokens of relevant
  evidence for one entity inside a 1,500-token budget) + 240 mixed distractors.
- Track F (stress): 24 scattered dated schedule commitments/slips/checkpoints
  (~2,200+ tokens of relevant timeline evidence) with exact computed slip counts/days.
- Track D (stress): 12 curated 3-hop transitive dependency cascades across 5 sessions
  where the terminal affected decisions share zero lexical overlap with the root trigger.
- Track X (stress): 12 curated cross-domain safety/legal/physical constraints verified
  to have EXACTLY 0 content-word overlap with the neutral user action query.
- Track I (stress): 12 curated 16-session longitudinal portfolio experiments (~1,900+
  tokens of relevant evidence) testing cross-entity confounder filtering and aggregation.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 1. Track R Stress: 28-step Evidence-Exceeds-Budget Revision Chains
# ---------------------------------------------------------------------------

STRESS_REVISION_COUNT = 28

_R_STRESS_EXTRA_VALUES: Dict[str, List[str]] = {
    "r_00_prod_db": [
        "MariaDB 10.6", "PostgreSQL 12", "MySQL 8.0", "Percona Server 8.0", "PostgreSQL 13",
        "Aurora MySQL 3", "MariaDB 10.11", "PostgreSQL 13.8", "AlloyDB Preview", "CitrixDB",
        "TiDB 6.5", "YugabyteDB 2.16", "SingleStore 8.0", "MySQL 8.1", "Aurora Postgres 14",
        "PlanetScale Vitess", "Neon Serverless 14", "Citus 11", "YugabyteDB 2.18", "TiDB 7.1",
        "Spanner PGAdapter", "PostgreSQL 15.1", "Aurora Postgres 15", "Neon Serverless 15",
        "Citus 12",
    ],
    "r_01_gym_membership": [
        "24 Hour Fitness", "Gold's Gym", "YMCA Downtown", "LA Fitness", "Anytime Fitness",
        "Blink Fitness", "Planet Fitness", "Retro Fitness", "Snap Fitness", "Crunch Signature",
        "Athletic Club East", "Bay Club", "Chelsea Piers Fitness", "F45 Studio", "Orangetheory",
        "Barry's Bootcamp", "SoulCycle Studio", "CorePower Loft", "Midtown Athletic", "Harbor Fitness",
        "Summit Climbing Gym", "Brooklyn Boulders", "Ironworks Barbell", "Metroflex Club", "Form50 Studio",
    ],
    "r_02_savings_target": [
        "$650", "$700", "$725", "$750", "$775", "$800", "$825", "$850", "$875", "$900",
        "$925", "$950", "$975", "$1,050", "$1,100", "$1,150", "$1,250", "$1,300", "$1,350", "$1,400",
        "$1,450", "$1,550", "$1,600", "$1,650", "$1,700",
    ],
    "r_03_frontend_framework": [
        "Backbone.js", "Ember.js", "AngularJS 1.8", "Knockout.js", "jQuery UI",
        "Polymer 3", "Preact X", "Mithril.js", "LitElement", "Alpine.js",
        "InfernoJS", "Aurelia 2", "StencilJS", "Marko 5", "SolidJS 1.4",
        "Qwik 0.19", "Astro 2", "Remix 1.15", "Next.js 12", "Nuxt 2",
        "Angular 15", "SolidStart", "Qwik City", "Remix 2", "Next.js 14",
    ],
    "r_04_airline_loyalty": [
        "JetBlue Airways", "Alaska Airlines", "Southwest Airlines", "Spirit Airlines", "Frontier Airlines",
        "Hawaiian Airlines", "Air Canada", "British Airways", "Lufthansa", "Air France",
        "KLM Royal Dutch", "Swiss International", "Iberia Airlines", "Aer Lingus", "SAS Scandinavian",
        "Finnair", "TAP Air Portugal", "Turkish Airlines", "Emirates", "Qatar Airways",
        "Singapore Airlines", "ANA All Nippon", "Japan Airlines", "Qantas Airways", "Virgin Atlantic",
    ],
    "r_05_countertop_material": [
        "laminate Formica", "ceramic tile", "bamboo butcher block", "maple butcher block", "walnut butcher block",
        "cultured marble", "Corian solid surface", "honed slate", "polished limestone", "travertine slab",
        "terrazzo composite", "recycled glass slab", "poured concrete", "stainless steel sheet", "zinc sheet",
        "basalt stone", "dolomite slab", "porcelain Dekton", "sintered Neolith", "silestone suede",
        "honed travertine", "vermont slate", "leathered granite", "honed quartzite", "calacatta porcelain",
    ],
    "r_06_project_deadline": [
        "August 5", "August 12", "August 19", "August 26", "September 2",
        "September 9", "September 16", "September 23", "September 30", "October 7",
        "October 14", "October 21", "October 28", "November 4", "November 11",
        "November 18", "November 25", "December 1", "December 5", "December 8",
        "December 10", "December 12", "December 20", "December 28", "January 5",
    ],
    "r_07_ci_platform": [
        "Hudson CI", "CruiseControl", "TeamCity 2021", "Bamboo Server", "Travis CI",
        "Semaphore 2", "Codeship Pro", "Buddy Works", "AppVeyor", "Drone CI",
        "Concourse CI", "GoCD", "Spinnaker", "Woodpecker CI", "Tekton Pipelines",
        "Argo Workflows", "Bitrise", "Codemagic", "Azure Pipelines", "AWS CodeBuild",
        "Google Cloud Build", "Harness CI", "Earthly Cloud", "Dagger Engine", "Buildbot",
    ],
    "r_08_brokerage_account": [
        "E*TRADE", "TD Ameritrade", "Merrill Edge", "Ally Invest", "SoFi Invest",
        "Robinhood", "Webull", "Public.com", "M1 Finance", "Betterment",
        "Wealthfront", "Acorns", "Stash Invest", "TradeStation", "Tastytrade",
        "Firstrade", "J.P. Morgan Self-Directed", "WellsTrade", "Edward Jones", "Raymond James",
        "TIAA Brokerage", "USAA Investments", "Zacks Trade", "Lightspeed Trading", "Cobra Trading",
    ],
    "r_09_office_city": [
        "Portland", "Minneapolis", "Columbus", "Indianapolis", "Nashville",
        "Charlotte", "Raleigh", "Tampa", "Phoenix", "Salt Lake City",
        "Kansas City", "Pittsburgh", "Cleveland", "Detroit", "Baltimore",
        "Philadelphia", "San Diego", "Sacramento", "San Antonio", "Dallas",
        "Houston", "Atlanta", "Miami", "Boston", "Chicago",
    ],
    "r_10_messaging_queue": [
        "Beanstalkd", "Gearman", "ZeroMQ", "NSQ", "ActiveMQ Classic",
        "ActiveMQ Artemis", "IBM MQ", "HornetQ", "IronMQ", "Celery Redis",
        "BullMQ", "Sidekiq Queue", "Resque", "Azure Service Bus", "Azure Event Hubs",
        "Google Pub/Sub", "Redpanda 22", "NATS Core", "NATS JetStream", "Apache Pulsar 2.10",
        "Apache Pulsar 3.0", "Redpanda 23", "EMQX Broker", "Mosquitto MQTT", "Solace PubSub+",
    ],
    "r_11_heating_fuel": [
        "heating oil #2", "kerosene tank", "cordwood stove", "wood pellet furnace", "anthracite coal",
        "baseboard electric", "hydronic oil boiler", "propane radiant", "ducted electric coil", "infrared quartz",
        "solar thermal loop", "biomass chip boiler", "dual-fuel propane", "mini-split zone 1", "condensing propane",
        "electric thermal storage", "air-to-water chiller", "ground-loop vertical", "closed-loop pond", "open-loop well",
        "hybrid oil-electric", "cast-iron steam boiler", "radiant floor hydronic", "variable-speed furnace", "ductless inverter",
    ],
}

_R_RATIONALE_CLAUSES: List[str] = [
    "after reviewing quarterly operating metrics, vendor reliability benchmarks, and total recurring ownership overhead with stakeholders",
    "following a thorough architecture and cost audit that flagged bottlenecks during peak weekday utilization windows",
    "because our recent contract renewal introduced unexpected surcharges and degraded support response times",
    "after running a two-week parallel pilot comparing maintenance burden, failure recovery speed, and day-to-day ergonomics",
    "to align our operational baseline with updated quarterly capacity targets and simplify routine onboarding workflows",
    "in response to recurring friction logged across three consecutive retrospectives and quarterly budget reconciliations",
]


def build_stress_revision_chain(pool_item: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a curated Track R template into a 28-step overflow revision chain (~2,600+ tokens).

    Preserves the template's initial value (values[0]) as revision #1 (so as_of on the
    initial window matches values[0]), inserts 25 intermediate historical revisions in the
    middle, and places values[-2] at revision #27 (previous) and values[-1] at revision #28
    (current). Total switches = 27. Every value in the 28-step chain is pairwise distinct.
    """
    base_vals: List[str] = list(pool_item["values"])
    base_aliases: Dict[str, List[str]] = {k: list(v) for k, v in pool_item["value_aliases"].items()}
    tid = pool_item["id"]
    extras = list(_R_STRESS_EXTRA_VALUES.get(tid, []))

    # Exclude any extra that collides case-insensitively with base values
    used_lower = {v.lower() for v in base_vals}
    clean_extras: List[str] = []
    for ex in extras:
        if ex.lower() not in used_lower:
            clean_extras.append(ex)
            used_lower.add(ex.lower())

    needed_middle = STRESS_REVISION_COUNT - len(base_vals)
    # Pad deterministically if a custom template isn't in _R_STRESS_EXTRA_VALUES
    pad_idx = 1
    while len(clean_extras) < needed_middle:
        cand = f"{pool_item['entity'].title()} Option {pad_idx}"
        if cand.lower() not in used_lower:
            clean_extras.append(cand)
            used_lower.add(cand.lower())
        pad_idx += 1

    # Chain order: [initial_val] + middle_revisions + [remaining base vals ending with prev, current]
    chain_vals: List[str] = [base_vals[0]] + clean_extras[:needed_middle] + base_vals[1:]
    assert len(chain_vals) == STRESS_REVISION_COUNT
    assert len({v.lower() for v in chain_vals}) == STRESS_REVISION_COUNT

    chain_aliases: Dict[str, List[str]] = {}
    for v in chain_vals:
        chain_aliases[v] = list(base_aliases.get(v, [v]))

    entity_phrase = pool_item["entity_phrase"]
    phrasings: List[str] = []
    for idx, val in enumerate(chain_vals):
        rationale = _R_RATIONALE_CLAUSES[idx % len(_R_RATIONALE_CLAUSES)]
        cycle_label = f"Revision log #{idx + 1} for {entity_phrase}"
        if idx == 0:
            base_p = pool_item["user_phrasings"][0].format(val=val)
            text = (
                f"{cycle_label}: {base_p} We documented this baseline choice {rationale}, "
                f"and recorded exact configuration notes so future transitions stay auditable."
            )
        elif idx == len(chain_vals) - 1:
            base_p = pool_item["user_phrasings"][-1].format(val=val)
            text = (
                f"{cycle_label}: {base_p} This supersedes {chain_vals[idx - 1]} permanently {rationale}, "
                f"and all active runbooks now point strictly to {val}."
            )
        elif idx == len(chain_vals) - 2:
            base_p = pool_item["user_phrasings"][-2].format(val=val)
            text = (
                f"{cycle_label}: {base_p} We retired {chain_vals[idx - 1]} in favor of {val} {rationale}, "
                f"completing the cutover checklist across all affected workflows."
            )
        else:
            text = (
                f"{cycle_label}: We formally changed {entity_phrase} from {chain_vals[idx - 1]} to {val} "
                f"{rationale}. Please update the active operational ledger to reflect {val} as our selection."
            )
        phrasings.append(text)

    return {
        "values": chain_vals,
        "value_aliases": chain_aliases,
        "user_phrasings": phrasings,
    }


# ---------------------------------------------------------------------------
# 2. Track F Stress: 24-step Overflow Forecast & Slippage Timelines
# ---------------------------------------------------------------------------

STRESS_FORECAST_EVENT_COUNT = 24  # 1 commitment + 22 updates (slips or on-time checkpoints) + 1 completion


def build_stress_forecast_events(pool_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a Track F template into 24 dated schedule sessions (~2,300+ tokens).

    - For slipping projects (`is_realistic == False`): 1 commitment + 22 dated slips
      (each shifting the target forward by 5–9 days) + 1 final completion.
    - For on-time projects (`is_realistic == True`): 1 commitment + 22 dated on-time
      milestone checkpoints (0 slips, target unchanged) + 1 on-time completion.
    All gold counts and calendar slip days are derived strictly by compute_forecast_gold().
    """
    project = pool_item["project"]
    is_realistic = bool(pool_item["is_realistic"])
    start_dt = date(2023, 1, 10)
    initial_target = date(2023, 4, 15) if is_realistic else date(2023, 2, 20)

    events: List[Dict[str, Any]] = [
        {
            "date": start_dt.isoformat(),
            "event_kind": "commitment",
            "target_date": initial_target.isoformat(),
            "text": (
                f"Schedule baseline for {project}: Starting execution today ({start_dt.isoformat()}) with a "
                f"committed target completion date of {initial_target.isoformat()}. We logged resource allocations, "
                f"dependency gates, and weekly milestone verification rules in the delivery ledger."
            ),
        }
    ]

    if is_realistic:
        # 22 on-time checkpoints over ~90 days (every 4 days) -> 0 slips, 0 slip days
        for i in range(1, 23):
            chk_dt = start_dt + timedelta(days=4 * i)
            events.append({
                "date": chk_dt.isoformat(),
                "event_kind": "checkpoint",
                "target_date": initial_target.isoformat(),
                "text": (
                    f"Checkpoint #{i} for {project} on {chk_dt.isoformat()}: Workstream remains strictly on schedule "
                    f"with zero slippage. Our original committed completion target of {initial_target.isoformat()} "
                    f"stands unchanged after validating QA coverage, staffing availability, and upstream dependencies."
                ),
            })
        events.append({
            "date": initial_target.isoformat(),
            "event_kind": "completion",
            "target_date": initial_target.isoformat(),
            "text": (
                f"Final delivery sign-off for {project}: Completed right on schedule on {initial_target.isoformat()} "
                f"with zero date moves and zero slip days against the original commitment."
            ),
        })
    else:
        # 22 slips spanning ~15 months: every ~14 days, target slips by 7..10 days
        cur_target = initial_target
        cur_log_dt = start_dt + timedelta(days=12)
        slip_causes = [
            "integration regression in staging verification",
            "upstream vendor schema revision requiring rework",
            "unplanned compliance review cycle",
            "capacity contention during peak incident triage",
            "data backfill latency exceeding the cutover window",
            "third-party certification queue delay",
        ]
        for i in range(1, 23):
            step = 7 + (i % 4)  # 8, 9, 10, 7 deterministic pattern
            prev_target = cur_target
            cur_target = cur_target + timedelta(days=step)
            cur_log_dt = max(cur_log_dt + timedelta(days=14), prev_target - timedelta(days=2))
            if cur_log_dt >= cur_target:
                cur_log_dt = cur_target - timedelta(days=1)
            cause = slip_causes[(i - 1) % len(slip_causes)]
            events.append({
                "date": cur_log_dt.isoformat(),
                "event_kind": "slip",
                "target_date": cur_target.isoformat(),
                "text": (
                    f"Schedule revision #{i} for {project}: We have to push the target completion date back "
                    f"from {prev_target.isoformat()} to {cur_target.isoformat()} (+{step} days) due to {cause}. "
                    f"Stakeholders were notified of the revised milestone window."
                ),
            })
        events.append({
            "date": cur_target.isoformat(),
            "event_kind": "completion",
            "target_date": cur_target.isoformat(),
            "text": (
                f"Final delivery sign-off for {project}: Workstream finally shipped and closed out on "
                f"{cur_target.isoformat()} after repeated timeline adjustments since the initial {initial_target.isoformat()} target."
            ),
        })

    assert len(events) == STRESS_FORECAST_EVENT_COUNT
    return events


# ---------------------------------------------------------------------------
# 3. Track D Stress: 12 Curated 3-Hop Transitive Dependency Cascades
#    Root Trigger (Hop 1 -> Hop 2 Bridge -> Hop 3 Terminal) with ZERO lexical
#    overlap between the Root Trigger / Hop 4 Breakage and Terminal Initiatives.
# ---------------------------------------------------------------------------

STRESS_MULTIHOP_DEPENDENCY_POOLS: List[Dict[str, Any]] = [
    {
        "id": "d_stress_00_egress_campus",
        "domain": "software_engineering",
        "hop1_premise": "Our cloud vendor NebulaCompute guarantees wholesale GPU egress rates at $0.015 per gigabyte, which is the core input powering our Unit Margin Model v4.",
        "hop2_bridge": "Unit Margin Model v4 is the sole financial pillar allowing us to subsidize the Unmetered Student Sandbox Tier without burning runway.",
        "hop3_terminal": "Because the Unmetered Student Sandbox Tier stays open all autumn, we locked in two flagship growth bets: the Autumn Hackathon Sponsorship and the Campus Ambassador Referral Program.",
        "unaffected_premise": "Our corporate legal retainer is fixed annually under a separate multi-year counsel agreement.",
        "unaffected_decision": {
            "id": "dec_patent_filing",
            "label": "annual trademark renewal filing",
            "aliases": ["annual trademark renewal filing", "trademark renewal filing", "trademark renewal"],
            "text": "Using the separate corporate legal retainer, we scheduled our annual trademark renewal filing.",
        },
        "hop4_trigger": "Emergency vendor alert: NebulaCompute just terminated our wholesale discount and tripled GPU egress pricing to $0.045 per gigabyte effective immediately.",
        "affected_decisions": [
            {
                "id": "dec_hackathon",
                "label": "Autumn Hackathon Sponsorship",
                "aliases": ["Autumn Hackathon Sponsorship", "hackathon sponsorship", "autumn hackathon"],
                "reason": "NebulaCompute tripling GPU egress breaks Unit Margin Model v4, which breaks the Unmetered Student Sandbox Tier, invalidating the Autumn Hackathon Sponsorship.",
            },
            {
                "id": "dec_ambassador",
                "label": "Campus Ambassador Referral Program",
                "aliases": ["Campus Ambassador Referral Program", "campus ambassador referral", "ambassador referral program"],
                "reason": "NebulaCompute tripling GPU egress breaks Unit Margin Model v4, which breaks the Unmetered Student Sandbox Tier, invalidating the Campus Ambassador Referral Program.",
            },
        ],
        "questions": {
            "revisit_set": "Following today's emergency vendor pricing alert, which downstream student growth initiatives must be paused or revisited? List every affected initiative.",
            "why": "Trace the full multi-hop dependency chain explaining why the Autumn Hackathon Sponsorship must be revisited after today's alert.",
            "still_standing": "Does our annual trademark renewal filing still stand unaffected after today's vendor pricing alert? Answer YES or NO.",
        },
        "still_standing_target": "annual trademark renewal filing",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_01_substation_bakery",
        "domain": "operations",
        "hop1_premise": "Municipal utility GridWest confirmed our commercial park transformer upgrade will deliver 400-amp three-phase service by May, anchoring our Heavy Equipment Commissioning Plan.",
        "hop2_bridge": "The Heavy Equipment Commissioning Plan is the mandatory prerequisite for firing up the Deck Oven Production Line inside Building B.",
        "hop3_terminal": "Relying on the Deck Oven Production Line running at full yield, we signed two downstream wholesale commitments: the Airport Cafe Pastry Supply Contract and the Downtown Hotel Croissant Standing Order.",
        "unaffected_premise": "Our cold-brew coffee bottling runs out of an independent leased commissary across town with its own power grid.",
        "unaffected_decision": {
            "id": "dec_coldbrew_bottles",
            "label": "cold-brew glass bottle label redesign",
            "aliases": ["cold-brew glass bottle label redesign", "bottle label redesign", "cold-brew label"],
            "text": "At the independent leased commissary, we approved the cold-brew glass bottle label redesign.",
        },
        "hop4_trigger": "Utility notice from GridWest: the commercial park transformer upgrade has been delayed indefinitely until next year due to switchgear shortages.",
        "affected_decisions": [
            {
                "id": "dec_airport_pastry",
                "label": "Airport Cafe Pastry Supply Contract",
                "aliases": ["Airport Cafe Pastry Supply Contract", "airport cafe pastry", "pastry supply contract"],
                "reason": "GridWest delaying the transformer breaks the Heavy Equipment Commissioning Plan, preventing Deck Oven Production Line startup and invalidating the Airport Cafe Pastry Supply Contract.",
            },
            {
                "id": "dec_hotel_croissant",
                "label": "Downtown Hotel Croissant Standing Order",
                "aliases": ["Downtown Hotel Croissant Standing Order", "hotel croissant order", "croissant standing order"],
                "reason": "GridWest delaying the transformer breaks the Heavy Equipment Commissioning Plan, preventing Deck Oven Production Line startup and invalidating the Downtown Hotel Croissant Standing Order.",
            },
        ],
        "questions": {
            "revisit_set": "Given the latest utility notice, which downstream wholesale food commitments must now be renegotiated or revisited?",
            "why": "Explain the full step-by-step chain connecting the utility notice to why the Airport Cafe Pastry Supply Contract is affected.",
            "still_standing": "Does our cold-brew glass bottle label redesign still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "cold-brew glass bottle label redesign",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_02_lithium_fleet",
        "domain": "travel_and_logistics",
        "hop1_premise": "Cell supplier VoltaCore committed to shipping UL-certified solid-state modules by July, which directly governs our Depot Fast-Charge Retrofit Schedule.",
        "hop2_bridge": "Completing the Depot Fast-Charge Retrofit Schedule is the strict operational precondition for launching Overnight Zero-Emission Linehaul Routes.",
        "hop3_terminal": "Because Overnight Zero-Emission Linehaul Routes were scheduled to go live, we committed to two client-facing pledges: the Green Freight Carbon Rebate Filing and the Dawn Fresh Produce Express Guarantee.",
        "unaffected_premise": "Our harbor barge charter operates under a separate maritime fuel-hedging contract.",
        "unaffected_decision": {
            "id": "dec_barge_paint",
            "label": "harbor barge hull anti-fouling repaint",
            "aliases": ["harbor barge hull anti-fouling repaint", "barge hull repaint", "anti-fouling repaint"],
            "text": "Under our separate maritime maintenance budget, we booked the harbor barge hull anti-fouling repaint.",
        },
        "hop4_trigger": "Supplier setback: VoltaCore failed thermal runaway testing and canceled all July solid-state module shipments indefinitely.",
        "affected_decisions": [
            {
                "id": "dec_carbon_rebate",
                "label": "Green Freight Carbon Rebate Filing",
                "aliases": ["Green Freight Carbon Rebate Filing", "carbon rebate filing", "green freight rebate"],
                "reason": "VoltaCore canceling module shipments halts the Depot Fast-Charge Retrofit Schedule, blocking Overnight Zero-Emission Linehaul Routes and invalidating the Green Freight Carbon Rebate Filing.",
            },
            {
                "id": "dec_produce_express",
                "label": "Dawn Fresh Produce Express Guarantee",
                "aliases": ["Dawn Fresh Produce Express Guarantee", "produce express guarantee", "dawn fresh produce"],
                "reason": "VoltaCore canceling module shipments halts the Depot Fast-Charge Retrofit Schedule, blocking Overnight Zero-Emission Linehaul Routes and invalidating the Dawn Fresh Produce Express Guarantee.",
            },
        ],
        "questions": {
            "revisit_set": "After today's supplier setback, which downstream client-facing delivery pledges must be revisited?",
            "why": "Trace the multi-hop chain linking the supplier setback to the Dawn Fresh Produce Express Guarantee.",
            "still_standing": "Does the harbor barge hull anti-fouling repaint still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "harbor barge hull anti-fouling repaint",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_03_visa_berlin",
        "domain": "education_and_career",
        "hop1_premise": "Our principal systems architect Priya expects her EU Blue Card transfer approval by August 1, which determines the Staffing Anchor for the Munich Reliability Pod.",
        "hop2_bridge": "Having the Staffing Anchor for the Munich Reliability Pod on-site is required to qualify for German BaFin Local-Operations Attestation.",
        "hop3_terminal": "Counting on German BaFin Local-Operations Attestation, sales closed two regional rollouts: the Frankfurt Retail Banking Pilot and the Stuttgart Automotive Treasury Integration.",
        "unaffected_premise": "Our Japanese localization team in Tokyo is fully staffed locally and completely independent of European operations.",
        "unaffected_decision": {
            "id": "dec_tokyo_docs",
            "label": "Tokyo developer documentation portal translation",
            "aliases": ["Tokyo developer documentation portal translation", "tokyo documentation translation", "developer documentation portal"],
            "text": "Our independent Tokyo team started the Tokyo developer documentation portal translation.",
        },
        "hop4_trigger": "Immigration update: Priya's EU Blue Card transfer was deferred by six months due to consulate backlog, so she cannot relocate this year.",
        "affected_decisions": [
            {
                "id": "dec_frankfurt_pilot",
                "label": "Frankfurt Retail Banking Pilot",
                "aliases": ["Frankfurt Retail Banking Pilot", "frankfurt banking pilot", "retail banking pilot"],
                "reason": "Priya's relocation delay stalls the Munich Reliability Pod Staffing Anchor, blocking BaFin Local-Operations Attestation and invalidating the Frankfurt Retail Banking Pilot.",
            },
            {
                "id": "dec_stuttgart_treasury",
                "label": "Stuttgart Automotive Treasury Integration",
                "aliases": ["Stuttgart Automotive Treasury Integration", "stuttgart treasury integration", "automotive treasury integration"],
                "reason": "Priya's relocation delay stalls the Munich Reliability Pod Staffing Anchor, blocking BaFin Local-Operations Attestation and invalidating the Stuttgart Automotive Treasury Integration.",
            },
        ],
        "questions": {
            "revisit_set": "Given today's immigration update, which downstream regional customer rollouts must now be revisited?",
            "why": "Explain every hop linking the immigration update to why the Frankfurt Retail Banking Pilot must be paused.",
            "still_standing": "Does the Tokyo developer documentation portal translation still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "Tokyo developer documentation portal translation",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_04_grant_clinic",
        "domain": "health_and_fitness",
        "hop1_premise": "State health agency BioFund tentatively awarded our $1.2M Genomics Seed Grant, which funds the Wet-Lab Sequencing Core Lease.",
        "hop2_bridge": "Operating the Wet-Lab Sequencing Core Lease is the technical foundation enabling our Rapid Biomarker Assay Pipeline.",
        "hop3_terminal": "Built on the Rapid Biomarker Assay Pipeline, our outreach team scheduled two public programs: the Rural Oncology Screening Van Tour and the Pediatric Rare-Disease Registry Drive.",
        "unaffected_premise": "Our annual ergonomic office furniture stipend comes from general hospital overhead reserves.",
        "unaffected_decision": {
            "id": "dec_standing_desks",
            "label": "nurse station ergonomic standing-desk upgrade",
            "aliases": ["nurse station ergonomic standing-desk upgrade", "ergonomic standing-desk upgrade", "standing-desk upgrade"],
            "text": "From general hospital overhead reserves, we ordered the nurse station ergonomic standing-desk upgrade.",
        },
        "hop4_trigger": "Funding reversal: State agency BioFund froze the entire $1.2M Genomics Seed Grant due to legislative budget cuts.",
        "affected_decisions": [
            {
                "id": "dec_screening_van",
                "label": "Rural Oncology Screening Van Tour",
                "aliases": ["Rural Oncology Screening Van Tour", "oncology screening van", "screening van tour"],
                "reason": "BioFund freezing the grant cancels the Wet-Lab Sequencing Core Lease, halting the Rapid Biomarker Assay Pipeline and invalidating the Rural Oncology Screening Van Tour.",
            },
            {
                "id": "dec_pediatric_registry",
                "label": "Pediatric Rare-Disease Registry Drive",
                "aliases": ["Pediatric Rare-Disease Registry Drive", "pediatric registry drive", "rare-disease registry"],
                "reason": "BioFund freezing the grant cancels the Wet-Lab Sequencing Core Lease, halting the Rapid Biomarker Assay Pipeline and invalidating the Pediatric Rare-Disease Registry Drive.",
            },
        ],
        "questions": {
            "revisit_set": "After today's funding reversal, which downstream public outreach programs must be halted or revisited?",
            "why": "Trace the full causal chain linking the BioFund freeze to the Pediatric Rare-Disease Registry Drive.",
            "still_standing": "Does the nurse station ergonomic standing-desk upgrade still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "nurse station ergonomic standing-desk upgrade",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_05_zoning_duplex",
        "domain": "home_renovation",
        "hop1_premise": "County planners indicated our corner lot qualifies for an R-3 Multi-Unit Variance, which is required to approve the Detached Backyard ADU Blueprint.",
        "hop2_bridge": "Building the Detached Backyard ADU Blueprint is the sole way we generate Projected Accessory Rental Cashflow.",
        "hop3_terminal": "Counting on Projected Accessory Rental Cashflow to cover debt service, we locked in two household financial plans: the Alpine Ski Cabin Downpayment Reserve and the Sabbatical Year Family Travel Fund.",
        "unaffected_premise": "My spouse's employer automatically matches 401(k) contributions out of regular W-2 payroll.",
        "unaffected_decision": {
            "id": "dec_401k_match",
            "label": "automatic payroll 401(k) index allocation",
            "aliases": ["automatic payroll 401(k) index allocation", "payroll 401(k) allocation", "401(k) index allocation"],
            "text": "Using regular W-2 payroll deductions, we kept our automatic payroll 401(k) index allocation.",
        },
        "hop4_trigger": "County board ruling: our corner lot R-3 Multi-Unit Variance application was permanently denied under historic district preservation rules.",
        "affected_decisions": [
            {
                "id": "dec_ski_cabin",
                "label": "Alpine Ski Cabin Downpayment Reserve",
                "aliases": ["Alpine Ski Cabin Downpayment Reserve", "ski cabin downpayment", "alpine ski cabin"],
                "reason": "Denying the R-3 variance blocks the Detached Backyard ADU Blueprint, eliminating Projected Accessory Rental Cashflow and invalidating the Alpine Ski Cabin Downpayment Reserve.",
            },
            {
                "id": "dec_sabbatical_travel",
                "label": "Sabbatical Year Family Travel Fund",
                "aliases": ["Sabbatical Year Family Travel Fund", "sabbatical travel fund", "family travel fund"],
                "reason": "Denying the R-3 variance blocks the Detached Backyard ADU Blueprint, eliminating Projected Accessory Rental Cashflow and invalidating the Sabbatical Year Family Travel Fund.",
            },
        ],
        "questions": {
            "revisit_set": "Following the county board ruling, which downstream household financial plans must be revisited?",
            "why": "Trace the multi-hop chain showing why the Sabbatical Year Family Travel Fund is invalidated by the county board ruling.",
            "still_standing": "Does our automatic payroll 401(k) index allocation still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "automatic payroll 401(k) index allocation",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_06_chip_foundry",
        "domain": "software_engineering",
        "hop1_premise": "Foundry partner SiliconArc allocated us 3nm wafer tape-out slots in June, setting the delivery schedule for the Edge NPU Reference Board.",
        "hop2_bridge": "Receiving the Edge NPU Reference Board is the hardware gate for freezing the Offline Voice-Wake Firmware SDK.",
        "hop3_terminal": "Assuming the Offline Voice-Wake Firmware SDK freezes on time, product marketing booked two holiday launches: the Smart Thermostat Retail Endcap Display and the CES Keynote Live Acoustic Demo.",
        "unaffected_premise": "Our web billing dashboard runs purely on cloud servers with no custom silicon dependencies.",
        "unaffected_decision": {
            "id": "dec_invoice_pdf",
            "label": "web billing portal PDF invoice export",
            "aliases": ["web billing portal PDF invoice export", "PDF invoice export", "billing portal invoice export"],
            "text": "For our cloud-only web billing dashboard, we shipped the web billing portal PDF invoice export.",
        },
        "hop4_trigger": "Foundry notice: SiliconArc bumped our 3nm wafer tape-out slots out by nine months due to lithography equipment failures.",
        "affected_decisions": [
            {
                "id": "dec_thermostat_endcap",
                "label": "Smart Thermostat Retail Endcap Display",
                "aliases": ["Smart Thermostat Retail Endcap Display", "thermostat retail endcap", "retail endcap display"],
                "reason": "SiliconArc bumping wafer slots delays the Edge NPU Reference Board, stalling the Offline Voice-Wake Firmware SDK and invalidating the Smart Thermostat Retail Endcap Display.",
            },
            {
                "id": "dec_ces_demo",
                "label": "CES Keynote Live Acoustic Demo",
                "aliases": ["CES Keynote Live Acoustic Demo", "CES keynote demo", "live acoustic demo"],
                "reason": "SiliconArc bumping wafer slots delays the Edge NPU Reference Board, stalling the Offline Voice-Wake Firmware SDK and invalidating the CES Keynote Live Acoustic Demo.",
            },
        ],
        "questions": {
            "revisit_set": "Following today's foundry notice, which downstream holiday marketing launches must be revisited?",
            "why": "Trace every dependency hop from the foundry notice to the CES Keynote Live Acoustic Demo.",
            "still_standing": "Does our web billing portal PDF invoice export still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "web billing portal PDF invoice export",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_07_soc2_enterprise",
        "domain": "legal_and_compliance",
        "hop1_premise": "External auditor TrustGuard confirmed zero critical findings on our encryption key rotation controls, clearing the path for our ISO 27001 Recertification Letter.",
        "hop2_bridge": "Holding a current ISO 27001 Recertification Letter is mandatory for staying listed on the Public-Sector Procurement Approved Vendor Catalog.",
        "hop3_terminal": "Because we remain active on the Public-Sector Procurement Approved Vendor Catalog, account executives scheduled two Q4 engagements: the Municipal Transit Ticketing Bid and the State Library Archive Modernization Proposal.",
        "unaffected_premise": "Our open-source CLI formatter is distributed free under MIT license to hobbyist developers.",
        "unaffected_decision": {
            "id": "dec_cli_darkmode",
            "label": "open-source CLI color theme preset",
            "aliases": ["open-source CLI color theme preset", "CLI color theme", "color theme preset"],
            "text": "For hobbyist users of our MIT tool, we released the open-source CLI color theme preset.",
        },
        "hop4_trigger": "Audit alert: TrustGuard reopened our review and cited a critical finding in encryption key rotation controls, suspending sign-off until next quarter.",
        "affected_decisions": [
            {
                "id": "dec_transit_bid",
                "label": "Municipal Transit Ticketing Bid",
                "aliases": ["Municipal Transit Ticketing Bid", "transit ticketing bid", "municipal transit bid"],
                "reason": "TrustGuard's critical finding blocks ISO 27001 Recertification, removing us from the Public-Sector Approved Vendor Catalog and invalidating the Municipal Transit Ticketing Bid.",
            },
            {
                "id": "dec_library_proposal",
                "label": "State Library Archive Modernization Proposal",
                "aliases": ["State Library Archive Modernization Proposal", "state library proposal", "library archive modernization"],
                "reason": "TrustGuard's critical finding blocks ISO 27001 Recertification, removing us from the Public-Sector Approved Vendor Catalog and invalidating the State Library Archive Modernization Proposal.",
            },
        ],
        "questions": {
            "revisit_set": "After today's audit alert, which downstream Q4 public-sector engagements must be paused or revisited?",
            "why": "Trace the multi-hop chain explaining why the Municipal Transit Ticketing Bid is invalidated by today's audit alert.",
            "still_standing": "Does our open-source CLI color theme preset still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "open-source CLI color theme preset",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_08_interest_refi",
        "domain": "personal_finance",
        "hop1_premise": "Mortgage broker HarborLend quoted a 5.1% thirty-year fixed lock, which underpins our Primary Residence Refinance Closing.",
        "hop2_bridge": "Completing the Primary Residence Refinance Closing is the sole source freeing up $950 per month in Discretionary Household Cash Surplus.",
        "hop3_terminal": "Allocating that Discretionary Household Cash Surplus, we committed to two recurring enrollments: Private Cello Conservatory Lessons and the Weekend Equestrian Riding Club.",
        "unaffected_premise": "Our municipal water and sewer autopay is budgeted out of fixed baseline household utilities.",
        "unaffected_decision": {
            "id": "dec_water_autopay",
            "label": "municipal water utility autopay setup",
            "aliases": ["municipal water utility autopay setup", "water utility autopay", "municipal water autopay"],
            "text": "From our fixed baseline utility account, we maintained our municipal water utility autopay setup.",
        },
        "hop4_trigger": "Lender setback: HarborLend withdrew our 5.1% rate lock after bond yields spiked to 6.8%, making refinancing uneconomical.",
        "affected_decisions": [
            {
                "id": "dec_cello_lessons",
                "label": "Private Cello Conservatory Lessons",
                "aliases": ["Private Cello Conservatory Lessons", "cello conservatory lessons", "private cello lessons"],
                "reason": "HarborLend withdrawing the rate lock cancels the Primary Residence Refinance Closing, eliminating the Discretionary Cash Surplus and invalidating Private Cello Conservatory Lessons.",
            },
            {
                "id": "dec_equestrian_club",
                "label": "Weekend Equestrian Riding Club",
                "aliases": ["Weekend Equestrian Riding Club", "equestrian riding club", "weekend equestrian club"],
                "reason": "HarborLend withdrawing the rate lock cancels the Primary Residence Refinance Closing, eliminating the Discretionary Cash Surplus and invalidating the Weekend Equestrian Riding Club.",
            },
        ],
        "questions": {
            "revisit_set": "Given today's lender setback, which recurring personal enrollments must now be paused or revisited?",
            "why": "Trace the full multi-hop dependency chain connecting HarborLend's rate withdrawal to the Weekend Equestrian Riding Club.",
            "still_standing": "Does our municipal water utility autopay setup still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "municipal water utility autopay setup",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_09_fiber_broadcast",
        "domain": "operations",
        "hop1_premise": "Telecom carrier ClearSpan promised dark-fiber splicing into our studio basement by October 1, enabling our Redundant 100Gbps Uplink Backbone.",
        "hop2_bridge": "The Redundant 100Gbps Uplink Backbone is the mandatory technical safeguard for certifying Uncompressed 8K Live Broadcast Feeds.",
        "hop3_terminal": "With Uncompressed 8K Live Broadcast Feeds certified, programming signed two flagship media deals: the Philharmonic Opening-Night Simulcast and the Esports Arena Championship Stream.",
        "unaffected_premise": "Our vinyl record archive digitization happens completely offline onto local tape drives.",
        "unaffected_decision": {
            "id": "dec_vinyl_tape",
            "label": "offline vinyl archive tape cataloguing",
            "aliases": ["offline vinyl archive tape cataloguing", "vinyl archive cataloguing", "vinyl tape cataloguing"],
            "text": "Using our local tape drives, we continued offline vinyl archive tape cataloguing.",
        },
        "hop4_trigger": "Carrier bulletin: ClearSpan halted all basement dark-fiber trenching until spring due to street-excavation permit suspension.",
        "affected_decisions": [
            {
                "id": "dec_philharmonic",
                "label": "Philharmonic Opening-Night Simulcast",
                "aliases": ["Philharmonic Opening-Night Simulcast", "philharmonic simulcast", "opening-night simulcast"],
                "reason": "ClearSpan halting dark-fiber splicing blocks the 100Gbps Uplink Backbone, preventing 8K Live Broadcast certification and invalidating the Philharmonic Opening-Night Simulcast.",
            },
            {
                "id": "dec_esports_stream",
                "label": "Esports Arena Championship Stream",
                "aliases": ["Esports Arena Championship Stream", "esports championship stream", "esports arena stream"],
                "reason": "ClearSpan halting dark-fiber splicing blocks the 100Gbps Uplink Backbone, preventing 8K Live Broadcast certification and invalidating the Esports Arena Championship Stream.",
            },
        ],
        "questions": {
            "revisit_set": "After today's carrier bulletin, which flagship media deals must be renegotiated or revisited?",
            "why": "Trace the full dependency chain linking ClearSpan's trenching halt to the Philharmonic Opening-Night Simulcast.",
            "still_standing": "Does our offline vinyl archive tape cataloguing still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "offline vinyl archive tape cataloguing",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_10_coldchain_vaccine",
        "domain": "travel_and_logistics",
        "hop1_premise": "Refrigeration maker CryoTech guaranteed delivery of ultra-low -80C compressor units by September, qualifying our Regional Cold-Chain Hub Certification.",
        "hop2_bridge": "Securing Regional Cold-Chain Hub Certification is the regulatory gate for onboarding Temperature-Sensitive Biologics Distribution.",
        "hop3_terminal": "Backed by Temperature-Sensitive Biologics Distribution capacity, our medical division signed two hospital agreements: the Island Pediatric Immunization Shuttle and the Mountain Trauma Plasma Reserve.",
        "unaffected_premise": "Our standard dry-goods gauze and bandage warehouse operates at ambient room temperature.",
        "unaffected_decision": {
            "id": "dec_gauze_shelves",
            "label": "ambient gauze warehouse pallet racking",
            "aliases": ["ambient gauze warehouse pallet racking", "gauze warehouse racking", "pallet racking"],
            "text": "In our ambient dry-goods depot, we installed ambient gauze warehouse pallet racking.",
        },
        "hop4_trigger": "Manufacturer notice: CryoTech issued a recall on its -80C compressor series and pushed replacement deliveries out by eight months.",
        "affected_decisions": [
            {
                "id": "dec_island_immunization",
                "label": "Island Pediatric Immunization Shuttle",
                "aliases": ["Island Pediatric Immunization Shuttle", "pediatric immunization shuttle", "island immunization shuttle"],
                "reason": "CryoTech recalling compressors blocks Regional Cold-Chain Hub Certification, preventing Biologics Distribution and invalidating the Island Pediatric Immunization Shuttle.",
            },
            {
                "id": "dec_trauma_plasma",
                "label": "Mountain Trauma Plasma Reserve",
                "aliases": ["Mountain Trauma Plasma Reserve", "trauma plasma reserve", "mountain plasma reserve"],
                "reason": "CryoTech recalling compressors blocks Regional Cold-Chain Hub Certification, preventing Biologics Distribution and invalidating the Mountain Trauma Plasma Reserve.",
            },
        ],
        "questions": {
            "revisit_set": "Following CryoTech's recall notice, which downstream hospital supply agreements must be revisited?",
            "why": "Explain every hop connecting the CryoTech compressor recall to the Mountain Trauma Plasma Reserve.",
            "still_standing": "Does our ambient gauze warehouse pallet racking still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "ambient gauze warehouse pallet racking",
        "still_standing_gold": "YES",
    },
    {
        "id": "d_stress_11_accreditation_fellowship",
        "domain": "education_and_career",
        "hop1_premise": "University senate approved our joint faculty charter with Northbridge Institute, which unlocks Title IV Cross-Enrollment Eligibility.",
        "hop2_bridge": "Title IV Cross-Enrollment Eligibility is the sole administrative vehicle permitting Federal Work-Study Stipend Matching.",
        "hop3_terminal": "Relying on Federal Work-Study Stipend Matching to fund student hours, the department launched two community projects: the Coastal Estuary Water-Testing Fellowship and the High-School Robotics Mentorship Corps.",
        "unaffected_premise": "Our alumni magazine is printed annually from a dedicated endowed publishing trust.",
        "unaffected_decision": {
            "id": "dec_alumni_magazine",
            "label": "annual alumni magazine winter print issue",
            "aliases": ["annual alumni magazine winter print issue", "alumni magazine print issue", "winter print issue"],
            "text": "Using the endowed publishing trust, we finalized the annual alumni magazine winter print issue.",
        },
        "hop4_trigger": "Governance setback: The university senate rescinded the Northbridge Institute joint faculty charter over accreditation bylaw conflicts.",
        "affected_decisions": [
            {
                "id": "dec_estuary_fellowship",
                "label": "Coastal Estuary Water-Testing Fellowship",
                "aliases": ["Coastal Estuary Water-Testing Fellowship", "estuary water-testing fellowship", "water-testing fellowship"],
                "reason": "Rescinding the joint charter revokes Title IV Cross-Enrollment Eligibility, terminating Federal Work-Study Stipend Matching and invalidating the Coastal Estuary Water-Testing Fellowship.",
            },
            {
                "id": "dec_robotics_corps",
                "label": "High-School Robotics Mentorship Corps",
                "aliases": ["High-School Robotics Mentorship Corps", "robotics mentorship corps", "high-school robotics mentorship"],
                "reason": "Rescinding the joint charter revokes Title IV Cross-Enrollment Eligibility, terminating Federal Work-Study Stipend Matching and invalidating the High-School Robotics Mentorship Corps.",
            },
        ],
        "questions": {
            "revisit_set": "After the senate rescinded the joint charter, which downstream community student projects must be paused or revisited?",
            "why": "Trace the full multi-hop dependency chain from the charter rescission to the Coastal Estuary Water-Testing Fellowship.",
            "still_standing": "Does our annual alumni magazine winter print issue still stand unaffected? Answer YES or NO.",
        },
        "still_standing_target": "annual alumni magazine winter print issue",
        "still_standing_gold": "YES",
    },
]


# ---------------------------------------------------------------------------
# 4. Track X Stress: 12 Strictly Zero-Content-Word-Overlap Constraints
#    Verified by test_epistemic_stress_v2_2.py to have EXACTLY 0 shared
#    content words between constraint_statement and query!
# ---------------------------------------------------------------------------

STRESS_ZERO_OVERLAP_CONSTRAINT_POOLS: List[Dict[str, Any]] = [
    {
        "id": "x_stress_00_warfarin_vitk",
        "domain": "health_and_fitness",
        "constraint_statement": "My hematologist placed me on daily warfarin anticoagulation following a deep-vein thrombosis, requiring strictly consistent minimal Vitamin K intake.",
        "query": "Blending a huge bowl of raw curly kale and baby spinach pesto for dinner tonight — got any bright herb seasoning ideas?",
        "gold_response_requirement": "Must warn against eating a huge bowl of raw kale and spinach because dark leafy greens are very high in Vitamin K, which counteracts the user's daily warfarin anticoagulation therapy.",
        "constraint_keyword": "warfarin",
        "constraint_aliases": ["warfarin", "Vitamin K", "anticoagulation", "deep-vein thrombosis"],
        "gold_verdict": "warn_warfarin_vitamin_k",
    },
    {
        "id": "x_stress_01_noncompete_checkout",
        "domain": "legal_and_compliance",
        "constraint_statement": "My executive severance covenant legally bars advising or holding equity in any merchant acquiring or card-processing venture for twelve months.",
        "query": "My cousin invited me to mentor his early-stage online checkout startup on weekends in exchange for half a point of founder stock — sound smart?",
        "gold_response_requirement": "Must warn against mentoring or taking stock in the checkout startup because it likely violates the user's 12-month severance non-compete barring advising or holding equity in merchant acquiring / card-processing ventures.",
        "constraint_keyword": "severance covenant",
        "constraint_aliases": ["severance covenant", "merchant acquiring", "card-processing", "twelve months"],
        "gold_verdict": "blocked_noncompete_covenant",
    },
    {
        "id": "x_stress_02_breaker_amperage",
        "domain": "home_renovation",
        "constraint_statement": "Our vintage 1960s kitchen wiring sits on a single 15-amp circuit breaker that trips immediately above 1,800 total watts.",
        "query": "Planning to warm the breakfast nook with a 1,500-watt ceramic space heater while pulling shots on our dual-boiler espresso machine — any practical tips?",
        "gold_response_requirement": "Must warn that running a 1,500-watt space heater simultaneously with a dual-boiler espresso machine will exceed 1,800 watts and trip the 15-amp kitchen circuit breaker.",
        "constraint_keyword": "15-amp",
        "constraint_aliases": ["15-amp", "1,800", "circuit breaker", "1800"],
        "gold_verdict": "warn_circuit_overload",
    },
    {
        "id": "x_stress_03_statin_grapefruit",
        "domain": "health_and_fitness",
        "constraint_statement": "My cardiologist prescribed nightly atorvastatin for elevated lipids and explicitly banned furanocoumarin citrus juices due to CYP3A4 toxicity.",
        "query": "Picked up a fresh crate of ruby red pomelos and pink grapefruits at the farmers market for morning smoothies — recommend a good blend?",
        "gold_response_requirement": "Must warn against drinking grapefruit or pomelo smoothies because furanocoumarins inhibit CYP3A4 and interact dangerously with the user's nightly atorvastatin prescription.",
        "constraint_keyword": "atorvastatin",
        "constraint_aliases": ["atorvastatin", "CYP3A4", "furanocoumarin", "cardiologist"],
        "gold_verdict": "warn_statin_citrus_interaction",
    },
    {
        "id": "x_stress_04_arachis_maafe",
        "domain": "health_and_fitness",
        "constraint_statement": "Our youngest daughter carries an EpiPen for severe arachis anaphylaxis, meaning zero legume-nut products may ever enter the kitchen.",
        "query": "Hosting neighbors Friday evening and excited to simmer authentic Senegalese maafe stew alongside Indonesian satay skewers — what sides pair well?",
        "gold_response_requirement": "Must stop the user from making Senegalese maafe and Indonesian satay because both traditionally rely on peanut sauce, violating the strict zero-arachis anaphylaxis house safety rule.",
        "constraint_keyword": "EpiPen",
        "constraint_aliases": ["EpiPen", "arachis", "anaphylaxis", "legume-nut"],
        "gold_verdict": "blocked_peanut_anaphylaxis",
    },
    {
        "id": "x_stress_05_itar_zurich",
        "domain": "legal_and_compliance",
        "constraint_statement": "Our aerospace telemetry codebase contains ITAR-controlled defense technical data restricted strictly to verified US persons.",
        "query": "Our brilliant visiting summer intern from Zurich arrives Monday — should I grant him clone credentials on the flight-guidance repository right away?",
        "gold_response_requirement": "Must refuse granting repository access to the visiting intern from Zurich until citizenship/export authorization is confirmed, citing ITAR restrictions limiting defense technical data to US persons.",
        "constraint_keyword": "ITAR",
        "constraint_aliases": ["ITAR", "US persons", "defense technical data", "aerospace telemetry"],
        "gold_verdict": "blocked_itar_export_control",
    },
    {
        "id": "x_stress_06_scuba_flight",
        "domain": "travel_and_logistics",
        "constraint_statement": "I follow strict PADI decompression safety protocols requiring a full 24-hour surface interval after repetitive deep-water tank dives.",
        "query": "Wrapping up our afternoon two-cylinder 90-foot Cozumel reef excursion around 3 PM — should I book the 9 PM evening jet home to Miami?",
        "gold_response_requirement": "Must advise against booking the 9 PM flight (only 6 hours after diving) because it violates the mandatory 24-hour surface interval after repetitive deep dives to avoid decompression sickness.",
        "constraint_keyword": "24-hour surface interval",
        "constraint_aliases": ["24-hour", "surface interval", "PADI", "decompression"],
        "gold_verdict": "blocked_no_fly_after_dive",
    },
    {
        "id": "x_stress_07_galvanic_aluminum",
        "domain": "home_renovation",
        "constraint_statement": "Our basement radiant-heating manifold uses bare aluminum coils that suffer rapid galvanic pitting if coupled with copper-alloy plumbing.",
        "query": "Picking up unlined bronze ball valves and red-metal piping at the hardware store tomorrow to extend the downstairs hydronic loop — solid parts choice?",
        "gold_response_requirement": "Must warn against using bronze valves and copper/red-metal piping on the downstairs loop because coupling copper alloys with the bare aluminum radiant-heating coils causes rapid galvanic pitting.",
        "constraint_keyword": "galvanic pitting",
        "constraint_aliases": ["galvanic pitting", "bare aluminum", "copper-alloy", "radiant-heating manifold"],
        "gold_verdict": "warn_galvanic_corrosion",
    },
    {
        "id": "x_stress_08_celiac_guinness",
        "domain": "health_and_fitness",
        "constraint_statement": "I have biopsy-confirmed celiac disease and suffer severe intestinal damage from even trace wheat, rye, or barley gluten.",
        "query": "Cooking a cozy pub-style tasting supper tonight featuring classic stout beef braise and crispy malt-vinegar chips — any garnish suggestions?",
        "gold_response_requirement": "Must warn against stout beer and malt vinegar because both are derived from barley and contain gluten, which causes severe intestinal damage given the user's biopsy-confirmed celiac disease.",
        "constraint_keyword": "celiac",
        "constraint_aliases": ["celiac", "barley", "gluten", "intestinal damage"],
        "gold_verdict": "blocked_celiac_barley_gluten",
    },
    {
        "id": "x_stress_09_sec_blackout",
        "domain": "personal_finance",
        "constraint_statement": "As a Section 16 corporate officer, insider-trading policy imposes an absolute equity blackout starting fourteen days prior to quarter-end until earnings disclosure.",
        "query": "Our fiscal Q3 wraps up next Friday and shares rallied 18% today — should I exercise and sell half my vested options at market open tomorrow?",
        "gold_response_requirement": "Must warn against selling shares tomorrow because next Friday is within 14 days of fiscal quarter-end, placing the Section 16 officer inside the mandatory corporate insider-trading blackout window.",
        "constraint_keyword": "blackout",
        "constraint_aliases": ["blackout", "Section 16", "insider-trading", "fourteen days"],
        "gold_verdict": "blocked_sec_blackout_window",
    },
    {
        "id": "x_stress_10_latex_workshop",
        "domain": "health_and_fitness",
        "constraint_statement": "I suffer acute Type-I natural rubber latex hypersensitivity causing bronchospasm and urticaria upon dermal contact.",
        "query": "Restocking our garage woodshop and first-aid cabinet with bulk powdered examination gloves and woven elastic compression wraps — which budget packs hold up best?",
        "gold_response_requirement": "Must remind the user to purchase strictly latex-free (e.g., nitrile) gloves and synthetic elastic wraps because standard powdered exam gloves and elastic wraps often contain natural rubber latex, risking bronchospasm.",
        "constraint_keyword": "latex",
        "constraint_aliases": ["latex", "natural rubber", "bronchospasm", "hypersensitivity"],
        "gold_verdict": "warn_latex_allergy_use_nitrile",
    },
    {
        "id": "x_stress_11_lithium_cargo",
        "domain": "travel_and_logistics",
        "constraint_statement": "Aviation hazardous-materials rules strictly forbid stowing loose lithium-ion energy cells inside aircraft cargo holds due to thermal-runaway fire risk.",
        "query": "Tucking my hefty 27,000 mAh portable phone charger brick and four spare mirrorless camera packs deep within my checked suitcase for tomorrow's flight — smart way to lighten my backpack?",
        "gold_response_requirement": "Must tell the user NOT to pack the portable power bank or spare camera batteries in checked luggage because aviation hazmat rules forbid lithium-ion cells in cargo holds; they must go in carry-on.",
        "constraint_keyword": "lithium-ion",
        "constraint_aliases": ["lithium-ion", "cargo holds", "hazardous-materials", "thermal-runaway"],
        "gold_verdict": "blocked_lithium_in_checked_bag",
    },
]


# ---------------------------------------------------------------------------
# 5. Track I Stress: 12 Curated 16-Session Longitudinal Portfolio Experiments
#    Each scenario generates 16 evidence sessions (~1,900+ tokens total,
#    exceeding a 1,500-token budget!) covering 4 distinct interventions so
#    cross-entity aggregation and confounder separation require stateful summary.
# ---------------------------------------------------------------------------

STRESS_PORTFOLIO_SESSION_COUNT = 16

STRESS_PORTFOLIO_INTERVENTION_POOLS: List[Dict[str, Any]] = [
    {
        "id": "i_stress_00_sprint_carryover",
        "domain": "software_engineering",
        "subject": "engineering sprint carryover tickets",
        "metric": "unfinished tickets per sprint",
        "clean_name": "pre-sprint dependency grooming checklist",
        "confounded_name": "strict WIP cap of two tickets per engineer",
        "confounder_note": "headcount simultaneously doubled from 4 to 8 engineers and scope was cut in half",
        "unimplemented_name": "automated PR size linter",
        "anecdotal_name": "no-meeting Wednesdays",
        "baseline_vals": ["9 tickets", "8 tickets", "9 tickets", "10 tickets"],
        "clean_post_vals": ["2 tickets", "2 tickets", "1 ticket"],
        "confounded_post_vals": ["3 tickets", "2 tickets", "3 tickets"],
        "anecdotal_val": "4 tickets (single week observation)",
    },
    {
        "id": "i_stress_01_support_triage",
        "domain": "operations",
        "subject": "customer support first-response time",
        "metric": "median first-response latency",
        "clean_name": "tiered severity auto-routing macro",
        "confounded_name": "weekend on-call rotation",
        "confounder_note": "a major enterprise client paused onboarding at the exact same time, dropping ticket volume 60%",
        "unimplemented_name": "AI draft-reply sidebar",
        "anecdotal_name": "hourly Slack queue alerts",
        "baseline_vals": ["4.6 hours", "4.8 hours", "4.5 hours", "4.9 hours"],
        "clean_post_vals": ["1.1 hours", "1.0 hours", "1.2 hours"],
        "confounded_post_vals": ["1.4 hours", "1.5 hours", "1.3 hours"],
        "anecdotal_val": "2.1 hours (single shift observation)",
    },
    {
        "id": "i_stress_02_sleep_efficiency",
        "domain": "health_and_fitness",
        "subject": "deep sleep duration",
        "metric": "nightly deep sleep minutes",
        "clean_name": "zero caffeine after 1 PM cutoff",
        "confounded_name": "magnesium glycinate supplement",
        "confounder_note": "simultaneously went on a relaxing two-week beach vacation with zero work alarm",
        "unimplemented_name": "blackout motorized blinds",
        "anecdotal_name": "weighted blanket trial",
        "baseline_vals": ["46 minutes", "44 minutes", "48 minutes", "45 minutes"],
        "clean_post_vals": ["78 minutes", "81 minutes", "79 minutes"],
        "confounded_post_vals": ["74 minutes", "76 minutes", "72 minutes"],
        "anecdotal_val": "63 minutes (single night observation)",
    },
    {
        "id": "i_stress_03_ci_build_time",
        "domain": "software_engineering",
        "subject": "CI test pipeline duration",
        "metric": "p90 CI build duration",
        "clean_name": "remote Bazel artifact caching",
        "confounded_name": "splitting integration tests into 4 shards",
        "confounder_note": "we simultaneously disabled the entire browser end-to-end test suite during that window",
        "unimplemented_name": "ARM64 runner migration",
        "anecdotal_name": "Docker layer squashing",
        "baseline_vals": ["29 minutes", "31 minutes", "30 minutes", "32 minutes"],
        "clean_post_vals": ["9 minutes", "8 minutes", "9 minutes"],
        "confounded_post_vals": ["11 minutes", "10 minutes", "12 minutes"],
        "anecdotal_val": "18 minutes (single run observation)",
    },
    {
        "id": "i_stress_04_checkout_conversion",
        "domain": "operations",
        "subject": "mobile checkout conversion rate",
        "metric": "completed checkout percentage",
        "clean_name": "one-tap Apple Pay express button",
        "confounded_name": "redesigned green checkout CTA color",
        "confounder_note": "marketing simultaneously ran a 40% Black Friday sitewide discount banner",
        "unimplemented_name": "SMS cart-recovery reminders",
        "anecdotal_name": "progress-bar step header",
        "baseline_vals": ["2.1%", "2.0%", "2.2%", "2.1%"],
        "clean_post_vals": ["3.8%", "3.9%", "3.7%"],
        "confounded_post_vals": ["3.6%", "3.5%", "3.7%"],
        "anecdotal_val": "2.9% (single Tuesday observation)",
    },
    {
        "id": "i_stress_05_migraine_days",
        "domain": "health_and_fitness",
        "subject": "monthly migraine frequency",
        "metric": "migraine episodes per two weeks",
        "clean_name": "consistent 7 AM wake-time schedule",
        "confounded_name": "blue-light filtering desk glasses",
        "confounder_note": "simultaneously started prescription CGRP preventive injections from the neurologist",
        "unimplemented_name": "low-tyramine elimination diet",
        "anecdotal_name": "peppermint temple oil",
        "baseline_vals": ["5 episodes", "6 episodes", "5 episodes", "5 episodes"],
        "clean_post_vals": ["1 episode", "1 episode", "0 episodes"],
        "confounded_post_vals": ["2 episodes", "1 episode", "2 episodes"],
        "anecdotal_val": "2 episodes (single week observation)",
    },
    {
        "id": "i_stress_06_warehouse_pick",
        "domain": "travel_and_logistics",
        "subject": "warehouse order-picking error rate",
        "metric": "mis-picked orders per thousand",
        "clean_name": "barcode scan confirmation on every bin",
        "confounded_name": "rearranged aisle color signage",
        "confounder_note": "order SKU catalog was simultaneously pruned by 70% during seasonal clearance",
        "unimplemented_name": "voice-directed picking headsets",
        "anecdotal_name": "morning huddle reminder",
        "baseline_vals": ["14 per 1000", "15 per 1000", "13 per 1000", "14 per 1000"],
        "clean_post_vals": ["2 per 1000", "1 per 1000", "2 per 1000"],
        "confounded_post_vals": ["4 per 1000", "3 per 1000", "4 per 1000"],
        "anecdotal_val": "7 per 1000 (single day observation)",
    },
    {
        "id": "i_stress_07_study_retention",
        "domain": "education_and_career",
        "subject": "weekly Japanese vocabulary retention",
        "metric": "recall accuracy on Friday quiz",
        "clean_name": "spaced-repetition active recall flashcards",
        "confounded_name": "lo-fi instrumental study playlist",
        "confounder_note": "simultaneously switched to a much easier beginner JLPT N5 word deck",
        "unimplemented_name": "daily kanji handwriting workbook",
        "anecdotal_name": "studying at the library café",
        "baseline_vals": ["58%", "56%", "59%", "57%"],
        "clean_post_vals": ["89%", "91%", "90%"],
        "confounded_post_vals": ["86%", "88%", "85%"],
        "anecdotal_val": "76% (single quiz observation)",
    },
    {
        "id": "i_stress_08_cloud_spend",
        "domain": "personal_finance",
        "subject": "monthly AWS compute expenditure",
        "metric": "weekly compute burn",
        "clean_name": "automated nightly dev-cluster shutdown cron",
        "confounded_name": "switching log verbosity from INFO to WARN",
        "confounder_note": "we simultaneously decommissioned the entire legacy staging environment",
        "unimplemented_name": "3-year Savings Plan commitment",
        "anecdotal_name": "manual spot-instance bid tweak",
        "baseline_vals": ["$4,200/wk", "$4,350/wk", "$4,150/wk", "$4,300/wk"],
        "clean_post_vals": ["$2,450/wk", "$2,400/wk", "$2,500/wk"],
        "confounded_post_vals": ["$2,700/wk", "$2,650/wk", "$2,750/wk"],
        "anecdotal_val": "$3,400/wk (single week observation)",
    },
    {
        "id": "i_stress_09_api_p99",
        "domain": "software_engineering",
        "subject": "search endpoint p99 latency",
        "metric": "p99 response latency",
        "clean_name": "composite covering index on tenant_id and updated_at",
        "confounded_name": "switching HTTP JSON serializer to orjson",
        "confounder_note": "traffic simultaneously dropped 65% during the December holiday freeze",
        "unimplemented_name": "read-replica connection pooling",
        "anecdotal_name": "increasing Gunicorn workers from 4 to 6",
        "baseline_vals": ["840 ms", "860 ms", "830 ms", "855 ms"],
        "clean_post_vals": ["125 ms", "118 ms", "122 ms"],
        "confounded_post_vals": ["210 ms", "195 ms", "205 ms"],
        "anecdotal_val": "490 ms (single afternoon observation)",
    },
    {
        "id": "i_stress_10_hvac_kwh",
        "domain": "home_renovation",
        "subject": "weekly home electricity usage",
        "metric": "weekly kilowatt-hours consumed",
        "clean_name": "attic air-sealing and R-49 blown insulation",
        "confounded_name": "smart thermostat eco-schedule mode",
        "confounder_note": "outdoor temperatures simultaneously cooled by 22 degrees into mild autumn weather",
        "unimplemented_name": "heat-pump water heater retrofit",
        "anecdotal_name": "closing upstairs guest-room vents",
        "baseline_vals": ["310 kWh", "325 kWh", "318 kWh", "320 kWh"],
        "clean_post_vals": ["195 kWh", "190 kWh", "198 kWh"],
        "confounded_post_vals": ["205 kWh", "210 kWh", "200 kWh"],
        "anecdotal_val": "265 kWh (single mild week observation)",
    },
    {
        "id": "i_stress_11_churn_renewal",
        "domain": "operations",
        "subject": "SMB quarterly renewal retention",
        "metric": "monthly cohort retention rate",
        "clean_name": "day-30 live customer onboarding health check",
        "confounded_name": "redesigning the monthly usage digest email",
        "confounder_note": "sales simultaneously switched new signups from monthly billing to mandatory annual contracts",
        "unimplemented_name": "in-app NPS survey popup",
        "anecdotal_name": "handwritten renewal thank-you card",
        "baseline_vals": ["76%", "75%", "77%", "76%"],
        "clean_post_vals": ["91%", "92%", "90%"],
        "confounded_post_vals": ["88%", "89%", "87%"],
        "anecdotal_val": "83% (single cohort observation)",
    },
]


def build_stress_portfolio_sessions(pool_item: Dict[str, Any], base_date: datetime) -> List[Dict[str, Any]]:
    """Build 16 longitudinal portfolio experiment sessions (~1,900+ tokens) for Track I stress."""
    subj = pool_item["subject"]
    metric = pool_item["metric"]
    clean = pool_item["clean_name"]
    conf = pool_item["confounded_name"]
    conf_note = pool_item["confounder_note"]
    unimp = pool_item["unimplemented_name"]
    anec = pool_item["anecdotal_name"]
    b_vals = pool_item["baseline_vals"]
    c_vals = pool_item["clean_post_vals"]
    f_vals = pool_item["confounded_post_vals"]
    a_val = pool_item["anecdotal_val"]

    entries: List[str] = [
        # 4 baseline checkpoints
        (
            f"Portfolio Experiment Log #1 ({subj}): Baseline period 1 recorded prior to launching any experimental changes. "
            f"Measured {metric} at {b_vals[0]} across a full standard operating cycle with stable staffing, unchanged workload volume, "
            f"and verified telemetry instrumentation logged in the central review workbook for longitudinal comparison."
        ),
        (
            f"Portfolio Experiment Log #2 ({subj}): Baseline period 2 confirming pre-intervention stability before candidate rollout. "
            f"Measured {metric} at {b_vals[1]} under identical calendar and resource conditions with zero concurrent workflow modifications, "
            f"establishing repeatable control variance across consecutive measurement intervals."
        ),
        (
            f"Portfolio Experiment Log #3 ({subj}): Baseline period 3 logged prior to evaluating any proposed interventions. "
            f"Measured {metric} at {b_vals[2]}; baseline variance remains tight with no external seasonal anomalies, staffing shifts, "
            f"or tooling disruptions observed during the full audit window."
        ),
        (
            f"Portfolio Experiment Log #4 ({subj}): Baseline period 4 finalized and locked as the reference benchmark. "
            f"Measured {metric} at {b_vals[3]}. We now hold four consecutive clean baseline observations documenting normal "
            f"pre-change performance before testing Candidates A, B, C, and D."
        ),
        # 4 sessions for CLEAN intervention (rollout + 3 clean post observations)
        (
            f"Portfolio Experiment Log #5 ({subj}): Launched Candidate A — '{clean}' — as an isolated single-variable experiment. "
            f"Confirmed that every other operational parameter, workload volume, staffing ratio, and external calendar condition "
            f"remained completely unchanged relative to the four baseline periods."
        ),
        (
            f"Portfolio Experiment Log #6 ({subj}): Post-change measurement 1 for Candidate A ('{clean}') under strictly clean control conditions: "
            f"{metric} improved decisively to {c_vals[0]} compared against the {b_vals[0]}–{b_vals[3]} baseline range, "
            f"with zero concurrent confounders, holidays, or scope reductions logged."
        ),
        (
            f"Portfolio Experiment Log #7 ({subj}): Post-change measurement 2 for Candidate A ('{clean}') continuing under clean controlled conditions: "
            f"{metric} remained sustained at {c_vals[1]} under normal full operating load, replicating the initial gain "
            f"across a second independent review interval with no external changes."
        ),
        (
            f"Portfolio Experiment Log #8 ({subj}): Post-change measurement 3 for Candidate A ('{clean}') completing the longitudinal confirmation window: "
            f"{metric} held steady at {c_vals[2]}, establishing three consecutive unconfounded post-intervention measurements "
            f"clearly separating '{clean}' from the pre-change baseline."
        ),
        # 4 sessions for CONFOUNDED intervention
        (
            f"Portfolio Experiment Log #9 ({subj}): Initiated trial of Candidate B — '{conf}' — during a separate review cycle. "
            f"Critical experimental note: this trial window was immediately compromised by a major confounder because {conf_note}, "
            f"violating single-variable isolation."
        ),
        (
            f"Portfolio Experiment Log #10 ({subj}): Observation 1 during the Candidate B ('{conf}') window recorded {metric} at {f_vals[0]}, "
            f"but the result remains heavily confounded because {conf_note}, making it impossible to separate the intervention effect "
            f"from the simultaneous external shift."
        ),
        (
            f"Portfolio Experiment Log #11 ({subj}): Observation 2 during the Candidate B ('{conf}') window recorded {metric} at {f_vals[1]}, "
            f"still overlapping the exact same confounder ({conf_note}); stakeholders explicitly flagged that causal attribution "
            f"to '{conf}' cannot be supported from these readings."
        ),
        (
            f"Portfolio Experiment Log #12 ({subj}): Observation 3 closing the Candidate B ('{conf}') review window recorded {metric} at {f_vals[2]}. "
            f"Because {conf_note} persisted across all three readings, '{conf}' remains strictly confounded and cannot be credited "
            f"with clean causal efficacy."
        ),
        # 2 sessions for UNIMPLEMENTED intervention
        (
            f"Portfolio Experiment Log #13 ({subj}): Candidate C — '{unimp}' — was drafted and discussed in quarterly planning notes, "
            f"but implementation was deferred indefinitely due to backlog and resource prioritization, so it was never actually deployed "
            f"into active day-to-day operations."
        ),
        (
            f"Portfolio Experiment Log #14 ({subj}): Operational audit verified that Candidate C ('{unimp}') is STILL not implemented anywhere "
            f"in the live workflow. Any claim crediting '{unimp}' with improving {metric} would be invalid because zero deployment "
            f"or post-change observations exist."
        ),
        # 2 sessions for SINGLE-OBSERVATION anecdotal intervention
        (
            f"Portfolio Experiment Log #15 ({subj}): Briefly tested Candidate D — '{anec}' — during one single isolated period "
            f"without pre-registered controls and logged an anecdotal reading of {a_val} before reverting back to routine operations "
            f"the following cycle."
        ),
        (
            f"Portfolio Experiment Log #16 ({subj}): Final review of Candidate D ('{anec}'): no follow-up or repeat measurements were ever collected "
            f"beyond that lone anecdotal observation ({a_val}). With only N=1 post-change data point and no longitudinal replication, "
            f"evidence for '{anec}' is insufficient to claim causal efficacy."
        ),
    ]

    assert len(entries) == STRESS_PORTFOLIO_SESSION_COUNT
    sessions: List[Dict[str, Any]] = []
    for idx, text in enumerate(entries):
        s_dt = base_date + timedelta(days=10 * idx)
        sessions.append({
            "session_id": f"s_portfolio_{idx + 1:02d}",
            "session_date": s_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "turns": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "Logged this longitudinal experiment entry."},
            ],
            "is_evidence": True,
        })
    return sessions
