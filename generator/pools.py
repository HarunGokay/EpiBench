"""EpiBench Entity and Value Pools.

Contains diverse domain entities, attributes, values, dates, and realistic
conversational templates across 5 practical domains.
Strictly decoupled from any specific memory architecture.
Zero proprietary vocabulary or internal system references.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

DOMAINS: List[str] = [
    "software_engineering",
    "personal_finance",
    "health_and_fitness",
    "home_renovation",
    "travel_and_logistics",
]

# Track R (Revision) Entities and Value Chains
# Each template provides a sequence of values that change over time,
# representing realistic life/project state transitions.
REVISION_POOLS: List[Dict[str, Any]] = [
    {
        "domain": "software_engineering",
        "entity": "production database",
        "attribute": "primary database engine",
        "values": ["PostgreSQL 14", "CockroachDB", "PostgreSQL 16"],
        "entity_phrase": "the production database",
        "question_templates": {
            "current": "What is the team's current primary production database engine?",
            "previous": "What was the team's primary production database engine before the current one?",
            "as_of": "What was the primary database engine as of {date}?",
            "count": "How many times did the team switch or upgrade their primary database engine?",
        },
        "user_phrasings": [
            "We are currently running our primary database on {val}.",
            "We migrated our production database over to {val}.",
            "Due to latency issues, we just switched our primary database to {val}.",
        ],
    },
    {
        "domain": "health_and_fitness",
        "entity": "gym membership",
        "attribute": "current gym",
        "values": ["Equinox", "Crunch Fitness", "Lifetime Fitness"],
        "entity_phrase": "gym membership",
        "question_templates": {
            "current": "Which gym does the user currently attend?",
            "previous": "Which gym did the user attend right before their current gym?",
            "as_of": "Which gym was the user attending as of {date}?",
            "count": "How many times has the user changed gyms?",
        },
        "user_phrasings": [
            "I finally signed up for a membership at {val}.",
            "I cancelled my old gym and joined {val} instead.",
            "I wasn't happy with the crowd, so I moved my membership to {val}.",
        ],
    },
    {
        "domain": "personal_finance",
        "entity": "monthly savings target",
        "attribute": "monthly savings goal",
        "values": ["$750", "$1,200", "$900"],
        "entity_phrase": "monthly savings target",
        "question_templates": {
            "current": "What is the user's current monthly savings target?",
            "previous": "What was the user's monthly savings target prior to the current goal?",
            "as_of": "What was the monthly savings target as of {date}?",
            "count": "How many times has the user adjusted their monthly savings target?",
        },
        "user_phrasings": [
            "I'm establishing a strict monthly savings goal of {val}.",
            "I got a raise, so I'm increasing my monthly savings target to {val}.",
            "Unexpected car expenses came up, so I had to revise my monthly savings goal to {val}.",
        ],
    },
    {
        "domain": "software_engineering",
        "entity": "frontend framework",
        "attribute": "frontend framework",
        "values": ["React", "Vue 3", "SvelteKit"],
        "entity_phrase": "frontend framework",
        "question_templates": {
            "current": "What frontend framework is the web app currently using?",
            "previous": "What frontend framework was used before the current one?",
            "as_of": "What frontend framework was in use as of {date}?",
            "count": "How many times did the web team switch frontend frameworks?",
        },
        "user_phrasings": [
            "Our web app is built on {val}.",
            "We decided to rewrite our frontend components in {val}.",
            "We just completed the migration of our front-end over to {val}.",
        ],
    },
    {
        "domain": "travel_and_logistics",
        "entity": "primary airline loyalty program",
        "attribute": "preferred airline",
        "values": ["Delta Air Lines", "United Airlines", "American Airlines"],
        "entity_phrase": "preferred airline",
        "question_templates": {
            "current": "What airline loyalty program does the user currently use as their primary airline?",
            "previous": "Which airline was the user's primary choice before the current one?",
            "as_of": "Which airline was the user's primary airline as of {date}?",
            "count": "How many times did the user switch primary airlines?",
        },
        "user_phrasings": [
            "I'm booking all my business travel on {val} to build status.",
            "Delta devalued their points, so I status matched and switched over to {val}.",
            "My home airport changed routes, so I moved my primary loyalty over to {val}.",
        ],
    },
    {
        "domain": "home_renovation",
        "entity": "kitchen countertop material",
        "attribute": "selected countertop material",
        "values": ["Carrara Marble", "Engineered Quartz", "Honed Granite"],
        "entity_phrase": "kitchen countertop material",
        "question_templates": {
            "current": "What kitchen countertop material has the user currently chosen?",
            "previous": "What countertop material did the user have picked out right before the current choice?",
            "as_of": "What countertop material was chosen as of {date}?",
            "count": "How many times did the user change their kitchen countertop selection?",
        },
        "user_phrasings": [
            "We picked out {val} for the kitchen counters.",
            "The marble supplier warned about staining, so we switched our kitchen counter choice to {val}.",
            "We found a great deal on {val} and finalized that for the kitchen.",
        ],
    },
]

# Track D (Dependency) Templates
# Decisions resting on explicit assumptions. When assumption A1 is invalidated,
# decisions D_affected need revisiting, while decisions resting on A2 remain standing.
DEPENDENCY_POOLS: List[Dict[str, Any]] = [
    {
        "domain": "software_engineering",
        "assumption_1": "The mobile app release date is scheduled for October 15th.",
        "assumption_1_revised": "The mobile app release date has been postponed to December 1st.",
        "affected_decisions": [
            {
                "id": "decision_ad_campaign",
                "label": "paid mobile advertising campaign",
                "text": "Because the app launches on October 15th, I scheduled the $5,000 paid launch ad campaign to start October 16th.",
                "reason": "The ad campaign start date was aligned with the original October 15th app launch.",
            },
            {
                "id": "decision_press_embargo",
                "label": "press embargo lifting date",
                "text": "Based on the October 15th launch date, we set the tech blog press embargo to lift on the morning of October 15th.",
                "reason": "The press embargo was scheduled specifically for the October 15th release morning.",
            },
        ],
        "assumption_2": "Our maximum monthly cloud infrastructure budget is $1,500.",
        "unaffected_decision": {
            "id": "decision_cloud_tier",
            "label": "standard cloud hosting tier selection",
            "text": "To stay strictly within our $1,500 monthly cloud budget, we selected the Standard 4-node cluster plan.",
            "stands": True,
            "reason": "The cloud infrastructure budget ($1,500) was not changed.",
        },
        "questions": {
            "revisit": "Which decisions should be revisited or rescheduled following the latest project updates?",
            "why": "Why does the paid launch ad campaign decision need to be revisited?",
            "stands": "Does the cloud hosting tier selection still stand or does it need to be revisited?",
        },
    },
    {
        "domain": "personal_finance",
        "assumption_1": "We are receiving an annual bonus of $10,000 in December.",
        "assumption_1_revised": "The company announced bonuses are completely cancelled this year.",
        "affected_decisions": [
            {
                "id": "decision_ski_vacation",
                "label": "luxury Aspen ski trip booking",
                "text": "Counting on the $10,000 December bonus, I put down a deposit for a luxury Aspen ski vacation for New Year's.",
                "reason": "The ski trip was funded by the expected $10,000 bonus.",
            },
            {
                "id": "decision_hometheater_upgrade",
                "label": "surround sound home theater setup purchase",
                "text": "Relying on the upcoming year-end bonus, I pre-ordered the $2,500 surround sound setup.",
                "reason": "The home theater setup was planned using the bonus funds.",
            },
        ],
        "assumption_2": "Our monthly mortgage interest rate is fixed at 3.2%.",
        "unaffected_decision": {
            "id": "decision_mortgage_autopay",
            "label": "monthly mortgage autopay amount",
            "text": "Given our fixed 3.2% mortgage interest rate, we set our monthly payment autopay to exactly $2,100.",
            "stands": True,
            "reason": "The fixed mortgage interest rate remains unchanged at 3.2%.",
        },
        "questions": {
            "revisit": "Which financial commitments or purchases should be revisited or cancelled given recent news?",
            "why": "Why does the Aspen ski trip decision need to be revisited?",
            "stands": "Does the monthly mortgage autopay amount still stand?",
        },
    },
    {
        "domain": "home_renovation",
        "assumption_1": "The kitchen cabinet delivery date is confirmed for July 10th.",
        "assumption_1_revised": "The cabinet maker notified us that delivery is delayed until August 25th.",
        "affected_decisions": [
            {
                "id": "decision_countertop_template",
                "label": "countertop templating measurement appointment",
                "text": "Since the cabinets will be installed by July 12th, I booked the countertop templating crew for July 14th.",
                "reason": "Countertop templating requires lower cabinets to already be installed in place.",
            },
            {
                "id": "decision_plumber_hookup",
                "label": "sink and dishwasher plumbing connection appointment",
                "text": "Assuming cabinets are delivered July 10th, I reserved the finish plumber for July 16th.",
                "reason": "Plumbing hookups depend on cabinet and sink installation.",
            },
        ],
        "assumption_2": "The living room hardwood floor contractor has availability in June.",
        "unaffected_decision": {
            "id": "decision_hardwood_installation",
            "label": "living room hardwood refinishing schedule",
            "text": "Because the flooring crew is free in June, we scheduled the living room hardwood refinishing for June 20th.",
            "stands": True,
            "reason": "Living room hardwood refinishing is in a separate room and independent of kitchen cabinet delivery.",
        },
        "questions": {
            "revisit": "Which scheduled contractor appointments need to be rescheduled or revisited?",
            "why": "Why does the countertop templating appointment need to be revisited?",
            "stands": "Does the living room hardwood refinishing schedule still stand?",
        },
    },
]

# Track X (Hidden Constraint) Templates
# A hard constraint is stated in an earlier session. A planning question is asked
# later without mentioning the constraint. The answer must incorporate the constraint.
HIDDEN_CONSTRAINT_POOLS: List[Dict[str, Any]] = [
    {
        "domain": "health_and_fitness",
        "constraint_statement": "My physical therapist explicitly instructed me to strictly avoid any high-impact exercises or running because of my knee meniscus tear.",
        "query": "Can you design a high-energy morning cardio workout routine for me this week including sprint intervals and jump rope?",
        "gold_response_requirement": "The workout plan must refuse sprint intervals and jump rope, citing the strict medical restriction against high-impact exercises and running due to the meniscus tear, and substitute low-impact cardio like swimming, rowing, or cycling.",
        "constraint_keyword": "meniscus",
        "gold_verdict": "rejected_or_modified_low_impact",
    },
    {
        "domain": "personal_finance",
        "constraint_statement": "My company travel policy strictly caps individual hotel room bookings at $220 per night, requiring formal executive exception approval for anything higher.",
        "query": "I am traveling to the Chicago conference and found the Hyatt Regency right next door for $340 per night. Should I go ahead and book it directly on the corporate portal?",
        "gold_response_requirement": "No, you cannot book it directly because it exceeds the $220/night corporate limit; it requires formal executive exception approval.",
        "constraint_keyword": "$220",
        "gold_verdict": "cannot_book_directly_requires_exception",
    },
    {
        "domain": "software_engineering",
        "constraint_statement": "Our staging Kubernetes cluster has a strict hard resource limit of at most 8 vCPUs total across all running pods.",
        "query": "The backend team wants to deploy a new distributed load-testing test suite on staging that requests 12 vCPUs. Can we spin that up alongside the existing staging pods?",
        "gold_response_requirement": "No, it cannot be spun up because the 12 vCPU request exceeds the staging cluster's strict hard limit of 8 vCPUs total.",
        "constraint_keyword": "8 vCPUs",
        "gold_verdict": "exceeds_staging_resource_limit",
    },
]

# Track F (Forecasts & Patterns) Templates
# Past milestones slipped multiple times. When evaluating a new target,
# the system must compute slips and recognize unrealistic optimism.
FORECAST_POOLS: List[Dict[str, Any]] = [
    {
        "domain": "software_engineering",
        "project": "Project Titan mobile app",
        "events": [
            {"date": "2024-01-10", "text": "We committed to completing the Alpha release by February 1st."},
            {"date": "2024-01-25", "text": "We ran into architecture roadblocks; Alpha release target is pushed to March 1st."},
            {"date": "2024-02-20", "text": "Third-party SDK integration failed; Alpha release date slipped again to March 25th."},
            {"date": "2024-03-25", "text": "Alpha release was finally completed and deployed today."},
        ],
        "move_count": 2,
        "total_slip_days": 52,  # Feb 1 to March 25 = 52 days
        "new_target_query": "The team just announced that the much larger Beta release will be fully delivered in just 7 calendar days. Given our past delivery history, is this 7-day target realistic?",
        "gold_answers": {
            "move_count": 2,
            "total_slip_days": 52,
            "realistic": "No, unrealistic. The team repeatedly slipped the smaller Alpha release by 52 days across 2 delays, making a 7-day Beta delivery unrealistic.",
        },
    },
    {
        "domain": "home_renovation",
        "project": "Master bathroom remodel",
        "events": [
            {"date": "2024-03-01", "text": "Contractor promised the bathroom tile installation would be finished by March 20th."},
            {"date": "2024-03-18", "text": "Tiles were backordered; contractor moved the completion date to April 10th."},
            {"date": "2024-04-05", "text": "Waterproofing failed inspection; completion date was pushed to April 25th."},
            {"date": "2024-04-22", "text": "Plumber had COVID; tile and fixtures moved to May 10th."},
            {"date": "2024-05-10", "text": "Bathroom tiling was finally finished today."},
        ],
        "move_count": 3,
        "total_slip_days": 51,  # March 20 to May 10 = 51 days
        "new_target_query": "The contractor says the entire kitchen remodel will be completed in 5 days. Based on the bathroom project history, is this estimate realistic?",
        "gold_answers": {
            "move_count": 3,
            "total_slip_days": 51,
            "realistic": "No, unrealistic. The bathroom remodel slipped 3 times and took 51 extra days beyond the original target.",
        },
    },
]

# Track I (Interventions & Causal Claims) Templates
# 4 distinct evidentiary archetypes:
# 1. Clean: comparison / reversal exists, no confound. Causal claim valid.
# 2. Confounded: concurrent confounding change. Causal claim unproven.
# 3. Not implemented: intervention discussed/planned but never carried out. Causal claim false.
# 4. Single observation: single un-replicated instance. Anecdotal, causal claim premature.
INTERVENTION_POOLS: List[Dict[str, Any]] = [
    {
        "variant": "clean",
        "domain": "health_and_fitness",
        "intervention": "drinking tart cherry juice before bed",
        "outcome": "deep sleep duration",
        "timeline": [
            "Baseline: My tracker showed average deep sleep of 42 minutes per night for 3 weeks.",
            "Intervention: I started drinking 8oz of tart cherry juice every night. After 2 weeks, deep sleep rose to 74 minutes per night.",
            "Reversal: I ran out of cherry juice for 10 days, and deep sleep dropped back down to 45 minutes.",
            "Re-introduction: I bought more juice, and deep sleep rose back to 76 minutes. My bedtime and workout routine stayed unchanged.",
        ],
        "query": "Did drinking tart cherry juice improve the user's deep sleep duration?",
        "gold_strength": "supported",
        "gold_rubric": "Valid causal attribution: The evidence demonstrates a clear baseline, an increase upon intervention, a return to baseline upon withdrawal, and re-elevation upon reintroduction with no confounding variables.",
    },
    {
        "variant": "confounded",
        "domain": "software_engineering",
        "intervention": "switching to HTTP/3",
        "outcome": "p99 API latency reduction",
        "timeline": [
            "We were experiencing high p99 latency (450ms) on our user feed endpoint.",
            "On June 1st, we enabled HTTP/3 support on the load balancers.",
            "On that exact same morning, our database admin doubled the database server RAM and added missing composite indexes on the feed table.",
            "Feed latency dropped to 120ms.",
        ],
        "query": "Did enabling HTTP/3 cause the reduction in p99 feed latency?",
        "gold_strength": "confounded",
        "gold_rubric": "Confounded / Unsupported causal claim: The latency reduction cannot be attributed solely or confidently to HTTP/3 because the database administrator concurrently doubled RAM and added composite indexes on the exact same morning.",
    },
    {
        "variant": "not_implemented",
        "domain": "personal_finance",
        "intervention": "cancelling unused streaming subscriptions",
        "outcome": "monthly credit card balance drop",
        "timeline": [
            "I noticed my monthly credit card bill was very high ($3,400) and decided I should cancel my 5 unused video streaming services.",
            "However, I was too busy with work and forgot to log in to cancel any of the subscriptions; all 5 billed as usual.",
            "The following month, my credit card bill dropped to $1,800 because I didn't have to buy annual car insurance like I did the previous month.",
        ],
        "query": "Did cancelling unused streaming subscriptions reduce the user's credit card bill?",
        "gold_strength": "not_implemented",
        "gold_rubric": "False / Not implemented: The user never actually cancelled the streaming subscriptions; the bill decreased due to the absence of the annual car insurance expense.",
    },
    {
        "variant": "single_observation",
        "domain": "health_and_fitness",
        "intervention": "taking cold showers in the morning",
        "outcome": "running personal record",
        "timeline": [
            "On Wednesday morning, I took a 3-minute cold shower for the very first time in my life.",
            "That afternoon, I went for a 5k run and beat my previous personal record by 15 seconds.",
            "I haven't taken another cold shower since then.",
        ],
        "query": "Does taking cold showers improve the user's running speed?",
        "gold_strength": "single_observation_insufficient",
        "gold_rubric": "Insufficient evidence / Anecdotal: A single observation on one day without replication, control, or ongoing measurement cannot substantiate a causal relationship.",
    },
]

# Track P (Provenance & Misinformation Rejection)
# Evaluates whether the memory system rejects assistant-asserted misinformation
# or user-retracted statements in favor of authentic user facts.
PROVENANCE_POOLS: List[Dict[str, Any]] = [
    {
        "variant": "assistant_unconfirmed_hallucination",
        "domain": "personal_facts",
        "timeline": [
            {"role": "user", "content": "I'm looking for a nice restaurant to celebrate my birthday this Friday."},
            {"role": "assistant", "content": "Since you mentioned you are vegetarian and love spicy Sichuan food, I recommend Spicy Grand Palace!"},
            {"role": "user", "content": "Wait, I eat meat and I really can't handle spicy food at all. Let's find an Italian steakhouse instead."},
            {"role": "assistant", "content": "Understood, here are three great Italian steakhouses downtown."},
        ],
        "query": "What type of food does the user prefer for dining out?",
        "gold_value": "Italian steakhouse / meat (dislikes spicy food, is not vegetarian)",
        "forbidden_distractor": "vegetarian / spicy Sichuan food",
    },
    {
        "variant": "user_retraction",
        "domain": "personal_facts",
        "timeline": [
            {"role": "user", "content": "I got a job offer in Seattle, so I will be relocating there at the end of August."},
            {"role": "assistant", "content": "Congratulations on the job offer! Let me know if you need help finding apartments in Seattle."},
            {"role": "user", "content": "Actually, scratch that—the Seattle company had a hiring freeze and revoked the offer. I signed a new lease to stay in Chicago."},
            {"role": "assistant", "content": "I'm sorry about the freeze, but glad you have your Chicago lease sorted out."},
        ],
        "query": "Where is the user currently planning to live?",
        "gold_value": "Chicago (the Seattle relocation was revoked/cancelled)",
        "forbidden_distractor": "Seattle",
    },
]

# Track K (Recall Control - Static Facts)
# Facts stated once or twice that never change. All systems expected to tie.
RECALL_CONTROL_POOLS: List[Dict[str, Any]] = [
    {
        "domain": "personal_background",
        "entity": "hometown",
        "statement": "I was born and grew up in Madison, Wisconsin before moving away for college.",
        "query": "What is the user's hometown?",
        "gold_value": "Madison, Wisconsin",
    },
    {
        "domain": "personal_background",
        "entity": "pet name",
        "statement": "My rescue golden retriever is named Barnaby, he just turned 4 years old.",
        "query": "What is the name of the user's dog?",
        "gold_value": "Barnaby",
    },
    {
        "domain": "personal_background",
        "entity": "university degree",
        "statement": "I completed my undergraduate degree in Mechanical Engineering at Purdue University.",
        "query": "What university did the user attend for undergraduate studies, and what was their major?",
        "gold_value": "Purdue University, Mechanical Engineering",
    },
]
