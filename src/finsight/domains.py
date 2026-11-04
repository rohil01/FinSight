"""Domain profiles for regulatory, market, and general research."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    """Search scope and prompt guidance for one research domain."""

    name: str
    include_domains: tuple[str, ...] | None
    planner_guidance: str
    critic_guidance: str
    writer_guidance: str


REGULATORY = DomainProfile(
    name="regulatory",
    include_domains=(
        "rbi.org.in",
        "npci.org.in",
        "sebi.gov.in",
        "pib.gov.in",
    ),
    planner_guidance=(
        "Across the sub-questions, cover the operative circular or master-direction text, "
        "effective dates, affected entities (payment aggregators, payment gateways, and "
        "prepaid payment instruments), and compliance timelines."
    ),
    critic_guidance=(
        "Treat the research as incomplete when it lacks the circular or master direction, "
        "its effective date, the entities it binds, or the compliance timeline."
    ),
    writer_guidance=(
        "6. Regulatory Implications\n"
        "7. Compliance Timeline\n\n"
        "For every regulatory claim, cite the circular or master direction by number and "
        "date when the notes contain them. Name the affected entities (payment aggregators, "
        "payment gateways, prepaid payment instruments) when they are in scope."
    ),
)

MARKET = DomainProfile(
    name="market",
    include_domains=(
        "livemint.com",
        "economictimes.indiatimes.com",
        "moneycontrol.com",
        "medianama.com",
        "inc42.com",
        "entrackr.com",
    ),
    planner_guidance=(
        "Cover volumes, market share, the companies involved, and the dates those figures refer to."
    ),
    critic_guidance=(
        "Treat the research as incomplete when market shares, volumes, or the companies "
        "involved are missing, or when figures have no date."
    ),
    writer_guidance=(
        "6. Market structure (players, shares, volumes)\n"
        "7. What changed, with dates\n\n"
        "Use dated figures from the notes. Do not invent statistics that are not in the research."
    ),
)

GENERAL = DomainProfile(
    name="general",
    include_domains=None,
    planner_guidance="",
    critic_guidance="",
    writer_guidance="",
)

PROFILES = {
    REGULATORY.name: REGULATORY,
    MARKET.name: MARKET,
    GENERAL.name: GENERAL,
}

DOMAINS = tuple(PROFILES)


def get_profile(domain: str) -> DomainProfile:
    """Return the profile for a domain name, or raise ValueError."""
    try:
        return PROFILES[domain]
    except KeyError as exc:
        known = ", ".join(DOMAINS)
        raise ValueError(f"Unknown domain {domain!r}. Expected one of: {known}.") from exc
