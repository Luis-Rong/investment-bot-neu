from pydantic import BaseModel


class UserProfile(BaseModel):
    goal: str | None = None
    horizon_years: int | None = None
    risk: int | None = None
    esg: bool | None = None
    saving_eur: int | None = None
    monthly_saving_eur: int | None = None
    experience: str | None = None  # low/medium/high
    liquidity_need: str | None = None  # low/medium/high


def missing_fields(profile: UserProfile):
    required = ["goal", "horizon_years", "risk", "esg", "saving_eur", "monthly_saving_eur"]
    missing = []
    for field in required:
        if getattr(profile, field) is None:
            missing.append(field)
    return missing
