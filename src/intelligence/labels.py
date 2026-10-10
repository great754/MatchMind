"""Display identifiers without treating tracker IDs as jersey numbers or names."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerLabel:
    tracker_id: int
    jersey_number: int | None = None
    player_name: str | None = None
    identity_verified: bool = False
    verification_source: str | None = None

    def __post_init__(self):
        if self.identity_verified and not self.verification_source:
            raise ValueError('Verified identities require a verification source')
        if self.jersey_number is not None and not 0 <= self.jersey_number <= 99:
            raise ValueError('Invalid jersey number')

    @property
    def display(self) -> str:
        if self.identity_verified and self.player_name:
            return self.player_name
        if self.identity_verified and self.jersey_number is not None:
            return f'#{self.jersey_number}'
        return f'Player {self.tracker_id}'


def recording_labels(recording) -> dict[int,PlayerLabel]:
    """Do not auto-promote provider actor IDs or metadata jerseys into verified identities."""
    return {int(actor):PlayerLabel(recording.mot_ids.get(int(actor),int(actor))) for actor in recording.player_ids}
